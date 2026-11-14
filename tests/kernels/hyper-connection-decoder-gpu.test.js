import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { runMatmul, recordMatmul } from '../../src/gpu/kernels/matmul.js';
import { resolveQwen4ExpSourceContracts } from '../../src/converter/qwen4-exp-source.js';
import { resolveHyperConnectionsContract } from '../../src/config/schema/hyper-connection-decoder.schema.js';
import { resolveNgramEmbeddingContract } from '../../src/config/schema/ngram-embedding.schema.js';
import { NgramTokenHistory } from '../../src/inference/pipelines/text/ngram/ngram-addresses.js';
import { PleConvState, runPleLayer } from '../../src/inference/pipelines/text/ngram/ple-layer.js';
import { replicateEmbeddings, runHyperConnectionDecoderLayer, runRootMixer } from '../../src/inference/pipelines/text/hyper-connection/decoder-layer.js';
import * as ref from './hyper-connection-decoder-reference.js';

// Contract: Doppler's decoder composition (entry replication, PLE before the
// attention hyper-connection, two gated residuals with injection per layer,
// root mixer instead of a final norm) reproduces the pinned upstream
// Qwen4ExpTextDecoderLayer stack for layers 0-3 within a propagated float64
// F32 bound, full and chunked with carried PLE state, immediate and recorded.
// Branches are the fixture's low-rank linear spies (composition order only).
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical decoder composition check unavailable: ${probe.reason}`);
const capabilities = getKernelCapabilities();
assert.ok(!/swiftshader|llvmpipe|lavapipe/.test(JSON.stringify(capabilities?.adapterInfo ?? {}).toLowerCase()));
const prep = '../../artifacts/qwen38-flash-next-20261013/';
const readJson = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const fixture = await readJson(`${prep}decoder-composition-reference.json`);
const flashNext = await readJson(`${prep}source/config.json`);
const source = resolveQwen4ExpSourceContracts(flashNext);
const hyper = { ...resolveHyperConnectionsContract(source.hyperConnections, { hiddenSize: 2560 }), normWeightOffset: true };
const ngram = resolveNgramEmbeddingContract(source.ngramEmbedding,
  { hiddenSize: 2560, numLayers: 48, vocabSize: 248320, layerTypes: flashNext.text_config.layer_types });
const T = fixture.tokens.length;
const S = hyper.streamCount;
const W = hyper.streamWidth;
const C = S * W;
const R = hyper.projectionRank;
const RANK = fixture.branchRank;

const MASK = (1n << 64n) - 1n;
const splitmix = (z) => {
  z = (z + 0x9E3779B97F4A7C15n) & MASK;
  z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & MASK;
  z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & MASK;
  return z ^ (z >> 31n);
};
const dyadic = (count, stream, exponent) => {
  const out = new Float32Array(count);
  const base = BigInt(stream) << 40n;
  for (let i = 0; i < count; i++) out[i] = (Number(splitmix(base + BigInt(i)) >> 40n) - 2 ** 23) * 2 ** (exponent - 23);
  return out;
};
const params = new Map(fixture.recipe.parameters.map((p) => [p.name, dyadic(p.shape.reduce((a, b) => a * b, 1), p.stream, p.exponent)]));
const param = (name) => {
  const value = params.get(name);
  assert.ok(value, `missing parameter ${name}`);
  return value;
};
const embeds = dyadic(T * W, fixture.recipe.embeds.stream, fixture.recipe.embeds.exponent);
const history = new NgramTokenHistory(ngram, ngram.layers[0]);
const addresses = history.append(fixture.tokens);
const E = ngram.embedDim;
const rowBits = new Uint16Array(T * E);
const embedding = new Float64Array(T * E);
const rowBase = BigInt(fixture.recipe.rows.stream) << 40n;
for (let i = 0; i < addresses.length; i++) {
  for (let d = 0; d < ngram.headDim; d++) {
    const k = Number(splitmix(rowBase + BigInt(addresses[i]) * BigInt(ngram.headDim) + BigInt(d)) >> 56n) - 128;
    const value = k * 2 ** (fixture.recipe.rows.exponent - 7);
    embedding[i * ngram.headDim + d] = value;
    rowBits[i * ngram.headDim + d] = new Uint32Array(Float32Array.of(value).buffer)[0] >>> 16;
  }
}
const branchName = (layer) => (flashNext.text_config.layer_types[layer] === 'full_attention' ? 'self_attn' : 'linear_attn');
const hcWeights = (prefix) => ({
  norm: param(`${prefix}.hc_norm.weight`), down: param(`${prefix}.input_mix_weight_down.weight`),
  up: param(`${prefix}.input_mix_weight_up.weight`), inject: params.get(`${prefix}.block_inject_weight.weight`) ?? null,
});

// Float64 reference of one layer from exact F32 inputs (layerwise: bounds do not compound).
const geometry = { rows: T, streamCount: S, width: W, rank: R, eps: hyper.normEps };
const pleParams = (() => {
  const p = `layers.${ngram.layers[0].decoderLayerIndex}.ple`;
  return { key: param(`${p}.key_proj.weight`), value: param(`${p}.value_proj.weight`), normKey: param(`${p}.norm_key.weight`),
    normQuery: param(`${p}.norm_query.weight`), normConv: param(`${p}.norm_conv.weight`), conv: param(`${p}.conv1d.weight`) };
})();
const spy = (x, name, rows) => ref.linear(ref.linear(x, param(`${name}.A`), rows, W, RANK, false), param(`${name}.B`), rows, RANK, W, false);
// The PLE convolution needs the whole prefix; the layerwise reference runs on full sequences only.
function referenceLayer(layer, input, { pleAfterAttention = false } = {}) {
  let streams = ref.exact(input);
  const runPle = (x) => ref.ple(x, embedding, pleParams, { ...geometry, embedDim: E, kernel: ngram.convKernelSize, dilation: ngram.convDilation });
  const isPle = layer === ngram.layers[0].decoderLayerIndex;
  if (isPle && !pleAfterAttention) streams = runPle(streams);
  const steps = [['attn_hyper_connection', `layers.${layer}.${branchName(layer)}`], ['mlp_hyper_connection', `layers.${layer}.mlp`]];
  steps.forEach(([role, branch], index) => {
    const { branch: x, gate } = ref.gatedResidual(streams, hcWeights(`layers.${layer}.${role}`), geometry);
    streams = ref.inject(streams, spy(x, branch, T), gate, geometry);
    if (isPle && pleAfterAttention && index === 0) streams = runPle(streams);
  });
  return streams;
}
const results = { runs: [] };

const owned = new Set();
const upload = (data, shape) => {
  const buffer = acquireBuffer(data.byteLength);
  owned.add(buffer);
  uploadData(buffer, data);
  return shape ? createTensor(buffer, 'f32', shape) : buffer;
};
const states = [];
try {
  const gpu = new Map([...params].map(([name, values]) => [name, upload(values,
    name.endsWith('hc_norm.weight') || name.includes('.norm_') ? [values.length] : null)]));
  const gpuHc = (prefix) => ({ norm: gpu.get(`${prefix}.hc_norm.weight`), down: gpu.get(`${prefix}.input_mix_weight_down.weight`),
    up: gpu.get(`${prefix}.input_mix_weight_up.weight`), inject: gpu.get(`${prefix}.block_inject_weight.weight`) ?? null });
  const p = `layers.${ngram.layers[0].decoderLayerIndex}.ple`;
  const pleWeights = { keyProj: gpu.get(`${p}.key_proj.weight`), valueProj: gpu.get(`${p}.value_proj.weight`),
    normKey: gpu.get(`${p}.norm_key.weight`), normQuery: gpu.get(`${p}.norm_query.weight`), normConv: gpu.get(`${p}.norm_conv.weight`),
    conv: createTensor(gpu.get(`${p}.conv1d.weight`), 'f32', [C * ngram.convKernelSize]) };
  for (const [mode, chunks] of [['immediate', [T]], ['immediate', fixture.chunks], ['recorded', fixture.chunks]]) {
    const recorder = mode === 'recorded' ? new CommandRecorder(getDevice(), 'hc_decoder_check') : null;
    const matmul = recorder ? recordMatmul.bind(null, recorder) : runMatmul;
    const state = new PleConvState({ channels: C, stateLength: ngram.convStateLength });
    states.push(state);
    const outputs = [];
    let start = 0;
    for (const size of chunks) {
      const project = (role, input, weight, outF, inF) => matmul(input, weight, size, outF, inF, { transposeB: true, outputDtype: 'f32', bDtype: 'f32' });
      const spyBranch = (name) => async (x) => {
        const low = await matmul(x, gpu.get(`${name}.A`), size, RANK, W, { transposeB: false, outputDtype: 'f32', bDtype: 'f32' });
        try {
          return await matmul(low, gpu.get(`${name}.B`), size, W, RANK, { transposeB: false, outputDtype: 'f32', bDtype: 'f32' });
        } finally {
          if (recorder) recorder.trackTemporaryBuffer(low.buffer); else releaseBuffer(low.buffer);
        }
      };
      let current = await replicateEmbeddings(recorder, upload(embeds.subarray(start * W, (start + size) * W), [size, W]), { hyper, numTokens: size });
      for (const layer of fixture.layers) {
        const isPle = layer === ngram.layers[0].decoderLayerIndex;
        const rows = isPle ? upload(rowBits.subarray(start * E, (start + size) * E)) : null;
        const next = await runHyperConnectionDecoderLayer(recorder, {
          streams: current, numTokens: size, hyper, project,
          weights: { attention: gpuHc(`layers.${layer}.attn_hyper_connection`), mlp: gpuHc(`layers.${layer}.mlp_hyper_connection`) },
          ple: isPle ? async (input) => {
            const result = await runPleLayer(recorder, { streams: input, rows, weights: pleWeights, state, ngram, hyper, numTokens: size,
              project: (role, x, weight, outF, inF) => matmul(x, weight, size, outF, inF, { transposeB: true, outputDtype: 'f32', bDtype: 'f32' }) });
            if (recorder) recorder.trackTemporaryBuffer(result.gateInput.buffer); else releaseBuffer(result.gateInput.buffer);
            return result.streams;
          } : null,
          attentionBranch: spyBranch(`layers.${layer}.${branchName(layer)}`),
          mlpBranch: spyBranch(`layers.${layer}.mlp`),
        });
        outputs.push({ layer, start, size, tensor: next });
        owned.add(next.buffer);
        current = next;
      }
      const mixed = await runRootMixer(recorder, { streams: current, weights: gpuHc('mixer'), hyper, numTokens: size, project });
      owned.add(mixed.buffer);
      outputs.push({ layer: 'mixed', start, size, tensor: mixed });
      start += size;
    }
    if (recorder) await recorder.submitAndWait();
    const run = { mode, chunks, layers: {}, mixed: 0, upstreamMixedMaxAbs: 0 };
    const read = async (tensor, size, width) => new Float32Array(await readBuffer(tensor.buffer, size * width * 4));
    // Assemble full-sequence outputs per layer from the chunks.
    const assembled = {};
    for (const { layer, start: s0, size, tensor } of outputs) {
      const width = layer === 'mixed' ? W : C;
      assembled[layer] ??= new Float32Array(T * width);
      assembled[layer].set(await read(tensor, size, width), s0 * width);
    }
    let input = Float32Array.from({ length: T * C }, (_, i) => embeds[Math.floor(i / C) * W + (i % W)]);
    for (const layer of fixture.layers) {
      run.layers[layer] = ref.boundRatio(assembled[layer], referenceLayer(layer, input));
      input = assembled[layer];
    }
    const mixedRef = ref.gatedResidual(ref.exact(input), hcWeights('mixer'), geometry).branch;
    run.mixed = ref.boundRatio(assembled.mixed, mixedRef);
    fixture.mixedOutput.forEach((row, t) => row.forEach((v, j) => {
      run.upstreamMixedMaxAbs = Math.max(run.upstreamMixedMaxAbs, Math.abs(v - assembled.mixed[t * W + j * fixture.outputChannels.stride]));
    }));
    assert.ok(run.mixed <= 1 && Object.values(run.layers).every((r) => r <= 1), `${mode} ${chunks}: ${JSON.stringify(run)}`);
    if (mode === 'immediate' && chunks.length === 1) {
      // Discrimination at the same depth: PLE after the attention injection, and a plain stream mean instead of the mixer.
      const pleLayer = ngram.layers[0].decoderLayerIndex;
      const previous = pleLayer === fixture.layers[0] ? Float32Array.from({ length: T * C }, (_, i) => embeds[Math.floor(i / C) * W + (i % W)])
        : assembled[fixture.layers[fixture.layers.indexOf(pleLayer) - 1]];
      const right = referenceLayer(pleLayer, previous);
      const wrong = referenceLayer(pleLayer, previous, { pleAfterAttention: true });
      // The rigorous PLE bound is loose near the signed square root (derivative floor 500), so order is
      // discriminated by distance: the GPU must sit far closer to the pinned order than the wrong order does.
      let toRight = 0;
      let wrongToRight = 0;
      const gpuPle = assembled[pleLayer];
      for (let i = 0; i < right.v.length; i++) {
        toRight = Math.max(toRight, Math.abs(gpuPle[i] - right.v[i]));
        wrongToRight = Math.max(wrongToRight, Math.abs(wrong.v[i] - right.v[i]));
      }
      results.pleOrderSeparation = wrongToRight / toRight;
      const mean = new Float32Array(T * W);
      for (let i = 0; i < T * W; i++) {
        const t = Math.floor(i / W);
        for (let s = 0; s < S; s++) mean[i] += input[(t * S + s) * W + (i % W)] / S;
      }
      results.mixerSeparation = ref.boundRatio(mean, mixedRef);
      assert.ok(results.pleOrderSeparation > 1000 && results.mixerSeparation > 2, `alternative compositions are distinguished: ${JSON.stringify({ ple: results.pleOrderSeparation, mixer: results.mixerSeparation, layers: run.layers, mixed: run.mixed })}`);
    }
    results.runs.push(run);
  }
} finally {
  for (const state of states) await state.destroy();
  await getDevice()?.queue.onSubmittedWorkDone();
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice();
}
console.log(JSON.stringify({ test: 'hyper-connection-decoder-gpu', adapter: capabilities?.adapterInfo ?? null, results,
  scope: 'Physical decoder composition of layers 0-3 and the root mixer with spy branches; no learned weights or timing' }));
