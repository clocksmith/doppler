import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { runMatmul, recordMatmul } from '../../src/gpu/kernels/matmul.js';
import { resolveQwen4ExpSourceContracts } from '../../src/converter/qwen4-exp-source.js';
import { resolveHyperConnectionDecoderContracts } from '../../src/config/schema/hyper-connection-decoder.schema.js';
import { NgramTokenHistory } from '../../src/inference/pipelines/text/ngram/ngram-addresses.js';
import { PleConvState, runPleLayer } from '../../src/inference/pipelines/text/ngram/ple-layer.js';

// Contract: the GPU PLE layer reproduces float64 truth of the pinned upstream
// Qwen4ExpTextPLELayer within a first-order F32 bound, full-sequence and
// chunked with carried convolution state, immediate and recorded; the upstream
// torch outputs lie within the same bound. Addresses match upstream exactly.
// Synthetic dyadic weights and BF16-exact spy rows; not learned-weight evidence.
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical PLE check unavailable: ${probe.reason}`);
const capabilities = getKernelCapabilities();
assert.ok(!/swiftshader|llvmpipe|lavapipe/.test(JSON.stringify(capabilities?.adapterInfo ?? {}).toLowerCase()));
const prep = '../../artifacts/qwen38-flash-next-20261013/';
const readJson = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const fixture = await readJson(`${prep}ple-reference.json`);
const flashNext = await readJson(`${prep}source/config.json`);
const source = resolveQwen4ExpSourceContracts(flashNext);
const contracts = resolveHyperConnectionDecoderContracts({ ...source,
  session: { ngramEmbedding: { rowCache: { maxRows: 0, maxBytes: 0, eviction: 'lru' }, staging: { maxRows: 16 }, reads: { maxConcurrent: 1 } } },
  normalization: { postAttentionNorm: false, preFeedforwardNorm: false, postFeedforwardNorm: false }, layerPattern: { residualBranchScale: 1 } },
{ hiddenSize: 2560, numLayers: 48, vocabSize: 248320, layerTypes: flashNext.text_config.layer_types, attentionRotaryDim: 64 });
const ngram = contracts.ngramEmbedding;
const hyper = contracts.hyperConnections;
const [layer] = ngram.layers;
const T = fixture.tokens.length;
const S = hyper.streamCount;
const W = hyper.streamWidth;
const C = S * W;
const E = ngram.embedDim;
const K = ngram.convKernelSize;

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
const recipe = Object.fromEntries(fixture.recipe.weights.map((w) => [w.name, w]));
const weight = (name) => dyadic(recipe[name].shape.reduce((a, b) => a * b, 1), recipe[name].stream, recipe[name].exponent);
const w = {
  key: weight('key_proj.weight'), value: weight('value_proj.weight'), normKey: weight('norm_key.weight'),
  normQuery: weight('norm_query.weight'), normConv: weight('norm_conv.weight'), conv: weight('conv1d.weight'),
};
const hidden = dyadic(T * C, fixture.recipe.hidden.stream, fixture.recipe.hidden.exponent);

// Exact addresses, then BF16-exact rows by address.
const addresses = new NgramTokenHistory(ngram, layer).append(fixture.tokens);
assert.deepEqual([...addresses], fixture.addresses.flat());
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

// Float64 reference with first-order F32 bounds (any summation order).
const u = 2 ** -24;
const gamma = (n) => (n * u) / (1 - n * u);
function linear(x, ex, weights, rows, inF, outF) {
  const v = new Float64Array(rows * outF);
  const e = new Float64Array(rows * outF);
  for (let t = 0; t < rows; t++) {
    for (let n = 0; n < outF; n++) {
      let sum = 0; let abs = 0; let prop = 0;
      for (let k = 0; k < inF; k++) {
        const a = x[t * inF + k]; const b = weights[n * inF + k];
        sum += a * b; abs += Math.abs(a * b); prop += Math.abs(b) * ex[t * inF + k];
      }
      v[t * outF + n] = sum;
      e[t * outF + n] = gamma(inF) * (abs + prop) + prop;
    }
  }
  return { v, e };
}
function groupedNorm(x, ex, weights, rows) {
  const v = new Float64Array(rows * C);
  const e = new Float64Array(rows * C);
  const rounding = 0.5 * (gamma(W + 1) + 2 * u) + 5 * u;
  for (let g = 0; g < rows * S; g++) {
    const base = g * W;
    const wb = (g % S) * W;
    let sq = 0; let rel = 0;
    for (let i = 0; i < W; i++) { sq += x[base + i] ** 2; rel += Math.abs(x[base + i]) * ex[base + i]; }
    const r = 1 / Math.sqrt(sq / W + ngram.normEps);
    for (let i = 0; i < W; i++) {
      const y = x[base + i] * r * (1 + weights[wb + i]);
      v[base + i] = y;
      e[base + i] = Math.abs(1 + weights[wb + i]) * r * (ex[base + i] + Math.abs(x[base + i]) * rel / sq) + Math.abs(y) * rounding;
    }
  }
  return { v, e };
}
const zeros = (n) => new Float64Array(n);
const key = groupedNorm(...Object.values(linear(embedding, zeros(T * E), w.key, T, E, C)), w.normKey, T);
const value = linear(embedding, zeros(T * E), w.value, T, E, W);
const query = groupedNorm(Float64Array.from(hidden), zeros(T * C), w.normQuery, T);
const divisor = Math.fround(Math.sqrt(W));
const gate = { v: new Float64Array(T * S), e: new Float64Array(T * S) };
const gated = { v: new Float64Array(T * C), e: new Float64Array(T * C) };
for (let t = 0; t < T; t++) {
  for (let s = 0; s < S; s++) {
    let g = 0; let abs = 0; let prop = 0;
    for (let d = 0; d < W; d++) {
      const i = t * C + s * W + d;
      g += key.v[i] * query.v[i]; abs += Math.abs(key.v[i] * query.v[i]);
      prop += Math.abs(query.v[i]) * key.e[i] + Math.abs(key.v[i]) * query.e[i];
    }
    g /= divisor;
    const eg = (gamma(W + 1) * abs + prop) / divisor + u * Math.abs(g);
    const gp = Math.sign(g) * Math.sqrt(Math.max(Math.abs(g), 1e-6));
    const egp = eg / (2 * Math.sqrt(Math.max(Math.abs(g) - eg, 1e-6))) + 3 * u * Math.abs(gp);
    const sig = 1 / (1 + Math.exp(-gp));
    const es = 0.25 * egp + 8 * u * sig;
    gate.v[t * S + s] = gp;
    gate.e[t * S + s] = egp;
    for (let d = 0; d < W; d++) {
      const i = t * C + s * W + d;
      const vv = value.v[t * W + d];
      gated.v[i] = sig * vv;
      gated.e[i] = Math.abs(vv) * es + sig * value.e[t * W + d] + u * Math.abs(sig * vv);
    }
  }
}
const normed = groupedNorm(gated.v, gated.e, w.normConv, T);
const output = { v: new Float64Array(T * C), e: new Float64Array(T * C) };
const stateLength = ngram.convStateLength;
for (let t = 0; t < T; t++) {
  for (let c = 0; c < C; c++) {
    let acc = 0; let abs = 0; let prop = 0;
    for (let k = 0; k < K; k++) {
      const src = t - stateLength + k * ngram.convDilation;
      if (src < 0) continue;
      const x = normed.v[src * C + c]; const wk = w.conv[c * K + k];
      acc += wk * x; abs += Math.abs(wk * x); prop += Math.abs(wk) * normed.e[src * C + c];
    }
    const eConv = gamma(K) * (abs + prop) + prop;
    const silu = acc / (1 + Math.exp(-acc));
    const eSilu = 1.0998 * eConv + 8 * u * Math.abs(silu) + 2 ** -126;
    const i = t * C + c;
    const inner = gated.v[i] + silu;
    output.v[i] = hidden[i] + inner;
    output.e[i] = gated.e[i] + eSilu + u * Math.abs(inner) + u * Math.abs(output.v[i]);
  }
}
const ratio = (actual, ref, index) => Math.abs(actual - ref.v[index]) / ref.e[index];

// Discrimination: reversed convolution taps and a gate without the signed square root exceed twice the bound.
{
  let reversed = 0;
  let unsquared = 0;
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < C; c += 7) {
      let acc = 0;
      for (let k = 0; k < K; k++) {
        const src = t - stateLength + k * ngram.convDilation;
        if (src >= 0) acc += w.conv[c * K + (K - 1 - k)] * normed.v[src * C + c];
      }
      const i = t * C + c;
      const alt = hidden[i] + (gated.v[i] + acc / (1 + Math.exp(-acc)));
      reversed = Math.max(reversed, Math.abs(alt - output.v[i]) / output.e[i]);
    }
    for (let s = 0; s < S; s++) {
      const g = Math.sign(gate.v[t * S + s]) * gate.v[t * S + s] ** 2;
      unsquared = Math.max(unsquared, Math.abs(g - gate.v[t * S + s]) / gate.e[t * S + s]);
    }
  }
  assert.ok(reversed > 2 && unsquared > 2, `alternatives distinguished: ${reversed}, ${unsquared}`);
}

// Upstream torch values lie within the bound.
const channels = Array.from({ length: fixture.outputChannels.count }, (_, i) => i * fixture.outputChannels.stride);
let upstreamOutputRatio = 0;
// Upstream PLELayer.forward returns gated + conv; the decoder layer adds the streams (an f32 add).
fixture.output.forEach((row, t) => row.forEach((v, j) => {
  const index = t * C + channels[j];
  upstreamOutputRatio = Math.max(upstreamOutputRatio, ratio(Math.fround(hidden[index] + v), output, index));
}));
let upstreamGateRatio = 0;
fixture.gateInput.forEach((row, t) => row.forEach((v, s) => { upstreamGateRatio = Math.max(upstreamGateRatio, ratio(v, gate, t * S + s)); }));
assert.ok(upstreamOutputRatio <= 1 && upstreamGateRatio <= 1, `upstream within bound: ${upstreamOutputRatio}, ${upstreamGateRatio}`);

const owned = new Set();
const upload = (data, shape, dtype = 'f32') => {
  const buffer = acquireBuffer(data.byteLength);
  owned.add(buffer);
  uploadData(buffer, data);
  return shape ? createTensor(buffer, dtype, shape) : buffer;
};
const results = { upstreamOutputRatio, upstreamGateRatio, runs: [] };
const states = [];
try {
  const weights = {
    keyProj: upload(w.key), valueProj: upload(w.value), normKey: upload(w.normKey, [C]), normQuery: upload(w.normQuery, [C]),
    normConv: upload(w.normConv, [C]), conv: upload(w.conv, [C * K]),
  };
  for (const [mode, chunks] of [['immediate', [T]], ['immediate', fixture.chunks], ['recorded', fixture.chunks]]) {
    const state = new PleConvState({ channels: C, stateLength });
    states.push(state);
    const recorder = mode === 'recorded' ? new CommandRecorder(getDevice(), 'ple_check') : null;
    const matmul = recorder ? recordMatmul.bind(null, recorder) : runMatmul;
    const outputs = [];
    let start = 0;
    for (const size of chunks) {
      const streams = upload(hidden.subarray(start * C, (start + size) * C), [size, C]);
      const rows = upload(rowBits.subarray(start * E, (start + size) * E));
      const result = await runPleLayer(recorder, { streams, rows, weights, state, ngram, hyper, numTokens: size,
        project: (role, input, weightBuffer, outF, inF) => matmul(input, weightBuffer, size, outF, inF,
          { transposeB: true, outputDtype: 'f32', bDtype: 'f32' }) });
      owned.add(result.streams.buffer);
      owned.add(result.gateInput.buffer);
      outputs.push({ start, size, result });
      start += size;
    }
    if (recorder) await recorder.submitAndWait();
    let outputRatio = 0;
    let gateRatio = 0;
    for (const { start: s0, size, result } of outputs) {
      const out = new Float32Array(await readBuffer(result.streams.buffer, size * C * 4));
      const g = new Float32Array(await readBuffer(result.gateInput.buffer, size * S * 4));
      for (let i = 0; i < size * C; i++) outputRatio = Math.max(outputRatio, ratio(out[i], output, s0 * C + i));
      for (let i = 0; i < size * S; i++) gateRatio = Math.max(gateRatio, ratio(g[i], gate, s0 * S + i));
    }
    const finalState = new Float32Array(await readBuffer(state.buffer, stateLength * C * 4));
    let stateRatio = 0;
    for (let j = 0; j < stateLength; j++) {
      for (let c = 0; c < C; c++) stateRatio = Math.max(stateRatio, ratio(finalState[j * C + c], normed, (T - stateLength + j) * C + c));
    }
    assert.ok(outputRatio <= 1 && gateRatio <= 1 && stateRatio <= 1, `${mode} ${chunks}: ${outputRatio} ${gateRatio} ${stateRatio}`);
    results.runs.push({ mode, chunks, outputRatio, gateRatio, stateRatio });
  }
  // A reset state reproduces a fresh full-sequence run; a destroyed state is rejected.
  const reused = states[1];
  reused.reset();
  const again = await runPleLayer(null, { streams: upload(hidden, [T, C]), rows: upload(rowBits), weights, state: reused, ngram, hyper,
    numTokens: T, project: (role, input, weightBuffer, outF, inF) => runMatmul(input, weightBuffer, T, outF, inF,
      { transposeB: true, outputDtype: 'f32', bDtype: 'f32' }) });
  owned.add(again.streams.buffer);
  owned.add(again.gateInput.buffer);
  const first = new Float32Array(await readBuffer(again.streams.buffer, 64 * 4));
  assert.ok(first.every((v, i) => ratio(v, output, i) <= 1), 'reset state reproduces a fresh sequence');
  await reused.destroy();
  await assert.rejects(runPleLayer(null, { streams: upload(hidden, [T, C]), rows: upload(rowBits), weights, state: reused, ngram, hyper,
    numTokens: T, project: async () => { throw new Error('unreachable'); } }), /live PleConvState/);
} finally {
  for (const state of states) await state.destroy();
  await getDevice()?.queue.onSubmittedWorkDone();
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice();
}
console.log(JSON.stringify({ test: 'ple-layer-gpu', adapter: capabilities?.adapterInfo ?? null, results,
  scope: 'Physical PLE layer on synthetic weights and BF16-exact spy rows; no learned weights or timing' }));
