import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { runMatmul, recordMatmul } from '../../src/gpu/kernels/matmul.js';
import {
  recordHyperConnectionActivation, recordHyperConnectionBranch, recordHyperConnectionInject, recordHyperConnectionNorm,
  runHyperConnectionActivation, runHyperConnectionBranch, runHyperConnectionInject, runHyperConnectionNorm,
} from '../../src/gpu/kernels/hyper-connection.js';
import {
  activationReference, boundRatio, branchReference, fixtureInput, fixtureParameter, gatedResidualReference,
  groupedNormReference, injectReference,
} from './hyper-connection-reference.js';

// Contract: each qwen4_exp hyper-connection operator, and their composition with
// F32 matmul projections, stays within the derived F32 bound of the float64
// reference on a physical WebGPU device, immediate and recorded alike. The
// composition is also compared with the pinned upstream outputs.
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical hyper-connection check unavailable: ${probe.reason}`);
const capabilities = getKernelCapabilities();
const adapter = JSON.stringify(capabilities?.adapterInfo ?? {}).toLowerCase();
assert.ok(!/swiftshader|llvmpipe|lavapipe/.test(adapter), 'hyper-connection check requires a physical adapter');
console.log(JSON.stringify({ timeUTC: new Date().toISOString(), adapter: capabilities?.adapterInfo ?? null,
  scope: 'Physical operator and composition correctness on synthetic weights; no learned weights or timing' }));

const fixtures = JSON.parse(await readFile(
  new URL('../../artifacts/qwen38-flash-next-20261013/reference-fixtures.json', import.meta.url), 'utf8'));
const geometry = { tokens: 2, streams: 4, width: 2560, rank: 320, eps: 1e-6 };
const hc = geometry.streams * geometry.width;
const weights = {
  norm: fixtureParameter(hc, 0),
  down: fixtureParameter(geometry.rank * hc, 1),
  up: fixtureParameter(hc * geometry.rank, 2),
  inject: fixtureParameter(geometry.streams * hc, 3),
};
const input = fixtureInput(geometry.tokens * hc);
const reference = gatedResidualReference(input, weights, geometry);
const owned = new Set();
const results = [];

const upload = (values, shape) => {
  const data = values instanceof Float32Array ? values : Float32Array.from(values);
  const buffer = acquireBuffer(data.byteLength);
  owned.add(buffer);
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape);
};
const read = async (tensor, count) => {
  owned.add(tensor.buffer);
  return new Float32Array(await readBuffer(tensor.buffer, count * 4));
};
const zeros = (count) => new Float64Array(count);
const check = (label, mode, actual, expected) => {
  const { ratio, index } = boundRatio(actual, expected.value, expected.bound);
  assert.ok(Number.isFinite(ratio) && ratio <= 1, `${mode}/${label}: error/bound=${ratio} at ${index}`);
  results.push({ mode, label, ratio });
};

try {
  for (const mode of ['immediate', 'recorded']) {
    const recorder = mode === 'recorded' ? new CommandRecorder(getDevice(), 'hyper_connection_check') : null;
    const norm = recorder ? recordHyperConnectionNorm.bind(null, recorder) : runHyperConnectionNorm;
    const activate = recorder ? recordHyperConnectionActivation.bind(null, recorder) : runHyperConnectionActivation;
    const branch = recorder ? recordHyperConnectionBranch.bind(null, recorder) : runHyperConnectionBranch;
    const inject = recorder ? recordHyperConnectionInject.bind(null, recorder) : runHyperConnectionInject;
    const matmul = recorder ? recordMatmul.bind(null, recorder) : runMatmul;
    const streamOptions = { numTokens: geometry.tokens, streamCount: geometry.streams, streamWidth: geometry.width, normWeightOffset: true };
    try {
      // Independent operators on exactly representable inputs.
      const x = upload(input, [geometry.tokens, hc]);
      const normWeight = upload(weights.norm, [hc]);
      const normed = await norm(x, normWeight, { ...streamOptions, eps: geometry.eps });
      const folded = upload(Float32Array.from(weights.norm, value => 1 + value), [hc]);
      const rawNormed = await norm(x, folded, { ...streamOptions, eps: geometry.eps, normWeightOffset: false });

      const edges = [0, -0, 1e-30, -1e-30, 3, -3, 40, -40, 100, -100, 1e4, -1e4];
      const activationInput = Float32Array.from({ length: 4096 }, (_, i) => (i < edges.length ? edges[i] : Math.sin(i * 0.37) * 30));
      const activationTensor = upload(activationInput, [64, 64]);
      const activations = {};
      for (const kind of ['low_rank_silu', 'mix_sigmoid', 'inject_gate']) {
        activations[kind] = await activate(activationTensor, { mode: kind, rows: 64, columns: 64, streamCount: geometry.streams });
      }

      const mixInput = Float32Array.from(reference.mix.value);
      const normedInput = Float32Array.from(reference.normed.value);
      const branchOut = await branch(upload(mixInput, [geometry.tokens, hc]), upload(normedInput, [geometry.tokens, hc]), streamOptions);

      const branchInput = Float32Array.from(reference.branch.value);
      const gateInput = Float32Array.from(reference.injection.value);
      const injected = await inject(x, upload(branchInput, [geometry.tokens, geometry.width]),
        upload(gateInput, [geometry.tokens, geometry.streams]), streamOptions);

      // Composition: norm -> down -> silu(x/S) -> up -> sigmoid -> mean; norm -> inject -> 2 sigmoid(x/S).
      const down = upload(weights.down, [geometry.rank, hc]);
      const up = upload(weights.up, [hc, geometry.rank]);
      const injectWeight = upload(weights.inject, [geometry.streams, hc]);
      const projection = { transposeB: true, outputDtype: 'f32', bDtype: 'f32' };
      const lowRankLogits = await matmul(normed, down.buffer, geometry.tokens, geometry.rank, hc, projection);
      const lowRank = await activate(lowRankLogits, { mode: 'low_rank_silu', rows: geometry.tokens, columns: geometry.rank, streamCount: geometry.streams });
      const mixLogits = await matmul(lowRank, up.buffer, geometry.tokens, hc, geometry.rank, projection);
      const mix = await activate(mixLogits, { mode: 'mix_sigmoid', rows: geometry.tokens, columns: hc, streamCount: geometry.streams });
      const composed = await branch(mix, normed, streamOptions);
      const injectLogits = await matmul(normed, injectWeight.buffer, geometry.tokens, geometry.streams, hc, projection);
      const gate = await activate(injectLogits, { mode: 'inject_gate', rows: geometry.tokens, columns: geometry.streams, streamCount: geometry.streams });
      if (recorder) await recorder.submitAndWait();

      assert.deepEqual(await read(rawNormed, geometry.tokens * hc), await read(normed, geometry.tokens * hc), 'folded GGUF scales equal HF residual scales');
      check('norm', mode, await read(normed, geometry.tokens * hc),
        groupedNormReference(input, weights.norm, { tokens: geometry.tokens, streams: geometry.streams, width: geometry.width, eps: geometry.eps }));
      const activationValues = Float64Array.from(activationInput);
      for (const kind of ['low_rank_silu', 'mix_sigmoid', 'inject_gate']) {
        check(kind, mode, await read(activations[kind], 4096), activationReference(kind, activationValues, zeros(4096), geometry.streams));
      }
      check('branch', mode, await read(branchOut, geometry.tokens * geometry.width),
        branchReference(Float64Array.from(mixInput), zeros(mixInput.length), Float64Array.from(normedInput), zeros(normedInput.length),
          { tokens: geometry.tokens, streams: geometry.streams, width: geometry.width }));
      check('inject', mode, await read(injected, geometry.tokens * hc),
        injectReference(Float64Array.from(input), Float64Array.from(branchInput), zeros(branchInput.length),
          Float64Array.from(gateInput), zeros(gateInput.length), { tokens: geometry.tokens, streamCount: geometry.streams, width: geometry.width }));

      const composedValues = await read(composed, geometry.tokens * geometry.width);
      const gateValues = await read(gate, geometry.tokens * geometry.streams);
      for (const tensor of [lowRankLogits, lowRank, mixLogits, mix, injectLogits]) owned.add(tensor.buffer);
      check('composed branch input', mode, composedValues, reference.branch);
      check('composed injection weights', mode, gateValues, reference.injection);
      const upstream = Float64Array.from(fixtures.gatedResidual.mixed.values[0].flat());
      const upstreamGate = Float64Array.from(fixtures.gatedResidual.injection.values[0].flat());
      let upstreamMax = 0;
      for (let i = 0; i < upstream.length; i++) upstreamMax = Math.max(upstreamMax, Math.abs(composedValues[i] - upstream[i]));
      let upstreamGateMax = 0;
      for (let i = 0; i < upstreamGate.length; i++) upstreamGateMax = Math.max(upstreamGateMax, Math.abs(gateValues[i] - upstreamGate[i]));
      results.push({ mode, label: 'max |gpu - upstream torch|', branch: upstreamMax, injection: upstreamGateMax });
    } finally {
      await recorder?.abort();
    }
  }

  // Validation fails before dispatch.
  const x = upload(input, [geometry.tokens, hc]);
  const normWeight = upload(weights.norm, [hc]);
  const streamOptions = { numTokens: geometry.tokens, streamCount: geometry.streams, streamWidth: geometry.width, normWeightOffset: true };
  await assert.rejects(runHyperConnectionNorm(x, normWeight, { ...streamOptions, streamWidth: 2048, eps: 1e-6 }), /shape/);
  await assert.rejects(runHyperConnectionNorm(x, normWeight, { ...streamOptions, eps: 0 }), /eps/);
  await assert.rejects(runHyperConnectionNorm(x, normWeight, { ...streamOptions, streamCount: 1, eps: 1e-6 }), /streamCount/);
  await assert.rejects(runHyperConnectionNorm(x, normWeight, { ...streamOptions, eps: 1e-6, outputBuffer: x.buffer }), /alias/);
  await assert.rejects(runHyperConnectionActivation(x, { mode: 'gelu', rows: 2, columns: hc, streamCount: 4 }), /unknown activation/);
  await assert.rejects(runHyperConnectionInject(x, x, x, streamOptions), /inject branch output/);
} finally {
  await getDevice()?.queue.onSubmittedWorkDone();
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice();
}

assert.equal(results.filter((entry) => entry.ratio !== undefined).length, 2 * 8);
console.log(JSON.stringify({ test: 'hyper-connection-gpu', results }));
