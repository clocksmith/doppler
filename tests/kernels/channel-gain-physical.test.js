import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { acquireBuffer, releaseBuffer, readBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { runDenseFFNGPU } from '../../src/inference/pipelines/text/ffn/dense.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { channelEnergy, deriveChannelGains, channelGainSwiGLU, controlChannelGains } from '../../src/gpu/kernels/channel-gain.js';
import { createChannelGainRequest, freezeChannelGains } from '../../src/inference/pipelines/text/channel-gain.js';

const began = performance.now();
const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, probe.reason);
const device = getDevice();
const adapter = getKernelCapabilities().adapterInfo;
assert.doesNotMatch(JSON.stringify(adapter), /swiftshader|llvmpipe|software/i);
const owned = new Set();
const remember = tensor => { owned.add(tensor.buffer); return tensor; };
function tensor(values, shape) {
  const data = new Float32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, 'channel-gain-fixture');
  owned.add(buffer); uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape);
}
const values = async t => [...new Float32Array(await readBuffer(t.buffer, t.shape.reduce((a, b) => a * b, 1) * 4))];
function near(actual, expected, tolerance = 2e-6) {
  assert.equal(actual.length, expected.length);
  actual.forEach((x, i) => assert.ok(Number.isFinite(x) && Math.abs(x - expected[i]) <= tolerance * Math.max(1, Math.abs(expected[i])), `${i}: ${x} vs ${expected[i]}`));
}
const timings = [];
let recorder;
device.pushErrorScope('validation');
try {
  const activation = tensor([100, 100, 100, 1, 2, 3, 3, 4, 5, 100, 100, 100], [4, 3]);
  const reference = tensor([1, 4, 9], [3]);
  const up = tensor([2, -3, 4, -1, 2, 0.5], [2, 3]);
  const gate = tensor([0, 1, -1, 2, -2, 0.25], [2, 3]);
  for (const transport of ['immediate', 'recorded']) {
    const start = performance.now();
    recorder = transport === 'recorded' ? createCommandRecorder('channel-gain-physical') : null;
    const energy = remember(await channelEnergy(recorder, activation, { tokens: 4, channels: 3, start: 1, stop: 3 }));
    const positive = remember(await deriveChannelGains(recorder, energy, reference, { channels: 3, beta: 0.5, clip: 1, epsilon: 1 }));
    const negative = remember(await deriveChannelGains(recorder, energy, reference, { channels: 3, beta: -0.5, clip: 1, epsilon: 1 }));
    const transformed = remember(await channelGainSwiGLU(recorder, up, gate, positive, { tokens: 2, channels: 3, swigluLimit: null }));
    if (recorder) await recorder.submitAndWait();
    near(await values(energy), [5, 10, 17]);
    // Raw clipped scores [1,1,0.8]; centered log gains [1/30,1/30,-1/15].
    const expected = [Math.exp(1 / 30), Math.exp(1 / 30), Math.exp(-1 / 15)];
    near(await values(positive), expected);
    near(await values(negative), expected.map(x => 1 / x));
    const ups = [2, -3, 4, -1, 2, 0.5], gates = [0, 1, -1, 2, -2, 0.25];
    near(await values(transformed), ups.map((x, i) => x * gates[i] / (1 + Math.exp(-gates[i])) * expected[i % 3]));
    timings.push({ transport, wallMsIncludingReadback: performance.now() - start });
    recorder = null;
  }
  const original = tensor([1, 2, 0.5], [3]);
  const frozen = await freezeChannelGains([{ layerIdx: 2, channels: 3, gains: original }]);
  const expectedHash = createHash('sha256').update('channel-gain-f32-v1:2:3:')
    .update(new Uint8Array(new Float32Array([1, 2, 0.5]).buffer)).digest('hex');
  assert.equal(frozen.gainHash, expectedHash);
  assert.equal(frozen.logicalBytes, 12);
  assert.equal(frozen.gpuBytes, original.buffer.size);
  assert.throws(() => { frozen.gpuBytes = 0; }, TypeError);
  assert.throws(() => { frozen.logicalBytes = 0; }, TypeError);
  const extraGains = tensor([1, 2], [2]);
  const multiLayer = await freezeChannelGains([
    { layerIdx: 2, channels: 3, gains: original },
    { layerIdx: 3, channels: 2, gains: extraGains },
  ]);
  assert.equal(multiLayer.logicalBytes, 20);
  assert.equal(multiLayer.gpuBytes, original.buffer.size + extraGains.buffer.size);
  assert.ok(multiLayer.gpuBytes >= multiLayer.logicalBytes);
  assert.deepEqual(Object.keys(multiLayer.layers[0]).sort(), ['channels', 'layerIdx']);
  multiLayer.close();
  assert.equal(multiLayer.logicalBytes, 20, 'closed handle metadata is an immutable allocation receipt');
  uploadData(original.buffer, new Float32Array([9, 9, 9]));
  const applySpec = { mode: 'apply', requestId: 'toy-apply', gainHash: frozen.gainHash,
    frozenGains: frozen, tokenCount: 2, span: { start: 0, stop: 2, templatePositions: [] },
    layers: [{ layerIdx: 2, channels: 3 }] };
  const context = { config: { useMoE: false, gatedActivation: true, hiddenActivation: 'silu' },
    activationDtype: 'f32', phase: 'prefill', currentSeqLen: 0 };
  const reprefill = createChannelGainRequest({ ...applySpec, cacheArm: 're-prefill' });
  const baseCache = createChannelGainRequest({ ...applySpec, cacheArm: 'base-cache' });
  assert.equal(reprefill.gainHash, baseCache.gainHash);
  assert.throws(() => frozen.close(), /active gain requests/);
  assert.throws(() => baseCache.validateLayer(2, context, 2, 3), /after the unmodified prefill/);
  reprefill.validateLayer(2, context, 2, 3);
  const reprefillOutput = remember(await reprefill.activate(2, up, gate, 2, 3, null, null));
  near(await values(reprefillOutput), [0, -6 / (1 + Math.exp(-1)), -2 / (1 + Math.exp(1)),
    -2 / (1 + Math.exp(-2)), -8 / (1 + Math.exp(2)), 0.0625 / (1 + Math.exp(-0.25))]);
  const decodeContext = { ...context, phase: 'decode', currentSeqLen: 2 };
  reprefill.validateLayer(2, decodeContext, 1, 3);
  baseCache.validateLayer(2, decodeContext, 1, 3);
  const decodeA = remember(await reprefill.activate(2, up, gate, 1, 3, null, null));
  const decodeB = remember(await baseCache.activate(2, up, gate, 1, 3, null, null));
  near(await values(decodeA), await values(decodeB));
  assert.throws(() => reprefill.validateLayer(2, context, 2, 3), /repeated capture/);
  reprefill.close(); baseCache.close(); frozen.close();
  assert.throws(() => createChannelGainRequest({ ...applySpec, cacheArm: 'base-cache' }), /live opaque/);
  const invalid = tensor([1, 0, 2], [3]);
  await assert.rejects(freezeChannelGains([{ layerIdx: 2, channels: 3, gains: invalid }]), /finite and positive/);
  const controlSource = tensor([Math.exp(-0.2), 1, Math.exp(0.2)], [3]);
  const indices = acquireBuffer(12, undefined, 'permutation-fixture'); owned.add(indices);
  uploadData(indices, new Uint32Array([2, 0, 1]));
  const z = tensor([0.5, -0.5, 0], [3]);
  const permuted = remember(await controlChannelGains(null, controlSource,
    { channels: 3, mode: 'permuted', seeds: indices }));
  near(await values(permuted), [Math.exp(0.2), Math.exp(-0.2), 1]);
  const random = remember(await controlChannelGains(null, controlSource,
    { channels: 3, mode: 'random-z', seeds: z.buffer }));
  near(await values(random), [Math.exp(0.2), Math.exp(-0.2), 1]);
  near(await values(controlSource), [Math.exp(-0.2), 1, Math.exp(0.2)]);
  const negativeEnergy = tensor([-1, 2, 3], [3]);
  const invalidDerived = remember(await deriveChannelGains(null, negativeEnergy, reference,
    { channels: 3, beta: 0.5, clip: 1, epsilon: 1 }));
  await assert.rejects(freezeChannelGains([{ layerIdx: 2, channels: 3, gains: invalidDerived }]), /finite and positive/);
  const capture = createChannelGainRequest({ mode: 'capture', requestId: 'toy-capture', gainHash: '',
    cacheArm: 'base-cache', tokenCount: 4, span: { start: 1, stop: 3, templatePositions: [0, 3] },
    layers: [{ layerIdx: 2, channels: 3 }] });
  await capture.capture(2, activation, 4, 3, null);
  await device.queue.onSubmittedWorkDone();
  const energies = capture.takeEnergies();
  near(await values(remember(energies.get(2))), [5, 10, 17]);
  await assert.rejects(capture.capture(2, activation, 4, 3, null), /already captured/);
  capture.close();
  const denseInput = tensor([1, 2, 3, 4], [2, 2]);
  const identity = tensor([1, 0, 0, 1], [2, 2]);
  const weight = { buffer: identity.buffer, dtype: 'f32', shape: [2, 2], layout: 'row' };
  const denseSpec = { mode: 'capture', requestId: 'dense-capture', gainHash: '', cacheArm: 'base-cache',
    tokenCount: 2, span: { start: 0, stop: 1, templatePositions: [] }, layers: [{ layerIdx: 2, channels: 2 }] };
  const denseRequest = createChannelGainRequest(denseSpec);
  const denseContext = { config: { hiddenSize: 2, intermediateSize: 2, hiddenActivation: 'silu',
    gatedActivation: true, swigluLimit: null, useDoubleWideMlp: false, useMoE: false },
    phase: 'prefill', currentSeqLen: 0, activationDtype: 'f32', recorder: null,
    channelGainRequest: denseRequest, kernelPath: null };
  const weights = { gate: weight, up: weight, down: weight };
  const denseBase = remember(await runDenseFFNGPU(2, denseInput, 2, denseContext, weights));
  const baseExpected = [1, 2, 3, 4].map(x => x * x / (1 + Math.exp(-x)));
  near(await values(denseBase), baseExpected);
  const denseEnergies = denseRequest.takeEnergies();
  near(await values(remember(denseEnergies.get(2))), baseExpected.slice(0, 2).map(x => x * x));
  denseRequest.close();
  const denseFrozen = await freezeChannelGains([{ layerIdx: 2, channels: 2, gains: tensor([2, 0.5], [2]) }]);
  const denseApply = createChannelGainRequest({ ...denseSpec, mode: 'apply', cacheArm: 're-prefill',
    frozenGains: denseFrozen, gainHash: denseFrozen.gainHash });
  const denseTransformed = remember(await runDenseFFNGPU(2, denseInput, 2,
    { ...denseContext, channelGainRequest: denseApply }, weights));
  near(await values(denseTransformed), baseExpected.map((x, i) => x * [2, 0.5][i % 2]));
  near(await values(identity), [1, 0, 0, 1]);
  denseApply.close(); denseFrozen.close();
  assert.equal(await device.popErrorScope(), null);
  console.log(JSON.stringify({ test: 'channel-gain-physical', passed: true, adapter,
    providerReceipt: probe.providerReceipt, timings, totalWallMs: performance.now() - began,
    evidence: 'Toy kernel numerical validation only; no model, cache comparison, selection or confirmation. Coexists with desktop services; timing is not a latency claim.' }));
} finally {
  if (recorder) await recorder.abort();
  await device.queue.onSubmittedWorkDone();
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice();
  releaseNodeWebGPU();
}
