import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { acquireBuffer, releaseBuffer, readBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { getDeviceMemorySnapshot } from '../../src/memory/device-budget.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { runDenseFFNGPU } from '../../src/inference/pipelines/text/ffn/dense.js';
import { PipelineGenerator } from '../../src/inference/pipelines/text/generator.js';
import {
  createChannelGainRequest, freezeChannelGains, withChannelGainOperation, assertChannelGainIdle,
} from '../../src/inference/pipelines/text/channel-gain.js';

// Two independent request states share tiny immutable input/weight buffers. This
// exercises the actual dense GPU path and reset hook, not a full model or KV cache.
const began = performance.now();
const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, probe.reason);
const device = getDevice();
const adapter = getKernelCapabilities().adapterInfo;
assert.doesNotMatch(JSON.stringify(adapter), /swiftshader|llvmpipe|software/i);
const buffers = new Set();
const handles = new Set();
const states = new Set();
const checks = [];
let releaseHeldOperation;
let heldOperation;
function tensor(values, shape) {
  const data = new Float32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, 'channel-gain-isolation-fixture');
  buffers.add(buffer);
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape);
}
async function values(value) {
  return [...new Float32Array(await readBuffer(value.buffer, value.shape.reduce((a, b) => a * b, 1) * 4))];
}
function near(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((v, i) => assert.ok(Number.isFinite(v) && Math.abs(v - expected[i]) <= 2e-6 * Math.max(1, Math.abs(expected[i])), `${i}: ${v} vs ${expected[i]}`));
}
function requestState(id, frozen) {
  const state = { isGenerating: false, channelGainRequest: createChannelGainRequest({
    mode: 'apply', requestId: id, cacheArm: 'base-cache', frozenGains: frozen, gainHash: frozen.gainHash,
    tokenCount: 1, span: { start: 0, stop: 1, templatePositions: [] },
    layers: [{ layerIdx: 30, channels: 2 }],
  }) };
  states.add(state);
  return state;
}
function reset(state) {
  assertChannelGainIdle(state);
  PipelineGenerator.prototype.resetGenerationState.call({ _state: state,
    _resetReplayPrefillRuntimeState() {}, _resetDecodeRuntimeState() {} });
  assert.equal(state.channelGainRequest, null);
}
let validationScopeOpen = false;
try {
  device.pushErrorScope('validation');
  validationScopeOpen = true;
  const input = tensor([1, 2], [1, 2]);
  const identity = tensor([1, 0, 0, 1], [2, 2]);
  const weight = { buffer: identity.buffer, dtype: 'f32', shape: [2, 2], layout: 'row' };
  const weights = { gate: weight, up: weight, down: weight };
  const context = { config: { hiddenSize: 2, intermediateSize: 2, hiddenActivation: 'silu',
    gatedActivation: true, swigluLimit: null, useDoubleWideMlp: false, useMoE: false },
  phase: 'decode', currentSeqLen: 1, activationDtype: 'f32', recorder: null, kernelPath: null };
  async function dispatch(state) {
    const output = await runDenseFFNGPU(30, input, 1,
      { ...context, channelGainRequest: state?.channelGainRequest ?? null }, weights);
    buffers.add(output.buffer);
    return values(output);
  }
  const baseline = await dispatch(null);
  near(baseline, [1 / (1 + Math.exp(-1)), 4 / (1 + Math.exp(-2))]);
  const gainInputA = tensor([2, 0.5], [2]);
  const gainInputB = tensor([0.5, 2], [2]);
  const frozenA = await freezeChannelGains([{ layerIdx: 30, channels: 2, gains: gainInputA }]);
  handles.add(frozenA);
  const frozenB = await freezeChannelGains([{ layerIdx: 30, channels: 2, gains: gainInputB }]);
  handles.add(frozenB);
  assert.notEqual(frozenA.gainHash, frozenB.gainHash);
  const stateA = requestState('isolation-a', frozenA);
  const stateB = requestState('isolation-b', frozenB);
  const expectedA = baseline.map((v, i) => v * [2, 0.5][i]);
  const expectedB = baseline.map((v, i) => v * [0.5, 2][i]);
  // Mutating caller-owned seed buffers cannot mutate the private frozen copies.
  uploadData(gainInputA.buffer, new Float32Array([9, 9]));
  uploadData(gainInputB.buffer, new Float32Array([7, 7]));
  for (const [state, expected] of [[stateA, expectedA], [stateB, expectedB], [stateA, expectedA], [stateB, expectedB]]) {
    near(await withChannelGainOperation(state, () => dispatch(state)), expected);
  }
  checks.push('two_live_distinct_gain_states_alternate_without_cross_talk', 'frozen_private_copies_resist_seed_buffer_mutation');
  // Hold A's asynchronous operation open while B completes a real GPU dispatch.
  const held = new Promise(resolve => { releaseHeldOperation = resolve; });
  heldOperation = withChannelGainOperation(stateA, async () => { await held; return dispatch(stateA); });
  assert.throws(() => assertChannelGainIdle(stateA), /active operation/);
  await assert.rejects(withChannelGainOperation(stateA, async () => {}), /serialized/);
  near(await withChannelGainOperation(stateB, () => dispatch(stateB)), expectedB);
  releaseHeldOperation();
  near(await heldOperation, expectedA);
  heldOperation = null;
  checks.push('independent_operation_lifetimes_overlap_while_same_state_reentry_is_rejected');
  const requestA = stateA.channelGainRequest;
  reset(stateA);
  assert.throws(() => requestA.has(30), /closed/);
  frozenA.close();
  await device.queue.onSubmittedWorkDone();
  near(await withChannelGainOperation(stateB, () => dispatch(stateB)), expectedB);
  near(await dispatch(stateA), baseline);
  checks.push('reset_and_discard_a_preserve_b_and_restore_unmodified_base');
  const failing = requestState('isolation-failure', frozenB);
  const failedRequest = failing.channelGainRequest;
  await assert.rejects(withChannelGainOperation(failing, async () => {
    near(await dispatch(failing), expectedB);
    throw new Error('injected application failure after completed GPU dispatch');
  }), /injected application failure/);
  assert.doesNotThrow(() => assertChannelGainIdle(failing));
  reset(failing);
  assert.throws(() => failedRequest.has(30), /closed/);
  assert.throws(() => frozenB.close(), /active gain requests/);
  near(await withChannelGainOperation(stateB, () => dispatch(stateB)), expectedB);
  reset(stateB);
  frozenB.close();
  await device.queue.onSubmittedWorkDone();
  near(await dispatch(null), baseline);
  checks.push('failure_cleanup_releases_only_its_borrow_and_other_state_remains_valid');
  assert.deepEqual(await values(input), [1, 2]);
  assert.deepEqual(await values(identity), [1, 0, 0, 1]);
  checks.push('shared_input_and_weight_buffers_unchanged');
  assert.equal(await device.popErrorScope(), null);
  validationScopeOpen = false;
  const { labels, ...memory } = getDeviceMemorySnapshot(device);
  assert.ok(memory.peakBytes < 1024 * 1024, 'synthetic test must stay below 1MiB of GPU-buffer storage');
  console.log(JSON.stringify({ test: 'channel-gain-isolation-physical', passed: true, adapter,
    providerReceipt: probe.providerReceipt, checks, memory: { ...memory, labelCount: labels.length },
    totalWallMs: performance.now() - began, modelLoads: 0, modelConcurrencyTested: false,
    kernelExecution: 'serialized GPU queue with distinct live request states and overlapping asynchronous operation lifetimes',
    resetScope: 'actual PipelineGenerator reset hook with minimal state fixture; no KV/cache or full-model sequence',
    evidence: 'Synthetic dense FFN/request isolation and cleanup only; no cross-domain quality, serving concurrency, or latency claim.' }));
} finally {
  try {
    releaseHeldOperation?.();
    await heldOperation?.catch(() => {});
    await device.queue.onSubmittedWorkDone();
    for (const state of states) reset(state);
    for (const handle of handles) handle.close();
    await device.queue.onSubmittedWorkDone();
    if (validationScopeOpen) await device.popErrorScope();
    for (const buffer of buffers) releaseBuffer(buffer);
  } finally {
    try { destroyDevice(); }
    finally { await releaseNodeWebGPU(); }
  }
}
