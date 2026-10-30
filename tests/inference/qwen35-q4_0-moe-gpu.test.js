import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer, getBufferPool, isBufferActive } from '../../src/memory/buffer-pool.js';
import { createKernelRegistry, enterKernelRegistry } from '../../src/gpu/kernels/kernel-configs.js';
import { q40Fixture,packedFixture } from '../helpers/q4_0-fixture.js';
import { MoERouter } from '../../src/inference/moe-router.js';
import { runMoEFFNGPU } from '../../src/inference/pipelines/text/ffn/moe.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';
import { float32ToFloat16, float16ToFloat32 } from '../../src/formats/float16.js';
import { PreparedDecodeWorkspace } from '../../src/gpu/prepared-decode-workspace.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical MoE ordering verification unavailable: ${probe.reason}`);
const allocated = [], errors = [], device = getDevice(), previous = getRuntimeConfig();
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const buffer = values => { const b = acquireBuffer(values.byteLength); allocated.push(b); uploadData(b, values); return b; };
const matrix = (values, shape) => createWeightBuffer(buffer(values), 'f32', 'row', shape);
const bank = (values, shape, dtype='q4_0') => createWeightBuffer(buffer(packedFixture(dtype,shape.slice(0,-1).reduce((a,b)=>a*b,1),shape.at(-1))), dtype, 'row', shape);
const T = 2, H = 256, I = 256, E = 3, S = 32;
const values = (n, f) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * f) * 0.1);
let control, recordedOutput, recorder, rawRecorder, workspace;
const rawOutputs = [];
try {
  setRuntimeConfig({ inference: { moe: { routing: { activeExpertSelection: 'topk-route' } } } });
  const source = buffer(values(T * H, 0.13)), pending = buffer(new Float32Array(T * H));
  const router = new MoERouter({ numExperts: E, topK: 2, hiddenSize: H, normalizeWeights: true });
  router.loadWeights(matrix(values(E * H, 0.07), [E, H]));
  const expertWeights = new Map([['layer_0_expert_0', { numExperts: E, expertFormat: 'qwen3_5',
    gateUp: bank(values(E * 2 * I * H, 0.03), [E, 2 * I, H]), down: bank(values(E * H * I, 0.09), [E, H, I], 'q4_1') }]]);
  const packed = expertWeights.get('layer_0_expert_0');
  for (let expertIdx = 0; expertIdx < E; expertIdx++) {
    expertWeights.set(`layer_0_expert_${expertIdx}`, { ...packed, expertIdx });
  }
  const shared = { sharedExpertGate: bank(values(S * H, 0.11), [S, H], 'q8_0'),
    sharedExpertUp: bank(values(S * H, 0.05), [S, H], 'q8_0'), sharedExpertDown: bank(values(H * S, 0.17), [H, S], 'q8_0'),
    sharedExpertRouter: matrix(values(H, 0.19), [1, H]) };
  const context = { config: { modelType: 'transformer', hiddenSize: H, intermediateSize: I,
    moeExpertIntermediateSize: I, moeSharedExpertIntermediateSize: S, numExperts: E, moeTopK: 2,
    expertFormat: 'qwen3_5', hiddenActivation: 'silu', swigluLimit: null, rmsNormEps: 1e-6 },
    moeRouter: router, expertWeights, expertLoader: { loadExpert() { throw Error('Resident banks must not reload'); } },
    weights: new Map([['layer_0', shared]]), kernelPath: null, executionPolicies: null };
  context.runtimeMoERoutingConfig = { ...getRuntimeConfig().inference.moe.routing, activeExpertSelection: 'topk-route' };
  control = await runMoEFFNGPU(0, createTensor(source, 'f32', [T, H]), T, context);
  const expected = new Float32Array(await readBuffer(control.buffer, T * H * 4));
  assert.ok(expected.some(value => Math.abs(value) > 0.0001), 'Fixture must expose a stale zero input');
  const combined=q40Fixture(E*2*I,H),rowBytes=H/32*18;
  const splitGate=new Uint8Array(E*I*rowBytes),splitUp=new Uint8Array(E*I*rowBytes);
  for(let e=0;e<E;e++){splitGate.set(combined.subarray(e*2*I*rowBytes,(e*2*I+I)*rowBytes),e*I*rowBytes);splitUp.set(combined.subarray((e*2*I+I)*rowBytes,(e+1)*2*I*rowBytes),e*I*rowBytes);}
  const separate={...packed,gateUp:undefined,projectionLayout:'separate',
    gate:createWeightBuffer(buffer(splitGate),'q4_0','row',[E,I,H]),up:createWeightBuffer(buffer(splitUp),'q4_0','row',[E,I,H])};
  for(let e=0;e<E;e++)expertWeights.set(`layer_0_expert_${e}`,{...separate,expertIdx:e});
  recorder = createCommandRecorder('moe-ordering-verification');
  recorder.getEncoder().copyBufferToBuffer(source, 0, pending, 0, T * H * 4);
  recordedOutput = await runMoEFFNGPU(0, createTensor(pending, 'f32', [T, H]), T, { ...context, recorder });
  await recorder.submitAndWait();
  const observed = new Float32Array(await readBuffer(recordedOutput.buffer, T * H * 4));
  let maxAbs = 0;
  for (let i = 0; i < expected.length; i++) {
    maxAbs = Math.max(maxAbs, Math.abs(observed[i] - expected[i]));
    assert.ok(Number.isFinite(observed[i]) && Math.abs(observed[i] - expected[i]) <= 0.00001,
      `Recorded MoE output[${i}] read stale upstream data: ${observed[i]} versus ${expected[i]}`);
  }
  const baselineActive = getBufferPool().getStats().activeBuffers;
  const failedRecorder = createCommandRecorder('moe-failure-verification');
  const leave = enterKernelRegistry(createKernelRegistry({ validators: { moe_route_matmul: { q4_0_split_f32: {
    id: 'moe-recorder-output-rollback', validate() { throw Error('Injected recorded expert failure'); },
  } } } }));
  try {
    await assert.rejects(runMoEFFNGPU(0, createTensor(source, 'f32', [T, H]), T,
      { ...context, recorder: failedRecorder }), /Injected recorded expert failure/);
    assert.ok(getBufferPool().getStats().activeBuffers > baselineActive,
      'Pending recorded inputs must stay owned until abort');
  } finally { leave(); await failedRecorder.abort(); }
  assert.equal(getBufferPool().getStats().activeBuffers, baselineActive, 'Aborted MoE releases every owned temporary');
  assert.ok(isBufferActive(source), 'Aborting MoE must preserve caller input');

  // An adapter with a CPU route readback explicitly submits the producer prefix
  // before immediate work and leaves the parent recorder open for shared/residual work.
  setRuntimeConfig({ inference: { moe: { routing: { activeExpertSelection: 'topk-readback' } } } });
  const fallbackRecorder = createCommandRecorder('moe-immediate-boundary-verification');
  let fallback;
  try {
    device.queue.writeBuffer(pending, 0, new Float32Array(T * H));
    fallbackRecorder.getEncoder().copyBufferToBuffer(source, 0, pending, 0, T * H * 4);
    fallback = await runMoEFFNGPU(0, createTensor(pending, 'f32', [T, H]), T, { ...context, recorder: fallbackRecorder,
      runtimeMoERoutingConfig: { ...context.runtimeMoERoutingConfig, activeExpertSelection: 'topk-readback' } });
    await fallbackRecorder.submitAndWait();
    const observed = new Float32Array(await readBuffer(fallback.buffer, T * H * 4));
    for (let i = 0; i < expected.length; i++) assert.ok(Number.isFinite(observed[i]) && Math.abs(observed[i] - expected[i]) <= 0.00001);
  } finally { await fallbackRecorder.abort(); if (fallback) releaseBuffer(fallback.buffer); }
  setRuntimeConfig({ inference: { moe: { routing: { activeExpertSelection: 'topk-route' } } } });
  workspace = new PreparedDecodeWorkspace(device);
  // Another run's global default cannot change this context's captured GPU route.
  setRuntimeConfig({ inference: { moe: { routing: { activeExpertSelection: 'all' } } } });
  const workspaceCases = [];
  const createBuffer = device.createBuffer.bind(device), createBindGroup = device.createBindGroup.bind(device);
  let createdBuffers = 0, createdBindings = 0;
  device.createBuffer = descriptor => { createdBuffers++; return createBuffer(descriptor); };
  device.createBindGroup = descriptor => { createdBindings++; return createBindGroup(descriptor); };
  try {
  for (let step = 0; step < 3; step++) {
    const stepRecorder = createCommandRecorder('moe-workspace-verification');
    stepRecorder.decodeWorkspace = workspace;
    const before = getBufferPool().getStats();
    const beforeBuffers = createdBuffers, beforeBindings = createdBindings;
    let output;
    try {
      output = await runMoEFFNGPU(0, createTensor(source, 'f32', [1, H]), 1, { ...context, recorder: stepRecorder });
      await stepRecorder.submitAndWait();
      const after = getBufferPool().getStats();
      const acquisitions = after.allocations + after.reuses - before.allocations - before.reuses;
      const gpuBufferCreations = createdBuffers - beforeBuffers, bindGroupCreations = createdBindings - beforeBindings;
      const observed = new Float32Array(await readBuffer(output.buffer, H * 4));
      for (let i = 0; i < H; i++) assert.ok(Number.isFinite(observed[i]) && Math.abs(observed[i] - expected[i]) <= 0.00001);
      if (step > 0) assert.equal(acquisitions, 0, 'Prepared MoE must reuse its scratch instead of acquiring pooled buffers per token');
      if (step > 0) assert.equal(gpuBufferCreations, 0, 'Prepared MoE must also reuse uniforms and raw GPU buffers after warmup');
      if (step > 0) assert.equal(bindGroupCreations, 0, 'Prepared MoE must reuse all operator bindings after warmup');
      workspaceCases.push({ step, acquisitions, gpuBufferCreations, bindGroupCreations, ...workspace.getStats() });
    } finally {
      await stepRecorder.abort();
      if (output && !workspace.ownsBuffer(output.buffer)) releaseBuffer(output.buffer);
    }
  }
  } finally { device.createBuffer = createBuffer; device.createBindGroup = createBindGroup; }
  const retained = workspace.getStats();
  const rejected = createCommandRecorder('prepared-moe-abort-verification');
  rejected.decodeWorkspace = workspace;
  const leavePreparedFailure = enterKernelRegistry(createKernelRegistry({ validators: { moe_route_matmul: { q4_0_split_f32: {
    id: 'prepared-moe-output-rollback', validate() { throw Error('Injected prepared expert failure'); },
  } } } }));
  try {
    await assert.rejects(runMoEFFNGPU(0, createTensor(source, 'f32', [1, H]), 1,
      { ...context, recorder: rejected }), /Injected prepared expert failure/);
  } finally { leavePreparedFailure(); await rejected.abort(); }
  assert.equal(workspace.getStats().buffers, retained.buffers, 'Aborting one recorder retains session-owned scratch');
  assert.equal(workspace.getStats().bytes, retained.bytes);
  const secondWorkspace = new PreparedDecodeWorkspace(device);
  const concurrent = [createCommandRecorder('prepared-session-a'), createCommandRecorder('prepared-session-b')];
  concurrent[0].decodeWorkspace = workspace; concurrent[1].decodeWorkspace = secondWorkspace;
  const zeroInput = buffer(new Float32Array(H));
  const concurrentOutputs = [];
  try {
    concurrentOutputs.push(...await Promise.all(concurrent.map((recorder, index) => runMoEFFNGPU(0,
      createTensor(index ? zeroInput : source, 'f32', [1, H]), 1, { ...context, recorder }))));
    assert.notEqual(concurrentOutputs[0].buffer, concurrentOutputs[1].buffer, 'Sessions cannot share MoE output slots');
    await concurrent[1].submitAndWait(); await concurrent[0].submitAndWait();
    for (const [index, output] of concurrentOutputs.entries()) {
      const values = new Float32Array(await readBuffer(output.buffer, H * 4));
      for (let i = 0; i < H; i++) assert.ok(Number.isFinite(values[i]) && Math.abs(values[i] - (index ? 0 : expected[i])) <= 0.00001);
    }
  } finally {
    for (const recorder of concurrent) await recorder.abort();
    secondWorkspace.release();
  }
  // One shared router is rebound across layers before submission. Uploaded
  // raw weights and biases must stay owned by that recording, not be destroyed
  // by the next loadWeights call.
  rawRecorder = createCommandRecorder('router-rebinding-verification');
  const rawRouter = new MoERouter({ numExperts: E, topK: 2, hiddenSize: H, normalizeWeights: true });
  const rawCases = [];
  const inputValues = values(T * H, 0.13);
  for (const [index, dtype] of ['f32', 'f16', 'f32'].entries()) {
    const original = values(E * H, 0.07 + index * 0.04);
    const weights = dtype === 'f16' ? Uint16Array.from(original, float32ToFloat16) : original;
    const decoded = dtype === 'f16' ? Float32Array.from(weights, float16ToFloat32) : weights;
    const bias = new Float32Array([0.1 + index, -0.2, 0.3]);
    rawRouter.loadWeights(weights, bias);
    const output = await rawRouter.computeRouterLogitsGPU(source, T, null,
      { inputDtype: 'f32', outputDtype: 'f32', recorder: rawRecorder });
    rawOutputs.push(output);
    rawCases.push({ output, decoded, bias, dtype });
  }
  await rawRecorder.submitAndWait();
  let rawMaxAbs = 0;
  for (const { output, decoded, bias } of rawCases) {
    const result = new Float32Array(await readBuffer(output, T * E * 4));
    for (let t = 0; t < T; t++) for (let e = 0; e < E; e++) {
      let expected = bias[e];
      for (let k = 0; k < H; k++) expected += inputValues[t * H + k] * decoded[e * H + k];
      rawMaxAbs = Math.max(rawMaxAbs, Math.abs(result[t * E + e] - expected));
      assert.ok(Number.isFinite(result[t * E + e]) && Math.abs(result[t * E + e] - expected) <= 0.00001);
    }
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ status: 'passed', maxAbs, rawMaxAbs, rawCases: rawCases.length, workspaceCases,
    provider: probe.providerReceipt, recorder: recorder.getStats() }));
} finally {
  await recorder?.abort();
  await rawRecorder?.abort();
  for (const output of rawOutputs) releaseBuffer(output);
  if (control) releaseBuffer(control.buffer);
  if (recordedOutput) releaseBuffer(recordedOutput.buffer);
  for (const b of allocated) releaseBuffer(b);
  workspace?.release();
  setRuntimeConfig(previous);
  destroyDevice();
}
