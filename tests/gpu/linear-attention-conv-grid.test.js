import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

// The real wrapper must admit the 2048 x 8192 boundary on every axis, with
// identical immediate/recorded grids, and reject oversized grids before allocation.
const target = new URL('../../src/gpu/kernels/linear-attention-core.js', import.meta.url).href;
const originalBuffer = globalThis.GPUBuffer;
const originalStage = globalThis.GPUShaderStage;
globalThis.GPUBuffer = class { size = 32 * 128 * 128 * 4; destroy() {} };
globalThis.GPUShaderStage = { COMPUTE: 4 };
const active = new Set(), dispatches = [];
let allocations = 0;
const device = {
  limits: { maxComputeWorkgroupsPerDimension: 65535 },
  createComputePipeline: descriptor => descriptor,
  createBindGroup: () => ({}),
  createCommandEncoder() {
    return { finish() { return {}; }, beginComputePass() {
      let selected;
      return { setPipeline(value) { selected = value; }, setBindGroup() {}, end() {},
        dispatchWorkgroups(...grid) { dispatches.push({ label: selected.label, grid, width: selected.compute.constants.WORKGROUP_SIZE }); } };
    } };
  },
  queue: { submit() {}, onSubmittedWorkDone: async () => {} },
};
globalThis.convGridProbe = {
  device,
  acquire() { const buffer = new GPUBuffer(); active.add(buffer); allocations++; return buffer; },
  release(buffer) { assert(active.delete(buffer), 'release exactly once'); },
};
const modules = {
  '../device.js': 'export const getDevice = () => convGridProbe.device;',
  './shader-source-scope.js': 'export const getShaderScopeCacheKey = () => 0;',
  '../../memory/buffer-pool.js': `export const acquireBuffer = () => convGridProbe.acquire();
    export const releaseBuffer = buffer => convGridProbe.release(buffer);`,
  '../../config/runtime.js': 'export const getRuntimeConfig = () => ({});',
  './uniform-utils.js': 'export const createUniformBufferFromData = () => new GPUBuffer();',
  './shader-cache.js': 'export const getShaderModule = async () => ({});',
  './pipeline-cache.js': 'export const getOrCreateBindGroupLayout = () => ({}); export const getOrCreatePipelineLayout = () => ({});',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) {
    return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  }
  return next(specifier, context);
} });
try {
  const { runLinearAttentionCoreGPU } = await import(target);
  const tensor = { buffer: new GPUBuffer(), dtype: 'f32' };
  const state = { sourceLayout: 'gguf-v1', headKDim: 128, headVDim: 128,
    numKHeads: 16, numVHeads: 32, qSize: 2048, kSize: 2048, valueDim: 4096,
    convDim: 8192, convKernelSize: 4, qRep: 2, normMode: 'shared', rmsNormEps: 1e-6,
    recurrentStatePrecision: 'f32', recurrentStateComponents: 1 };
  for (const key of ['convWeightGPU', 'convStateGPU', 'dtBiasGPU', 'aLogGPU', 'normWeightGPU', 'recurrentStateGPU']) {
    state[key] = new GPUBuffer();
  }
  const options = { normGateActivation: 'silu', outputDtype: 'f32', qkL2NormEps: 1e-6,
    runtimeSession: { useLinearAttentionRegisterPrefill: true,
      useLinearAttentionRegisterDecode: false, useLinearAttentionFusedDecodeCore: false } };
  for (const recorded of [false, true]) {
    for (const numTokens of [2047, 2048, 2049]) {
      dispatches.length = 0;
      const temporary = [];
      const recorder = recorded ? { device, getEncoder() {},
        trackTemporaryBuffer(buffer) { temporary.push(buffer); },
        recordDispatch(pipeline, _bindings, grid) { dispatches.push({ label: pipeline.label, grid, width: pipeline.compute.constants.WORKGROUP_SIZE }); },
      } : null;
      const output = await runLinearAttentionCoreGPU(tensor, tensor, tensor, tensor, state,
        { ...options, numTokens, recorder });
      const parallel = dispatches.find(row => row.label === 'linear_attention_parallel_conv');
      assert(parallel, 'Registered parallel convolution selected');
      assert(parallel.grid.every(n => Number.isSafeInteger(n) && n > 0 && n <= 65535));
      assert.equal(parallel.grid[0] * parallel.grid[1] * parallel.grid[2] * parallel.width,
        numTokens * state.convDim, 'The entire convolution output is covered at the failed boundary');
      convGridProbe.release(output.buffer);
      for (const buffer of temporary) convGridProbe.release(buffer);
      await Promise.resolve();
      assert.equal(active.size, 0, 'Success releases all owned scratch');
    }
    for (const numTokens of [65536, 1.5]) {
      const before = allocations;
      await assert.rejects(runLinearAttentionCoreGPU(tensor, tensor, tensor, tensor, state,
        { ...options, numTokens, recorder: recorded ? { device, getEncoder() {} } : null }),
      /maxComputeWorkgroupsPerDimension|positive safe integer/);
      assert.equal(allocations, before, 'Rejected geometry cannot acquire output or scratch');
      assert.equal(active.size, 0);
    }
  }
  console.log('linear-attention-conv-grid: immediate and recorded boundary coverage and rejection passed (GPU mocked)');
} finally {
  hooks.deregister(); delete globalThis.convGridProbe;
  if (originalBuffer === undefined) delete globalThis.GPUBuffer; else globalThis.GPUBuffer = originalBuffer;
  if (originalStage === undefined) delete globalThis.GPUShaderStage; else globalThis.GPUShaderStage = originalStage;
}
