import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

// Real wrapper, injected allocation failures. No claim of GPU numerical parity.
const target = new URL('../../src/gpu/kernels/linear-attention-core.js', import.meta.url).href;
globalThis.GPUBuffer = class {};
globalThis.GPUShaderStage = { COMPUTE: 4 };
const active = new Set();
let failure;
let fused;
globalThis.linearAllocationProbe = {
  device: { createComputePipeline: () => ({}) },
  acquire(_size, _usage, label) {
    if (failure === 'convolution' && label.includes('linear_conv_out')) throw Error('allocation rejected');
    const buffer = new GPUBuffer(); active.add(buffer); return buffer;
  },
  release(buffer) { assert(active.delete(buffer), 'release exactly once'); },
  uniform() { throw Error('allocation rejected'); },
  config: () => ({ inference: { session: { useLinearAttentionFusedDecodeCore: fused } } }),
};
const modules = {
  '../device.js': 'export const getDevice = () => linearAllocationProbe.device;',
  './shader-source-scope.js': 'export const getShaderScopeCacheKey = () => 0;',
  '../../memory/buffer-pool.js': `export const acquireBuffer = (...args) => linearAllocationProbe.acquire(...args);
    export const releaseBuffer = buffer => linearAllocationProbe.release(buffer);`,
  '../../config/runtime.js': 'export const getRuntimeConfig = () => linearAllocationProbe.config();',
  './uniform-utils.js': 'export const createUniformBufferFromData = () => linearAllocationProbe.uniform();',
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
  const state = { headVDim: 1, headKDim: 1, normMode: 'shared', convDim: 3,
    valueDim: 1, convKernelSize: 1, numVHeads: 1, numKHeads: 1,
    qSize: 1, kSize: 1, vSize: 1, qRep: 1 };
  for (const key of ['convWeightGPU', 'dtBiasGPU', 'aLogGPU', 'normWeightGPU', 'convStateGPU', 'recurrentStateGPU']) {
    state[key] = new GPUBuffer();
  }
  for (fused of [false, true]) {
    for (const recorded of [false, true]) {
      for (failure of fused ? ['uniform'] : ['convolution', 'uniform']) {
        const recorder = recorded ? { getEncoder() {}, trackTemporaryBuffer() {} } : null;
        await assert.rejects(() => runLinearAttentionCoreGPU(tensor, tensor, tensor, tensor, state,
          { numTokens: fused ? 1 : 2, abPacked: true, bProjOffsetElements: 1, outputDtype: 'f32', recorder }),
        /allocation rejected/);
        assert.equal(active.size, 0, `${failure}, fused=${fused}, recorded=${recorded}: no retained output`);
      }
    }
  }
} finally { hooks.deregister(); delete globalThis.linearAllocationProbe; }
console.log('linear-attention-allocation-cleanup: passed');
