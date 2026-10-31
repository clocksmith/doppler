import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { resolveGatedDeltaNormGateKernel } from '../../src/gpu/kernels/gated-delta-norm-gate.js';

// Real rule consumer and wrapper validation; mock device stops before allocation.
const select = context => resolveGatedDeltaNormGateKernel('gated_delta_recurrent', 'gatedDeltaRecurrent',
  'scalarVariant', { inputDtype: 'f32', normGateActivation: 'silu', ...context });
assert.equal(select({ sourceLayout: 'gguf-v1', useRegisterPrefill: false }).variant, 'gguf_f32');
assert.equal(select({ sourceLayout: 'gguf-v1', useRegisterPrefill: true }).variant, 'gguf_register_prefill');
assert.equal(select({ sourceLayout: 'hf', useRegisterPrefill: true }).variant, 'register_prefill');
assert.equal(select({ sourceLayout: 'hf', useRegisterPrefill: false }).variant, 'default');

const target = new URL('../../src/gpu/kernels/linear-attention-core.js', import.meta.url).href;
globalThis.GPUBuffer = class { size = 4 * 4 * 128 * 128; };
globalThis.GPUShaderStage = { COMPUTE: 4 };
const modulesSeen = [];
globalThis.gdnSelectionProbe = {
  module(name) { modulesSeen.push(name); return { name }; },
  device: { createComputePipeline(descriptor) {
    if (descriptor.label.startsWith('linear_attention_recurrent_pipeline')) throw Error('selected recurrent pipeline');
    return {};
  } },
};
const modules = {
  '../device.js': 'export const getDevice = () => gdnSelectionProbe.device;',
  './shader-source-scope.js': 'export const getShaderScopeCacheKey = () => 0;',
  './shader-cache.js': 'export const getShaderModule = async (_device, name) => gdnSelectionProbe.module(name);',
  './pipeline-cache.js': 'export const getOrCreateBindGroupLayout = () => ({}); export const getOrCreatePipelineLayout = () => ({});',
  '../../memory/buffer-pool.js': 'export const acquireBuffer = () => { throw Error("unexpected allocation"); }; export const releaseBuffer = () => {};',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  return next(specifier, context);
} });
try {
  const { runLinearAttentionCoreGPU } = await import(target);
  const tensor = { buffer: new GPUBuffer(), dtype: 'f32' };
  const state = { sourceLayout: 'gguf-v1', numKHeads: 2, numVHeads: 4, headKDim: 128, headVDim: 128,
    qSize: 256, kSize: 256, valueDim: 512, convDim: 1024, convKernelSize: 4, qRep: 2,
    normMode: 'shared', rmsNormEps: 1e-6, recurrentStatePrecision: 'f32', recurrentStateComponents: 1 };
  for (const key of ['convWeightGPU', 'convStateGPU', 'recurrentStateGPU', 'dtBiasGPU', 'aLogGPU', 'normWeightGPU']) state[key] = new GPUBuffer();
  const session = { useLinearAttentionRegisterPrefill: true, useLinearAttentionRegisterDecode: false, useLinearAttentionFusedDecodeCore: false };
  const options = { numTokens: 3, outputDtype: 'f32', normGateActivation: 'silu', qkL2NormEps: 1e-6, runtimeSession: session };
  const run = (layer = state, opts = options, input = tensor) => runLinearAttentionCoreGPU(input, tensor, tensor, tensor, layer, opts);
  for (const patch of [{ headKDim: 64 }, { headVDim: 129 }, { recurrentStatePrecision: 'two-f32', recurrentStateComponents: 2 },
    { numKHeads: 0 }, { numKHeads: 3 }, { qSize: 128 }, { kSize: 128 }, { convDim: 1023 }, { valueDim: 511 }]) {
    await assert.rejects(() => run({ ...state, ...patch }), /registered head width|storage layout|scalar F32|declared Q\/K\/V geometry/);
  }
  await assert.rejects(() => run(state, { ...options, numTokens: 2.5 }), /integral token\/head counts/);
  await assert.rejects(() => run(state, options, { ...tensor, dtype: 'f16' }), /scalar F32/);
  await assert.rejects(() => run(state, options, { ...tensor, precision: 'two-f32' }), /scalar F32/);
  await assert.rejects(() => run(state, { ...options, normGateActivation: 'sigmoid' }), /scalar F32/);
  for (const flag of ['useLinearAttentionRegisterDecode', 'useLinearAttentionFusedDecodeCore']) {
    await assert.rejects(() => run(state, { ...options, numTokens: 1, runtimeSession: { ...session, [flag]: true } }), /declared unfused F32 recurrent program/);
  }
  assert.equal(modulesSeen.length, 0, 'all invalid requests rejected before shader preparation');
  for (const [enabled, numTokens, expected] of [[true, 3, 'gguf_register_prefill'], [false, 3, 'gguf_f32'], [true, 1, 'gguf_f32']]) {
    modulesSeen.length = 0;
    await assert.rejects(() => run(state, { ...options, numTokens, runtimeSession: { ...session, useLinearAttentionRegisterPrefill: enabled } }), /selected recurrent pipeline/);
    assert(modulesSeen.includes(select({ sourceLayout: 'gguf-v1', useRegisterPrefill: expected === 'gguf_register_prefill' }).shaderFile));
  }
} finally {
  hooks.deregister(); delete globalThis.gdnSelectionProbe;
}
console.log('gdn-gguf-prefill-selection: opt-in, source-layout, state/geometry rejection and decode isolation passed (mock GPU)');
