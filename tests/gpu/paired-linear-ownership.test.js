import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { createTensor } from '../../src/gpu/tensor.js';
const target = new URL('../../src/gpu/kernels/paired-linear-attention.js', import.meta.url).href;
const live = new Set(), calls = [], limits = { maxStorageBuffersPerShaderStage: 9, maxComputeWorkgroupsPerDimension: 65535, maxStorageBufferBindingSize: 2 ** 28, maxBufferSize: 2 ** 28 };
globalThis.pairedLinearControl = {
  failure: null,
  acquire(size, _usage, label) { if (this.failure === label) throw Error('injected allocation'); const b = { size }; live.add(b); return b; },
  release(b) { assert(live.delete(b), 'one release per acquired buffer'); },
  dispatch(...args) { if (this.failure === args[0]) throw Error('injected dispatch'); calls.push(args); },
  device: { limits },
};
const modules = {
  '../../memory/buffer-pool.js': 'export const acquireBuffer=(...a)=>pairedLinearControl.acquire(...a); export const releaseBuffer=b=>pairedLinearControl.release(b);',
  './kernel-execution.js': 'export const unifiedKernelWrapper=async(...a)=>pairedLinearControl.dispatch(...a);',
  '../device.js': 'export const getDevice=()=>pairedLinearControl.device;',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  return next(specifier, context);
} });
const { executePairedLinearAttention } = await import(target);
const representation = { precision: 'two-f32', components: 2, elementStrideBytes: 8 };
const tensor = count => createTensor({ size: count * 8 }, 'f32', [count], 'pair', representation);
const qkv = tensor(12), z = tensor(4), a = tensor(2), b = tensor(2);
const state = { convDim: 6, convKernelSize: 4, valueDim: 2, numVHeads: 1, numKHeads: 1,
  headKDim: 2, headVDim: 2, qSize: 2, kSize: 2, qRep: 1, normMode: 'shared', rmsNormEps: 1e-6,
  convStatePrecision: 'two-f32', convStateComponents: 2, normWeightOffset: true,
  recurrentStatePrecision: 'two-f32', recurrentStateComponents: 2,
  convStateGPU: { size: 192 }, recurrentStateGPU: { size: 32 }, convWeightGPU: { size: 96 },
  dtBiasGPU: { size: 4 }, aLogGPU: { size: 4 }, normWeightGPU: { size: 8 } };
try {
  for (const recorded of [false, true]) {
    const held = new Set(), recorder = recorded ? { device: { limits }, trackTemporaryBuffer: b => held.add(b) } : null;
    const settle = () => { for (const b of held) pairedLinearControl.release(b); held.clear(); };
    const run = (overrides = {}, options = {}) => executePairedLinearAttention(qkv, z, a, b, { ...state, ...overrides },
      { numTokens: 2, outputDtype: 'f32', qkL2NormEps: 1e-6, recorder, ...options });
    for (const failure of ['paired_linear_conv', 'paired_linear_gates', 'paired_linear_output', 'gated_delta_conv', 'qwen_linear_attention_prepare', 'gated_delta_recurrent']) {
      pairedLinearControl.failure = failure; await assert.rejects(run(), /injected/); settle(); assert.equal(live.size, 0);
    }
    pairedLinearControl.failure = null;
    const countBeforeLimit = calls.length;
    limits.maxStorageBufferBindingSize = 31;
    await assert.rejects(run(), /Prepared gate buffer/);
    assert.equal(calls.length, countBeforeLimit); assert.equal(live.size, 0);
    limits.maxStorageBufferBindingSize = 2 ** 28;
    const result = await run(); assert.equal(result.precision, 'two-f32'); assert.deepEqual(result.shape, [2, 2]);
    assert.equal(held.size, recorded ? 2 : 0, 'conv and gate owners remain with recorder until settlement');
    assert.equal(calls.at(-2)[0], 'qwen_linear_attention_prepare');
    const preparedGate = calls.at(-2)[3].at(-1);
    assert.equal(preparedGate.size, 32);
    assert.equal(calls.at(-1)[3][2], preparedGate, 'state update consumes exact paired gate buffer');
    const u = calls.at(-1)[4]; assert.equal(u.rms_eps_hi + u.rms_eps_lo, 1e-6); assert.equal(u.norm_weight_offset, 1);
    pairedLinearControl.release(result.buffer); settle(); assert.equal(live.size, 0);
    for (const override of [{ headKDim: 129 }, { convStatePrecision: 'f32' }, { convStateComponents: 1 },
      { normWeightOffset: undefined }, { recurrentStateGPU: state.convStateGPU }, { convStateGPU: qkv.buffer }, { normWeightGPU: { size: 4 } }]) {
      const count = calls.length; await assert.rejects(run(override)); assert.equal(calls.length, count); assert.equal(live.size, 0);
    }
    await assert.rejects(run({}, { qkL2NormEps: undefined }), /epsilons/);
    limits.maxStorageBuffersPerShaderStage = 8; await assert.rejects(run(), /Nine storage/); limits.maxStorageBuffersPerShaderStage = 9;
  }
} finally { hooks.deregister(); delete globalThis.pairedLinearControl; }
console.log('paired linear: explicit layouts, epsilons, aliases, limits, allocation and both dispatch rollback passed');
