import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';
import { ggufBlockBytes } from '../../src/formats/gguf/block-storage.js';

const wrapperURL = new URL('../../src/gpu/kernels/moe-grouped.js', import.meta.url).href;
const groupedURL = new URL('../../src/inference/pipelines/text/moe/grouped-prefill.js', import.meta.url).href;
const executorURL = new URL('../../src/gpu/kernels/kernel-execution.js', import.meta.url).href;
const opsURL = new URL('../../src/inference/pipelines/text/ops.js', import.meta.url).href;
const limits = { maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupStorageSize: 16384, maxStorageBuffersPerShaderStage: 8,
  maxComputeWorkgroupsPerDimension: 65535, maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxBufferSize: 256 * 1024 * 1024 };
const previousUsage = globalThis.GPUBufferUsage;
globalThis.GPUBufferUsage = { STORAGE: 128, INDIRECT: 256, COPY_SRC: 4 };
const calls = [], acquired = [], disposed = [];
const buffer = (size, usage, label) => ({ size, usage, label });
globalThis.groupedOrderedExecutionTest = {
  device: { limits }, fail: null,
  acquire(size, usage = GPUBufferUsage.STORAGE, label) {
    const value = buffer(size, usage, label); acquired.push(value); return value;
  },
  release(value) { disposed.push({ value, mode: 'release' }); },
  async dispatch(op, recorder, variant, bindings, uniforms, workgroups, constants) {
    calls.push({ op, recorded: recorder !== null, variant, bindings, uniforms, workgroups, constants });
    if (this.fail === op) throw Error(`injected ${op}`);
  },
  activate(input, options) {
    if (this.fail === 'activate') throw Error('injected activate');
    assert.equal(input.shape[1], options.dim * 2);
    return { dtype: 'f32', shape: [options.numTokens, options.dim],
      buffer: this.acquire(options.numTokens * options.dim * 4, GPUBufferUsage.STORAGE, 'activated') };
  },
  scatter(input, _weights, options) {
    if (this.fail === 'scatter') throw Error('injected scatter');
    assert.equal(input.shape[0], options.numTokens * options.topK);
    return { dtype: 'f32', shape: [options.numTokens, options.hiddenSize],
      buffer: this.acquire(options.numTokens * options.hiddenSize * 4, GPUBufferUsage.STORAGE, 'scatter') };
  },
};
const poolModule = 'export const acquireBuffer=(...args)=>groupedOrderedExecutionTest.acquire(...args);'
  + 'export const releaseBuffer=b=>groupedOrderedExecutionTest.release(b);'
  + 'export const readBuffer=()=>{throw Error("Unexpected readback in CPU fixture");};'
  + 'export const uploadData=()=>{throw Error("Unexpected upload in CPU fixture");};';
const modules = {
  [wrapperURL]: {
    '../device.js': 'export const getDevice=()=>groupedOrderedExecutionTest.device;',
    './kernel-execution.js': `export { withKernelOutput } from ${JSON.stringify(executorURL)};
      export const unifiedKernelWrapper=(...args)=>groupedOrderedExecutionTest.dispatch(...args);`,
  },
  [executorURL]: { '../../memory/buffer-pool.js': poolModule },
  [opsURL]: { '../../../memory/buffer-pool.js': poolModule },
  [groupedURL]: {
    '../../../../gpu/device.js': 'export const getDevice=()=>groupedOrderedExecutionTest.device;',
    '../../../../memory/buffer-pool.js': poolModule,
    '../../../../gpu/kernels/silu.js': `export const runSiLURowSplit=async(...args)=>groupedOrderedExecutionTest.activate(...args);
      export const recordSiLURowSplit=async(_r,...args)=>groupedOrderedExecutionTest.activate(...args);`,
    '../../../../gpu/kernels/moe-route.js': `export const runRouteScatter=async(...args)=>groupedOrderedExecutionTest.scatter(...args);
      export const recordRouteScatter=async(_r,...args)=>groupedOrderedExecutionTest.scatter(...args);`,
  },
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  const source = modules[context.parentURL]?.[specifier];
  if (source) return { url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true };
  return next(specifier, context);
} });

try {
  const { executeGroupedPrefill } = await import(groupedURL);
  const geometry = { numTokens: 33, numExperts: 256, topK: 8, hiddenSize: 96, intermediateSize: 32 };
  const weight = (shape, dtype) => ({ dtype, layout: 'row', shape,
    buffer: buffer(Math.ceil(ggufBlockBytes(shape, dtype) / 4) * 4, GPUBufferUsage.STORAGE, 'borrowed-weight') });
  const options = { ...geometry, swigluLimit: null,
    inputTensor: { dtype: 'f32', shape: [33, 96], buffer: buffer(33 * 96 * 4, GPUBufferUsage.STORAGE, 'borrowed-input') },
    indicesBuffer: buffer(33 * 8 * 4, GPUBufferUsage.STORAGE, 'borrowed-indices'),
    weightsBuffer: buffer(33 * 8 * 4, GPUBufferUsage.STORAGE, 'borrowed-routing-weights'),
    weights: { numExperts: 256, gate: weight([256, 32, 96], 'q4_0'), up: weight([256, 32, 96], 'q4_0'),
      down: weight([256, 96, 32], 'q4_1') } };
  const program = variant => ({ prefill: { steps: ['moe_grouped_sort', 'moe_grouped_gu', 'moe_grouped_down'].map(op => {
    const config = getKernelConfig(op, op === 'moe_grouped_sort' ? 'native_f32' : variant);
    return { op, kernel: config.shaderFile, entry: config.entryPoint };
  }) } });
  const ordered = program('native_f32_ordered');
  const legacy = program('native_f32');
  const path = { ...legacy, layerOverrides: [{ layers: [1], prefill: ordered.prefill }] };
  for (const recorded of [false, true]) for (const layerIdx of [0, 1]) {
    for (const failure of [null, 'moe_grouped_sort', 'moe_grouped_gu', 'activate', 'moe_grouped_down', 'scatter']) {
      calls.length = 0; acquired.length = 0; disposed.length = 0;
      groupedOrderedExecutionTest.fail = failure;
      const recorder = recorded ? { device: groupedOrderedExecutionTest.device,
        beginComputePass() {}, trackTemporaryBuffer(value) { disposed.push({ value, mode: 'track' }); } } : null;
      const execution = executeGroupedPrefill(recorder, { ...options, kernelPath: path, phase: 'prefill', layerIdx });
      let output = null;
      if (failure) await assert.rejects(execution, /injected/);
      else output = await execution;
      const selectedVariant = layerIdx === 1 ? 'native_f32_ordered' : 'native_f32';
      const sort = calls[0];
      assert.equal(sort.op, 'moe_grouped_sort');
      assert.deepEqual([sort.uniforms.gu_groups, sort.uniforms.down_groups], layerIdx === 1 ? [4, 3] : [1, 1],
        'Sort must receive the projection variant geometry for indirect dispatch');
      for (const call of calls.slice(1)) {
        assert.equal(call.variant, selectedVariant);
        assert.equal(call.recorded, recorded);
        assert.equal(call.workgroups.indirectBuffer, sort.bindings[2]);
        assert.equal(call.workgroups.indirectOffset, call.op === 'moe_grouped_gu' ? 0 : 12);
        assert.equal(call.uniforms.num_routes, 264);
        assert.equal(call.constants.QUANT_FORMAT, call.op === 'moe_grouped_gu' ? 0 : 1,
          'Projection dtype must come from each actual packed bank');
      }
      const owned = acquired.filter(value => value !== output?.buffer);
      assert.equal(disposed.length, owned.length, `${selectedVariant}/${failure}: each scratch output disposed exactly once`);
      assert.equal(new Set(disposed.map(item => item.value)).size, owned.length);
      assert.deepEqual(new Set(disposed.map(item => item.value)), new Set(owned));
      assert(disposed.every(item => item.mode === (recorded ? 'track' : 'release')));
      if (output) assert.deepEqual(output.shape, [33, 96]);
    }
  }
  calls.length = 0; acquired.length = 0; disposed.length = 0; groupedOrderedExecutionTest.fail = null;
  await assert.rejects(executeGroupedPrefill(null, { ...options, kernelPath: ordered, phase: 'decode', layerIdx: 0 }),
    /Explicit prefill phase/);
  assert.equal(calls.length, 0); assert.equal(acquired.length, 0, 'Invalid selection must fail before allocation');
  console.log('moe-grouped-ordered-execution: selected dispatch, indirect geometry and run/record cleanup passed (CPU mocks; no numerical evidence)');
} finally {
  hooks.deregister(); delete globalThis.groupedOrderedExecutionTest;
  if (previousUsage === undefined) delete globalThis.GPUBufferUsage;
  else globalThis.GPUBufferUsage = previousUsage;
}
