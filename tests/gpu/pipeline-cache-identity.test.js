import assert from 'node:assert/strict';
import { test } from 'node:test';
import { setDevice } from '../../src/gpu/device.js';
import { clearShaderCaches, registerShaderSources } from '../../src/gpu/kernels/shader-cache.js';
import {
  clearPipelineCaches, getCachedPipeline, getPipelineFast,
  getOrCreateBindGroupLayout, getOrCreatePipelineLayout,
} from '../../src/gpu/kernels/pipeline-cache.js';

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function fixture(t) {
  const loss = deferred();
  const device = {
    features: new Set(), limits: {}, lost: loss.promise,
    createBindGroup: descriptor => descriptor,
    createBuffer: descriptor => ({ ...descriptor, destroy() {} }),
    queue: { submit() {}, writeBuffer() {} },
    createBindGroupLayout: descriptor => ({ ...descriptor }),
    createPipelineLayout: descriptor => ({ ...descriptor }),
    createShaderModule: descriptor => ({ ...descriptor }),
    async createComputePipelineAsync(descriptor) { return { ...descriptor }; },
  };
  setDevice(device, { platformConfig: null });
  clearShaderCaches(); clearPipelineCaches();
  registerShaderSources({ 'scale.wgsl': 'source A' });
  t.after(() => { clearPipelineCaches(); clearShaderCaches(); setDevice(null, { platformConfig: null }); });
  return { device, loss };
}

const entry = (buffer = {}) => ({ binding: 0, visibility: 4, buffer });

test('explicit devices retain provider WGSL requirements and reject missing features', async t => {
  const { device } = fixture(t);
  device.features.add('subgroups');
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'navigator', previous);
    else delete globalThis.navigator;
  });
  const provider = { wgslLanguageFeatures: new Set() };
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu: provider } });
  registerShaderSources({ 'rmsnorm_stats_subgroups.wgsl': 'source with subgroup_id' });
  await assert.rejects(getPipelineFast('rmsnorm_stats', 'subgroups', null, null, device), /subgroup_id/);
  provider.wgslLanguageFeatures.add('subgroup_id');
  setDevice(null, { platformConfig: null });
  assert.ok(await getPipelineFast('rmsnorm_stats', 'subgroups', null, null, device));
});

test('layout identity uses descriptors, including WebGPU defaults, and ordered layout objects', t => {
  const { device } = fixture(t);
  const uniform = getOrCreateBindGroupLayout('shared', [entry()], device);
  const storage = getOrCreateBindGroupLayout('shared', [entry({ type: 'storage' })], device);
  assert.notEqual(uniform, storage);
  assert.equal(getOrCreateBindGroupLayout('renamed', [entry({ type: 'uniform', hasDynamicOffset: false, minBindingSize: 0 })], device), uniform);
  for (const buffer of [{ hasDynamicOffset: true }, { minBindingSize: 16 }, { type: 'read-only-storage' }]) {
    assert.notEqual(getOrCreateBindGroupLayout('shared', [entry(buffer)], device), uniform);
  }
  const first = getOrCreatePipelineLayout('shared', [uniform, storage], device);
  assert.notEqual(getOrCreatePipelineLayout('shared', [storage, uniform], device), first);
  assert.equal(getOrCreatePipelineLayout('renamed', [uniform, storage], device), first);
  const other = { ...device };
  assert.notEqual(getOrCreateBindGroupLayout('shared', [entry()], other), uniform);
  const pair = [entry(), { binding: 1, visibility: 4, sampler: {} }];
  assert.equal(getOrCreateBindGroupLayout('pair', pair, device), getOrCreateBindGroupLayout('reverse', pair.toReversed(), device));
  for (const resource of ['sampler', 'texture', 'storageTexture', 'externalTexture']) {
    const a = { binding: 0, visibility: 4, [resource]: resource === 'storageTexture' ? { format: 'rgba8unorm' } : {} };
    const layout = getOrCreateBindGroupLayout('resource', [a], device);
    assert.notEqual(getOrCreateBindGroupLayout('resource', [{ ...a, visibility: 2 }], device), layout);
  }
});

test('compute cache distinguishes actual layouts and shader bytes', async t => {
  const { device } = fixture(t);
  const a = device.createBindGroupLayout({ label: 'shared', entries: [entry()] });
  const b = device.createBindGroupLayout({ label: 'shared', entries: [entry({ type: 'storage' })] });
  const first = await getPipelineFast('scale', 'default', a);
  assert.notEqual(await getPipelineFast('scale', 'default', b), first);
  const original = await getPipelineFast('scale', 'default');
  registerShaderSources({ 'scale.wgsl': 'source B' });
  const changed = await getPipelineFast('scale', 'default');
  assert.notEqual(changed, original);
  assert.equal(changed.compute.module.code, 'source B');
});

test('concurrent compilation is shared, failures retry, and sync lookup never returns a promise', async t => {
  const { device } = fixture(t);
  const compilation = deferred();
  const started = deferred();
  let calls = 0;
  device.createComputePipelineAsync = () => { calls++; started.resolve(); return compilation.promise; };
  const first = getPipelineFast('scale', 'default');
  const second = getPipelineFast('scale', 'default');
  await started.promise;
  await new Promise(resolve => setImmediate(resolve));
  const concurrentCalls = calls;
  assert.equal(getCachedPipeline('scale', 'default'), null);
  compilation.reject(new Error('compile failed'));
  await Promise.all([assert.rejects(first, /compile failed/), assert.rejects(second, /compile failed/)]);
  assert.equal(concurrentCalls, 1);
  device.createComputePipelineAsync = async descriptor => { calls++; return { ...descriptor }; };
  const retried = await getPipelineFast('scale', 'default');
  assert.equal(getCachedPipeline('scale', 'default'), retried);
  assert.equal(calls, 2);
});

test('device loss during compilation rejects the result and cannot populate the cache', async t => {
  const { device, loss } = fixture(t);
  const compilation = deferred(), started = deferred();
  device.createComputePipelineAsync = () => { started.resolve(); return compilation.promise; };
  const result = getPipelineFast('scale', 'default');
  await started.promise;
  loss.resolve({ reason: 'unknown', message: 'injected loss' });
  await Promise.resolve();
  compilation.resolve({});
  await assert.rejects(result, /lost|changed|invalidated/i);
  assert.equal(getCachedPipeline('scale', 'default'), null);
});

test('device replacement while shader diagnostics are pending cannot compile a pipeline', async t => {
  const { device } = fixture(t);
  const diagnostics = deferred(), started = deferred();
  let calls = 0;
  device.createShaderModule = () => ({ getCompilationInfo() { started.resolve(); return diagnostics.promise; } });
  device.createComputePipelineAsync = async () => { calls++; return {}; };
  const result = getPipelineFast('scale', 'default');
  await started.promise;
  setDevice(null, { platformConfig: null });
  diagnostics.resolve({ messages: [] });
  await assert.rejects(result, /lost|changed/i);
  assert.equal(calls, 0);
});

test('clearing caches during compilation prevents late cache publication', async t => {
  const { device } = fixture(t);
  const compilation = deferred(), started = deferred();
  device.createComputePipelineAsync = () => { started.resolve(); return compilation.promise; };
  const result = getPipelineFast('scale', 'default');
  await started.promise; clearPipelineCaches(); compilation.resolve({});
  await result;
  assert.equal(getCachedPipeline('scale', 'default'), null);
});

test('explicit device preparation survives unrelated default-device replacement', async t => {
  const { device } = fixture(t);
  const diagnostics = deferred(), started = deferred();
  device.createShaderModule = () => ({ getCompilationInfo() { started.resolve(); return diagnostics.promise; } });
  const result = getPipelineFast('scale', 'default', null, null, device);
  await started.promise;
  setDevice(null, { platformConfig: null });
  diagnostics.resolve({ messages: [] });
  const pipeline = await result;
  assert.equal(getCachedPipeline('scale', 'default', null, device), pipeline);
});
