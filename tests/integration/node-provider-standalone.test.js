import assert from 'node:assert/strict';
import { openNodeWebGPU as openProvider } from '../../src/tooling/provider-v1-contract.js';
import { bootstrapNodeWebGPU, bootstrapNodeWebGPUProvider, releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';

// These are host-composition/lifecycle tests, not hardware evidence.
const names = ['GPUBufferUsage', 'GPUShaderStage', 'GPUMapMode', 'GPUTextureUsage'];
const snapshots = new Map([...names, 'navigator'].map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
const moduleSource = `
  export const globals = Object.fromEntries(${JSON.stringify(names)}.map(name => [name, class { static marker = name; }]));
  export const gpu = { requestAdapter: async options => ({ options, requestDevice() {} }) };
  export function create(args) { if (args[0] === 'fail') throw new Error('factory rejected input'); return gpu; }
  export const nested = { marker: true, create() { if (!this.marker) throw new Error('lost factory receiver'); return { instance: gpu }; } };
`;
const moduleUrl = `data:text/javascript,${encodeURIComponent(moduleSource)}`;
const provider = (id = 'synthetic') => ({ id, kind: 'module', module: moduleUrl,
  gpu: { kind: 'factory', path: 'create', args: [[]] },
  globals: Object.fromEntries(names.map(name => [name, `globals.${name}`])) });
const options = (providers = [provider()], mode = 'replace') => ({ providers, adapterOptions: null, globals: { mode } });
const reset = () => {
  for (const [name, descriptor] of snapshots) {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else Reflect.deleteProperty(globalThis, name);
  }
};
try {
  const sentinel = { gpu: { marker: 'application-owned' } };
  Object.defineProperty(globalThis, 'navigator', { value: sentinel, configurable: true });
  const session = await openProvider(options());
  assert.equal(session.receipt.implementation, 'doppler');
  assert.equal(session.receipt.selectedProviderId, 'synthetic');
  assert.equal(globalThis.navigator, sentinel, 'preserve the application navigator object');
  assert.equal(globalThis.navigator.gpu, session.gpu);
  await session.close();
  assert.equal(sentinel.gpu.marker, 'application-owned');
  assert.equal(session.gpu, null);
  assert.equal(session.adapter, null);
  assert.equal(session.module, null);
  assert.equal(session.receipt.globals.restored, true);
  await session.close();
  reset();

  const missing = { ...provider('missing'), module: 'doppler-intentionally-absent-provider' };
  const ordered = await openProvider(options([missing, provider('selected')]));
  assert.deepEqual(ordered.receipt.attempts.map(attempt => [attempt.providerId, attempt.ok]), [['missing', false], ['selected', true]]);
  await ordered.close();
  await assert.rejects(openProvider(options([missing])), error => {
    assert.equal(error.receipt.attempts.length, 1, 'never discover an undeclared alternative');
    assert.equal(error.receipt.selectedProviderId, null);
    return /No authorized/.test(error.message);
  });
  const failing = provider(); failing.gpu.args = [['fail']];
  await assert.rejects(openProvider(options([failing])), /factory rejected input/);
  const noAdapter = { ...provider(), module: `data:text/javascript,${encodeURIComponent(moduleSource.replace('return gpu;', 'return { requestAdapter: async () => null };'))}` };
  await assert.rejects(openProvider(options([noAdapter])), /no usable WebGPU adapter/);
  const missingGlobals = provider(); missingGlobals.globals.GPUMapMode = 'globals.missing';
  await assert.rejects(openProvider(options([missingGlobals])), /missing GPUMapMode/);

  for (const invalid of [
    { ...options(), adapterOptions: undefined },
    { ...options(), adapterOptions: { powerPreference: 'invented' } },
    { ...options(), unknown: true }, options([]), options([provider(), provider()]),
    { ...options(), globals: { mode: 'automatic' } },
    options([{ ...provider(), gpu: { kind: 'factory', path: 'create' } }]),
    options([{ ...provider(), gpu: { kind: 'export', path: '__proto__.gpu' } }]),
  ]) await assert.rejects(openProvider(invalid), error => error.code === 'DOPPLER_PROVIDER_CONFIG_INVALID');

  const nested = provider(); nested.gpu = { kind: 'factory', path: 'nested.create', args: [], resultPath: 'instance' };
  const none = await openProvider(options([nested], 'none'));
  assert.equal(none.receipt.globals.installed.length, 0);
  await none.close();
  Object.defineProperty(globalThis, 'GPUShaderStage', { value: { sentinel: true }, configurable: true });
  await assert.rejects(openProvider(options([provider()], 'install-missing')), /already belongs to another provider/);
  assert.deepEqual(Object.getOwnPropertyDescriptor(globalThis, 'GPUBufferUsage'), snapshots.get('GPUBufferUsage'), 'partial installation rolls back');
  reset();

  const exported = await import(moduleUrl);
  Object.defineProperty(globalThis, 'navigator', { value: { gpu: exported.gpu }, configurable: true });
  for (const name of names) Object.defineProperty(globalThis, name, { value: exported.globals[name], configurable: true });
  const globalSession = await openProvider(options([{ id: 'pre-installed', kind: 'global' }]));
  assert.equal(globalSession.receipt.globals.installed.length, 0);
  await globalSession.close();
  assert.equal(globalThis.navigator.gpu, exported.gpu, 'never destroy or remove caller-owned GPU');

  const first = bootstrapNodeWebGPU();
  await assert.rejects(bootstrapNodeWebGPUProvider(moduleUrl), error => error.code === 'DOPPLER_PROVIDER_ALREADY_ACTIVE');
  const result = await first;
  assert.equal(result.ok, true);
  assert.equal(result.receipt.implementation, 'doppler', 'default bootstrap must not depend on Doe');
  await result.session.close();
  assert.equal((await releaseNodeWebGPU()).released, false, 'direct close releases bootstrap ownership');
  reset();

  for (let cycle = 0; cycle < 5; cycle++) {
    const current = await bootstrapNodeWebGPUProvider(moduleUrl);
    assert.equal(current.ok, true);
    const closed = await releaseNodeWebGPU();
    assert.equal(closed.receipt.globals.restored, true);
    assert.equal(current.session.gpu, null);
  }
  const conflict = await openProvider(options());
  const owned = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { thirdParty: true };
  await assert.rejects(conflict.close(), /restoration failed/);
  assert.equal(globalThis.GPUBufferUsage.thirdParty, true, 'do not clobber external changes');
  globalThis.GPUBufferUsage = owned;
  await conflict.close();
  const lifecycleModule = `data:text/javascript,${encodeURIComponent(`
    export const NODE_WEBGPU_PROVIDER_SCHEMA = 'doe.webgpu-provider/v1';
    export const state = { closeCalls: 0, opened: null, release: null, finish: null, fail: false };
    export async function openNodeWebGPU() {
      await new Promise(resolve => { state.opened = resolve; });
      return Object.freeze({ gpu: {}, adapter: {}, module: null, receipt: { selectedProviderId: 'lifecycle' },
        async close() {
          state.closeCalls++;
          if (state.fail) throw new Error('explicit cleanup failure');
          await new Promise(resolve => { state.finish = resolve; });
        } });
    }
  `)}`;
  const { state } = await import(lifecycleModule);
  const opening = bootstrapNodeWebGPU({ providerContractModule: lifecycleModule });
  const releaseDuringOpen = releaseNodeWebGPU();
  while (!state.opened) await new Promise(resolve => setImmediate(resolve));
  state.opened();
  const active = await opening;
  while (!state.finish) await new Promise(resolve => setImmediate(resolve));
  await assert.rejects(bootstrapNodeWebGPUProvider(moduleUrl), error => error.code === 'DOPPLER_PROVIDER_ALREADY_ACTIVE');
  const repeatedClose = active.session.close();
  state.finish();
  assert.equal((await releaseDuringOpen).released, true);
  await repeatedClose;
  assert.equal(state.closeCalls, 1, 'parallel closes share one cleanup');

  state.opened = null; state.fail = true;
  const retryOpening = bootstrapNodeWebGPU({ providerContractModule: lifecycleModule });
  while (!state.opened) await new Promise(resolve => setImmediate(resolve));
  state.opened(); await retryOpening;
  await assert.rejects(releaseNodeWebGPU(), error => error.code === 'DOPPLER_PROVIDER_RELEASE_FAILED');
  await assert.rejects(bootstrapNodeWebGPUProvider(moduleUrl), error => error.code === 'DOPPLER_PROVIDER_ALREADY_ACTIVE');
  state.fail = false; state.finish = null;
  const retryClose = releaseNodeWebGPU();
  while (!state.finish) await new Promise(resolve => setImmediate(resolve));
  state.finish(); assert.equal((await retryClose).released, true);
} finally { await releaseNodeWebGPU(); reset(); }
console.log('node-provider-standalone.test: ok (synthetic host lifecycle, no Doe required)');
