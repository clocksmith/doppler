import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
const target = new URL('../../src/inference/pipelines/text/loading/init-weights.js', import.meta.url).href;
const modules = {
  '../../../../loader/doppler-loader.js': 'export const createDopplerLoader=()=>{throw Error("unexpected factory");};',
  '../../../../debug/index.js': 'export const log={debug(){},verbose(){}};',
  '../generator/session/session-context.js': 'export const resolvePerLayerInputsSession=()=>null;',
  './load-policy.js': 'export const resolveShardHashVerification=()=>({verifyHashes:false});',
  './weight-loading-config.js': 'export const resolveWeightLoadingConfig=()=>({});',
  './load-errors.js': 'export const rewriteWeightLoadError=e=>e;',
};
const hook = registerHooks({ resolve(specifier, context, next) {
  return context.parentURL === target && modules[specifier]
    ? { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true } : next(specifier, context);
} });
try {
  const { loadWeights } = await import(target);
  const events = [];
  let finish, failPreload = false, failCleanup = false;
  const signal = new AbortController().signal;
  const loader = { setQ4KConfig() {}, setLoaderDebugConfig() {}, setPerLayerInputSession() {}, setTensorsJsonUrl() {},
    setTensorsJsonLoader() {}, async init() {}, async load(_id, options) { assert.equal(options.signal, signal); events.push('load'); },
    getRoutedExpertPolicy: () => ({ enabled: true, storageMode: 'tiered-compact' }),
    async prepareRoutedExpertResidency(value) { assert.equal(value, signal); events.push('preload'); if (failPreload) throw Error('preload failed'); await new Promise(resolve => { finish = resolve; }); },
    async unload() { events.push('unload'); if (failCleanup) throw Error('retirement failed'); },
    getLayerWeights() { events.push('weights'); return {}; },
  };
  const invoke = () => loadWeights({ modelId: 'fixture', shards: [] }, { numLayers: 1, useMoE: false },
    { loader, storageContext: {}, signal });
  let completed = false;
  const loading = invoke().then(value => { completed = true; return value; });
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(events, ['load', 'preload']); assert.equal(completed, false);
  finish(); const result = await loading;
  assert.equal(result.loader, loader); assert.deepEqual(events, ['load', 'preload', 'weights']);
  events.length = 0; failPreload = true;
  await assert.rejects(invoke(), /preload failed/);
  assert.deepEqual(events, ['load', 'preload', 'unload']);
  failCleanup = true;
  await assert.rejects(invoke(), error => error instanceof AggregateError && error.errors.length === 2);
  failCleanup = false; failPreload = false; events.length = 0;
  loader.getRoutedExpertPolicy = () => ({ enabled: true, storageMode: 'compact-decode' });
  await invoke(); assert.deepEqual(events, ['load', 'weights'], 'old modes remain on the ordinary public load path');
} finally { hook.deregister(); }
console.log('public weight initialization waits tiered preload and retires local loader on preload failure; legacy unchanged');
