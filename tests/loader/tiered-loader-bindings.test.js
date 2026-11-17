import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
const target = new URL('../../src/loader/experts/loader-bindings.js', import.meta.url).href;
const state = globalThis.tieredLoaderBinding = { events: [], plans: [], caches: [], failDestroy: false, device: {} };
const weights = { expertIdx: 0, expertFormat: 'qwen3_5', numExperts: 8, expertIntermediateSize: 2, down: {} };
state.weights = weights;
const modules = {
  '../../gpu/device.js': 'export const getDevice=()=>tieredLoaderBinding.device;',
  '../../memory/buffer-pool.js': 'export const getBufferPool=()=>({getCurrentBytesAllocated:()=>1234});',
  './routed-expert-plan.js': `export const planRoutedExpertBank=(manifest,locations,layer)=>{
    const plan={layer,numLayers:manifest.architecture.numLayers,numExperts:8};tieredLoaderBinding.plans.push(plan);return plan;};`,
  './tiered-expert-cache.js': `export class TieredExpertCache {
    constructor(options,plans,minimum){this.options=options;this.plans=plans;this.minimum=minimum;this.ready=false;tieredLoaderBinding.caches.push(this);}
    async preload(signal){signal?.throwIfAborted();this.ready=true;tieredLoaderBinding.events.push('preload');}
    getResidentWeights(layer){return this.ready&&layer===4?tieredLoaderBinding.weights:null;}
    capacity(layer){return this.plans.some(p=>p.layer===layer)?layer===4?8:2:null;}
    async prepare(plan,ids,signal,phase){if(!this.plans.includes(plan))throw Error('plan identity changed');tieredLoaderBinding.events.push({plan,ids,signal,phase});return tieredLoaderBinding.weights;}
    async destroy(){if(tieredLoaderBinding.failDestroy)throw Error('retirement blocked');tieredLoaderBinding.events.push('destroy');}
    getStats(){return {};}
  }`,
  './routed-expert-cache.js': `export class RoutedExpertCache {
    constructor(options){this.options=options;tieredLoaderBinding.caches.push(this);}
    async prepare(plan,ids,signal,phase){tieredLoaderBinding.events.push({plan,ids,signal,phase});return tieredLoaderBinding.weights;}
    async destroy(){}
  }`,
};
const hook = registerHooks({ resolve(specifier, context, next) {
  return context.parentURL === target && modules[specifier]
    ? { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true } : next(specifier, context);
} });
try {
  const api = await import(target);
  const signal = new AbortController().signal;
  const make = storageMode => ({ manifest: { architecture: { numLayers: 8 }, moeConfig: { numExpertsPerToken: 2 } },
    layers: new Map([[5, {}], [4, {}]]), tensorLocations: new Map(), routedExpertCache: null,
    getRoutedExpertPolicy: () => ({ enabled: true, storageMode }),
    shardCache: { hasCustomRangeIntoLoader: true, rangeLease: { identity: 'owned' },
      async loadRangeInto(shard, offset, destination, options) { assert.equal(options.requireRange, true); destination.fill(shard + offset); return destination.byteLength; },
      async loadRange(_shard, _offset, length, options) { assert.equal(options.requireRange, true); return new Uint8Array(length); } },
  });
  const loader = make('tiered-compact');
  assert.equal(api.getResidentRoutedExpertWeights(loader, 4), null);
  assert.equal(api.getRoutedExpertCapacity(loader, 5, 'prefill'), 2);
  const cache = loader.routedExpertCache;
  assert.deepEqual(cache.plans.map(plan => plan.layer), [4, 5]);
  assert.equal(cache.minimum, 2);
  assert.equal(cache.options.compactLayerCount, 2);
  assert.equal(cache.options.otherGpuBytes(), 1234);
  assert.equal(api.getResidentRoutedExpertWeights(loader, 4), null, 'no exposure before preload');
  await api.prepareRoutedExpertResidency(loader, signal);
  assert.equal(api.getResidentRoutedExpertWeights(loader, 4), weights);
  await api.prepareRoutedExperts(loader, 5, [3, 2], signal, 'prefill');
  await api.prepareRoutedExperts(loader, 5, [1], signal, 'decode');
  assert.equal(loader.routedExpertCache, cache);
  assert.ok(state.events.filter(event => event?.plan).every(event => cache.plans.includes(event.plan)));
  const destination = new Uint8Array(8);
  assert.equal(await cache.options.readRangeInto(1, 2, destination, signal), 8);
  assert.deepEqual([...destination], new Array(8).fill(3));
  const abort = new AbortController(); abort.abort(new Error('source cancelled'));
  await assert.rejects(cache.options.readRange(0, 0, 8, abort.signal), /source cancelled/);
  state.failDestroy = true;
  assert.equal((await api.disposeRoutedExperts(loader)).length, 1);
  assert.equal(loader.routedExpertCache, cache, 'failed destruction retains the retry owner');
  state.failDestroy = false;
  assert.deepEqual(await api.disposeRoutedExperts(loader), []);
  assert.equal(loader.routedExpertCache, null);
  const legacy = make('resident-bank');
  const before = state.caches.length;
  await api.prepareRoutedExpertResidency(legacy, signal);
  assert.equal(state.caches.length, before, 'old modes do not acquire a tiered preload path');
  assert.equal(api.getRoutedExpertCapacity(legacy, 5, 'prefill'), 8);
  await api.prepareRoutedExperts(legacy, 5, [2], signal, 'prefill');
  assert.equal(legacy.routedExpertCache.constructor.name, 'RoutedExpertCache');
  const invalid = make('tiered-compact'); invalid.manifest.moeConfig.numExpertsPerToken = null;
  assert.throws(() => api.getRoutedExpertCapacity(invalid, 5), /numExpertsPerToken/);
} finally { hook.deregister(); delete globalThis.tieredLoaderBinding; }
console.log('tiered loader bindings: assigned-layer planning, stable plan identity, preload admission and legacy behavior passed');
