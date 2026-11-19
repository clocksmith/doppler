import { getRuntimeConfig } from '../../config/runtime.js';
import { getDevice } from '../../gpu/device.js';
import { getBufferPool } from '../../memory/buffer-pool.js';
import { compactExpertCapacity } from './compact-expert-cache.js';
import { TieredExpertCache } from './tiered-expert-cache.js';
import { RoutedExpertCache } from './routed-expert-cache.js';
import { planRoutedExpertBank } from './routed-expert-plan.js';
import { prepareQwenExpertLayer } from './qwen-paging.js';
import { trace, isTraceEnabled } from '../../debug/index.js';
import { assertStagedVerification } from './mapped-expert-staging.js';

/** @type {typeof import('./loader-bindings.js').prepareExpertLayer} */
export async function prepareExpertLayer(loader, policy, layer, consumers, recorder, signal, activeSelection) {
  if (policy.qwen35Routed.enabled) {
    if (recorder || activeSelection !== 'topk-route') throw new Error('Routed Qwen experts require immediate topk-route execution.');
    return null;
  }
  return prepareQwenExpertLayer(loader._getExpertLoaderContext(), layer, consumers, policy.qwen35Pager, recorder, signal);
}

/** @type {typeof import('./loader-bindings.js').getRoutedExpertLayerCount} */
export function getRoutedExpertLayerCount(loader, layer, numLayers) {
  const count = loader.layers.size;
  if (!Number.isSafeInteger(numLayers) || numLayers < 1 || count < 1 || count > numLayers
    || !Number.isSafeInteger(layer) || layer < 0 || layer >= numLayers || !loader.layers.has(layer)) {
    throw new Error('Routed expert residency requires the loaded partition layers at their original indices.');
  }
  return count;
}

/** @type {WeakMap<TieredExpertCache, Map<number, import('./routed-expert-plan.js').RoutedExpertBankPlan>>} */
const tieredPlans = new WeakMap();

/** @param {import('../doppler-loader.js').DopplerLoader} loader
 * @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan */
function ensureRoutedCache(loader, plan) {
  if (loader.routedExpertCache) return loader.routedExpertCache;
  const policy = loader.getRoutedExpertPolicy();
  const device = getDevice();
  if (!device) throw new Error('Routed experts require the owning GPU device.');
  /** @type {import('./routed-expert-cache.js').RoutedExpertCacheOptions} */
  const options = { device, policy, compactLayerCount: getRoutedExpertLayerCount(loader, plan.layer, plan.numLayers),
    numLayers: plan.numLayers, numExperts: plan.numExperts,
    rangeLease: loader.shardCache.rangeLease ?? undefined,
    readRangeInto: loader.shardCache.hasCustomRangeIntoLoader ? async (shard, offset, destination, activeSignal) => {
      activeSignal.throwIfAborted();
      const count = await loader.shardCache.loadRangeInto(shard, offset, destination, { requireRange: true });
      activeSignal.throwIfAborted();
      return count;
    } : undefined,
    otherGpuBytes: () => getBufferPool(device).getCurrentBytesAllocated(),
    readRange: async (shard, offset, length, activeSignal) => {
      activeSignal.throwIfAborted();
      const bytes = await loader.shardCache.loadRange(shard, offset, length, { requireRange: true });
      activeSignal.throwIfAborted();
      return bytes;
    } };
  if (policy.storageMode === 'tiered-compact') {
    if (!loader.manifest) throw new Error('Tiered experts require a loaded manifest.');
    const manifest = loader.manifest;
    const minimumSlots = manifest.moeConfig?.numExpertsPerToken;
    if (!Number.isSafeInteger(minimumSlots) || typeof minimumSlots !== 'number' || minimumSlots < 1) {
      throw new Error('Tiered experts require the manifest numExpertsPerToken.');
    }
    const plans = new Map([...loader.layers.keys()].sort((a, b) => a - b).map(layer =>
      [layer, planRoutedExpertBank(manifest, loader.tensorLocations, layer)]));
    const cache = new TieredExpertCache(options, [...plans.values()], minimumSlots);
    tieredPlans.set(cache, plans);
    loader.routedExpertCache = cache;
  } else loader.routedExpertCache = new RoutedExpertCache(options);
  return loader.routedExpertCache;
}

/** @type {typeof import('./loader-bindings.js').getRoutedExpertCapacity} */
export function getRoutedExpertCapacity(loader, layer, phase) {
  if (!loader.manifest) throw new Error('Routed experts require the loaded manifest.');
  const plan = planRoutedExpertBank(loader.manifest, loader.tensorLocations, layer);
  const policy = loader.getRoutedExpertPolicy();
  if (policy.storageMode === 'tiered-compact') {
    const cache = ensureRoutedCache(loader, plan);
    if (!(cache instanceof TieredExpertCache)) throw new Error('Tiered expert policy does not match the owning cache.');
    const capacity = cache.capacity(layer);
    if (capacity === null) throw new Error('Tiered expert capacity requires an assigned original layer.');
    return capacity;
  }
  if (policy.storageMode !== 'compact-decode' || phase !== 'decode') return plan.numExperts;
  const device = getDevice();
  if (!device) throw new Error('Compact experts require the owning device.');
  return compactExpertCapacity(plan, { device, policy, compactLayerCount: getRoutedExpertLayerCount(loader, layer, plan.numLayers) });
}

/** @type {typeof import('./loader-bindings.js').prepareRoutedExpertResidency} */
export async function prepareRoutedExpertResidency(loader, signal) {
  const policy = loader.getRoutedExpertPolicy();
  if (!policy.enabled || policy.storageMode !== 'tiered-compact') return;
  if (!loader.manifest) throw new Error('Tiered experts require a loaded manifest.');
  const layer = loader.layers.keys().next().value;
  if (typeof layer !== 'number') throw new Error('Tiered experts require assigned loaded layers.');
  const cache = ensureRoutedCache(loader, planRoutedExpertBank(loader.manifest, loader.tensorLocations, layer));
  if (!(cache instanceof TieredExpertCache)) throw new Error('Tiered residency requires the owning tiered cache.');
  await cache.preload(signal);
}

/** @type {typeof import('./loader-bindings.js').getResidentRoutedExpertWeights} */
export function getResidentRoutedExpertWeights(loader, layer) {
  if (loader.getRoutedExpertPolicy().storageMode !== 'tiered-compact') return null;
  return loader.routedExpertCache instanceof TieredExpertCache ? loader.routedExpertCache.getResidentWeights(layer) : null;
}

/** @type {typeof import('./loader-bindings.js').prepareRoutedExperts} */
export async function prepareRoutedExperts(loader, layer, expertIds, signal, phase) {
  const policy = loader.getRoutedExpertPolicy();
  if (!policy.enabled) throw new Error('Routed experts are not enabled.');
  if (!loader.manifest) throw new Error('Routed experts require the loaded manifest.');
  const plan = planRoutedExpertBank(loader.manifest, loader.tensorLocations, layer);
  const cache = ensureRoutedCache(loader, plan);
  // After a failed deferred verification, unverified experts are resident in the pools. No caller gets them,
  // whether or not it opens a release gate, and whether it would upload or only hit the cache.
  if (loader.shardCache.rangeLease) assertStagedVerification(loader.shardCache.rangeLease);
  // With the loader trace on, one line when a layer asks for its experts and one when they are ready for
  // its kernels. Together with the staged-upload lines these put read, hash, upload and use on one clock.
  const traced = isTraceEnabled('loader');
  if (traced) trace.loader(`routed-experts request ${JSON.stringify({ layer, phase, experts: expertIds.length, at: Number(performance.now().toFixed(2)) })}`);
  let weights;
  if (cache instanceof TieredExpertCache) {
    const admitted = tieredPlans.get(cache)?.get(layer);
    if (!admitted) throw new Error('Tiered expert request lacks the admitted original layer plan.');
    weights = await cache.prepare(admitted, expertIds, signal, phase);
  } else weights = await cache.prepare(plan, expertIds, signal, phase);
  if (traced) trace.loader(`routed-experts ready ${JSON.stringify({ layer, phase, at: Number(performance.now().toFixed(2)) })}`);
  return weights;
}

/** @type {typeof import('./loader-bindings.js').disposeRoutedExperts} */
export async function disposeRoutedExperts(loader) {
  if (loader.routedExpertCache) {
    try { await loader.routedExpertCache.destroy(); loader.routedExpertCache = null; }
    catch (error) { return [error]; }
  }
  return [];
}

/** @type {typeof import('./loader-bindings.js').createExpertLoaderContext} */
export function createExpertLoaderContext(loader) {
  return { manifest: loader.manifest, tensorLocations: loader.tensorLocations,
    loadTensor: (name, toGPU, silent) => loader._loadTensor(name, toGPU, silent),
    loadShard: loader._getLoadShard(), shardCache: loader.shardCache,
    expertCache: loader.expertCache, experts: loader.experts,
    gpuBuffers: loader.gpuBuffers, keepF32Weights: loader.keepF32Weights };
}

/** @type {typeof import('./loader-bindings.js').recordRoutedReadback} */
export function recordRoutedReadback(loader, phase, elapsedMs, bytes) {
  if (!getRuntimeConfig().shared.debug.profiler.enabled) return;
  const key = phase === 'prefill' || phase === 'decode' ? phase : 'unspecified';
  const bucket = loader.routedReadbackStats[key] ??= { calls: 0, bytes: 0, wallMs: 0 };
  bucket.calls++;
  bucket.bytes += bytes;
  bucket.wallMs += elapsedMs;
}

/** @type {typeof import('./loader-bindings.js').getRoutedReadbackStats} */
export function getRoutedReadbackStats(loader) {
  return {
    scope: 'Routing indices readback wall time includes queued GPU work, copy and map wait; not GPU routing compute. May overlap cache preparation in chunked prefill.',
    phases: Object.fromEntries(Object.entries(loader.routedReadbackStats).map(([phase, bucket]) => [phase, { ...bucket }])),
  };
}
