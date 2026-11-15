import { getRuntimeConfig } from '../../config/runtime.js';
import { getDevice } from '../../gpu/device.js';
import { getBufferPool } from '../../memory/buffer-pool.js';
import { compactExpertCapacity } from './compact-expert-cache.js';
import { RoutedExpertCache } from './routed-expert-cache.js';
import { planRoutedExpertBank } from './routed-expert-plan.js';
import { prepareQwenExpertLayer } from './qwen-paging.js';

/** @type {typeof import('./loader-bindings.js').prepareExpertLayer} */
export async function prepareExpertLayer(loader, policy, layer, consumers, recorder, signal, activeSelection) {
  if (policy.qwen35Routed.enabled) {
    if (recorder || activeSelection !== 'topk-route') throw new Error('Routed Qwen experts require immediate topk-route execution.');
    return null;
  }
  return prepareQwenExpertLayer(loader._getExpertLoaderContext(), layer, consumers, policy.qwen35Pager, recorder, signal);
}

/** @type {typeof import('./loader-bindings.js').getRoutedExpertCapacity} */
export function getRoutedExpertCapacity(loader, layer, phase) {
  if (!loader.manifest) throw new Error('Routed experts require the loaded manifest.');
  const plan = planRoutedExpertBank(loader.manifest, loader.tensorLocations, layer);
  const policy = loader.getRoutedExpertPolicy();
  if (policy.storageMode !== 'compact-decode' || phase !== 'decode') return plan.numExperts;
  const device = getDevice();
  if (!device) throw new Error('Compact experts require the owning device.');
  return compactExpertCapacity(plan, { device, policy, compactLayerCount: plan.numLayers });
}

/** @type {typeof import('./loader-bindings.js').prepareRoutedExperts} */
export async function prepareRoutedExperts(loader, layer, expertIds, signal, phase) {
  const policy = loader.getRoutedExpertPolicy();
  if (!policy.enabled) throw new Error('Routed experts are not enabled.');
  if (!loader.manifest) throw new Error('Routed experts require the loaded manifest.');
  const plan = planRoutedExpertBank(loader.manifest, loader.tensorLocations, layer);
  if (!loader.routedExpertCache) {
    const device = getDevice();
    if (!device) throw new Error('Routed experts require the owning GPU device.');
    loader.routedExpertCache = new RoutedExpertCache({ device, policy, compactLayerCount: plan.numLayers,
      numLayers: typeof loader.manifest.architecture === 'object'
        && typeof loader.manifest.architecture?.numLayers === 'number'
        ? loader.manifest.architecture.numLayers : undefined, numExperts: plan.numExperts,
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
      } });
  }
  return loader.routedExpertCache.prepare(plan, expertIds, signal, phase);
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
