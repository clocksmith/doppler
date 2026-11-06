import { getDevice } from '../../gpu/device.js';
import { getBufferPool } from '../../memory/buffer-pool.js';
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

/** @type {typeof import('./loader-bindings.js').prepareRoutedExperts} */
export async function prepareRoutedExperts(loader, layer, expertIds, signal) {
  const policy = loader.getRoutedExpertPolicy();
  if (!policy.enabled) throw new Error('Routed experts are not enabled.');
  if (!loader.manifest) throw new Error('Routed experts require the loaded manifest.');
  const plan = planRoutedExpertBank(loader.manifest, loader.tensorLocations, layer);
  if (!loader.routedExpertCache) {
    const device = getDevice();
    if (!device) throw new Error('Routed experts require the owning GPU device.');
    loader.routedExpertCache = new RoutedExpertCache({ device, policy,
      otherGpuBytes: () => getBufferPool(device).getStats().currentBytesAllocated,
      readRange: async (shard, offset, length, activeSignal) => {
        activeSignal.throwIfAborted();
        const bytes = await loader.shardCache.loadRange(shard, offset, length, { requireRange: true });
        activeSignal.throwIfAborted();
        return bytes;
      } });
  }
  return loader.routedExpertCache.prepare(plan, expertIds, signal);
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
