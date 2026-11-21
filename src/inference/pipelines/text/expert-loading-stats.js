/** Explicit boundary snapshot only. This never acquires weights or reads GPU data.
 * @type {typeof import('./expert-loading-stats.js').snapshotExpertLoadingStats} */
export function snapshotExpertLoadingStats(owner) {
  if (owner.isLoaded !== true) throw new Error('Expert loading statistics require a loaded text pipeline.');
  if (owner.isGenerating === true) throw new Error('Expert loading statistics require an idle model boundary.');
  const loader = owner.dopplerLoader;
  if (!loader || typeof loader.getRoutedExpertCacheStats !== 'function'
    || typeof loader.getRoutedReadbackStats !== 'function') {
    throw new Error('Loaded pipeline does not support expert loading statistics.');
  }
  return structuredClone({ schema: 'doppler.expert-loading-stats/v1',
    scope: 'Loader-lifetime counters; async wait sums may overlap. Peaks and gauges are not additive. No tensor data or GPU timing.',
    profilerEnabled: owner.runtimeConfig?.shared?.debug?.profiler?.enabled === true,
    routedExpertCache: loader.getRoutedExpertCacheStats(), routedReadback: loader.getRoutedReadbackStats() });
}

/** @type {WeakMap<object, { selectionAwaitMs: number, preparationAwaitMs: number }>} */
const exposedWaits = new WeakMap();

/** @type {typeof import('./expert-loading-stats.js').routedPrefillWaits} */
export function routedPrefillWaits(loader) {
  let counters = exposedWaits.get(loader);
  if (!counters) {
    counters = { selectionAwaitMs: 0, preparationAwaitMs: 0 };
    exposedWaits.set(loader, counters);
  }
  return counters;
}

/** Selected additive counters only. GPU work and asynchronous source operations can overlap.
 * @type {typeof import('./expert-loading-stats.js').readPrefillExpertCounters} */
export function readPrefillExpertCounters(loader) {
  const cache = loader.getRoutedExpertCacheStats();
  // The normal loader creates the routed cache lazily, often during the first prompt.
  const prepare = cache?.phaseProfiles.prefill;
  const readback = loader.getRoutedReadbackStats().phases.prefill;
  return {
    ...routedPrefillWaits(loader),
    preparations: prepare?.preparations ?? null,
    prepareWallMs: prepare?.prepareWallMs ?? null,
    routeReadbacks: readback?.calls ?? 0,
    routeReadbackWallMs: readback?.wallMs ?? 0,
    sourceReads: cache?.sourceReads ?? 0,
    sourceReadBytes: cache?.sourceReadBytes ?? 0,
    sourceReadWaitMs: cache?.sourceReadWaitMs ?? 0,
    uploadBytes: cache?.uploadBytes ?? 0,
    cacheHits: cache?.cacheHits ?? 0,
    cacheMisses: cache?.cacheMisses ?? 0,
    initialQueueIdleForWeightsMs: cache?.initialQueueIdleForWeightsMs ?? 0,
  };
}
