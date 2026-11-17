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
