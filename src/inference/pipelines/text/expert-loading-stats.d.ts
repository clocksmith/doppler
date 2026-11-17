import type { DopplerLoader } from '../../../loader/doppler-loader.js';
export interface ExpertLoadingStats {
  schema: 'doppler.expert-loading-stats/v1';
  scope: string;
  profilerEnabled: boolean;
  routedExpertCache: ReturnType<DopplerLoader['getRoutedExpertCacheStats']>;
  routedReadback: ReturnType<DopplerLoader['getRoutedReadbackStats']>;
}
export interface ExpertLoadingStatsOwner {
  isLoaded?: boolean;
  isGenerating?: boolean;
  dopplerLoader?: Pick<DopplerLoader, 'getRoutedExpertCacheStats' | 'getRoutedReadbackStats'> | null;
  runtimeConfig?: { shared?: { debug?: { profiler?: { enabled?: boolean } } } };
}
/** Detached CPU-only snapshot. Requires a loaded, supported owner. */
export function snapshotExpertLoadingStats(owner: ExpertLoadingStatsOwner): ExpertLoadingStats;
