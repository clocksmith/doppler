import type { RoutedExpertCacheOptions, RoutedExpertCacheStats } from './routed-expert-cache.js';
import type { RoutedExpertPlane } from './routed-expert-plan.js';
export type RoutedExpertCounters = Omit<RoutedExpertCacheStats,
  'compactLayers' | 'phaseProfiles' | 'evictionPolicy' | 'frequencyHistoryEntries' | 'frequencyResidentEntries' |
  'cacheBytes' | 'workingBytes' | 'experts' | 'hostCacheBytes' | 'hostReservedBytes' | 'hostExperts'>;
export declare function createRoutedExpertCounters(): RoutedExpertCounters;
export declare function readRoutedExpertRange(options: RoutedExpertCacheOptions, stats: RoutedExpertCounters,
  plane: RoutedExpertPlane, expert: number, offset: number, length: number, signal: AbortSignal, destination?: Uint8Array | null): Promise<Uint8Array>;
