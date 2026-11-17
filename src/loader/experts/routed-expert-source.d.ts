import type { RoutedExpertCacheOptions, RoutedExpertCacheStats } from './routed-expert-cache.js';
import type { RoutedExpertBankPlan, RoutedExpertPlane } from './routed-expert-plan.js';
export type RoutedExpertCounters = Omit<RoutedExpertCacheStats,
  'compactLayers' | 'phaseProfiles' | 'evictionPolicy' | 'frequencyHistoryEntries' | 'frequencyResidentEntries' |
  'cacheBytes' | 'workingBytes' | 'experts' | 'hostCacheBytes' | 'hostReservedBytes' | 'hostExperts'>;
export declare function createRoutedExpertCounters(): RoutedExpertCounters;
export declare function readRoutedExpertRange(options: RoutedExpertCacheOptions, stats: RoutedExpertCounters,
  plane: RoutedExpertPlane, expert: number, offset: number, length: number, signal: AbortSignal, destination?: Uint8Array | null): Promise<Uint8Array>;

export declare function assertExpertReadShape(options: RoutedExpertCacheOptions, plan: RoutedExpertBankPlan): void;
export declare function readRoutedExpertLease(options: RoutedExpertCacheOptions, stats: RoutedExpertCounters,
  plan: RoutedExpertBankPlan, expert: number, signal: AbortSignal,
  layerStats: import('./routed-expert-cache.js').CompactLayerStats | null): Promise<import('../../storage/verified-piece-gather.js').VerifiedRangeLease>;
export declare function readRoutedExpertsLease(options: RoutedExpertCacheOptions, stats: RoutedExpertCounters,
  plan: RoutedExpertBankPlan, experts: number[], signal: AbortSignal,
  layerStats: import('./routed-expert-cache.js').CompactLayerStats | null): Promise<import('../../storage/verified-piece-gather.js').VerifiedRangeLease>;
/** Authenticate one expert's planes on distinct lanes before returning any lease.
 * The caller owns every returned plane lease and must release all after uploads. */
export declare function readRoutedExpertPlaneLeases(options: RoutedExpertCacheOptions, stats: RoutedExpertCounters,
  plan: RoutedExpertBankPlan, expert: number, signal: AbortSignal,
  layerStats: import('./routed-expert-cache.js').CompactLayerStats | null): Promise<import('../../storage/verified-piece-gather.js').VerifiedRangeLease[]>;
export declare function readRoutedExpertPayload(options: RoutedExpertCacheOptions, stats: RoutedExpertCounters,
  plan: RoutedExpertBankPlan, expert: number, signal: AbortSignal, destination: Uint8Array,
  layerStats: import('./routed-expert-cache.js').CompactLayerStats | null): Promise<Uint8Array>;
