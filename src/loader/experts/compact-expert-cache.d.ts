import type { RoutedExpertCacheOptions, RoutedExpertWeights } from './routed-expert-cache.js';
import type { RoutedExpertBankPlan } from './routed-expert-plan.js';
import type { RoutedExpertCounters } from './routed-expert-source.js';
export interface CompactExpertCacheOptions extends RoutedExpertCacheOptions { compactLayerCount: number }
export declare function compactExpertCapacity(plan: RoutedExpertBankPlan, options: Pick<CompactExpertCacheOptions, 'device' | 'policy' | 'compactLayerCount'>): number;
export declare class CompactExpertCache {
  constructor(options: CompactExpertCacheOptions, stats: RoutedExpertCounters);
  prepare(plan: RoutedExpertBankPlan, ids: number[], signal: AbortSignal): Promise<RoutedExpertWeights>;
  getStats(): { cacheBytes: number; workingBytes: number; experts: number; compactLayers: Record<string, CompactLayerStats & { capacity: number }> };
  destroy(): Promise<void>;
}

export interface CompactLayerStats {
  cacheHits: number; cacheMisses: number; evictions: number; sourceReadBytes: number;
  uploadBytes: number; sourceReadWaitMs: number; uploadEnqueueMs: number;
}
export interface CompactExpertPool {
  stats: CompactLayerStats;
  capacity: number;
  planes: Map<string, GPUBuffer>;
  map: GPUBuffer;
  entries: Map<number, number>;
  free: number[];
}
