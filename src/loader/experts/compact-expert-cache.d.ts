import type { CompactLayerStats, RoutedExpertCacheOptions, RoutedExpertWeights } from './routed-expert-cache.js';
import type { RoutedExpertBankPlan } from './routed-expert-plan.js';
import type { RoutedExpertCounters } from './routed-expert-source.js';
export interface CompactExpertCacheOptions extends RoutedExpertCacheOptions { compactLayerCount: number }
export declare function compactExpertCapacity(plan: RoutedExpertBankPlan, options: Pick<CompactExpertCacheOptions, 'device' | 'policy' | 'compactLayerCount'>): number;
export declare class CompactExpertCache {
  constructor(options: CompactExpertCacheOptions, stats: RoutedExpertCounters);
  prepare(plan: RoutedExpertBankPlan, ids: number[], signal: AbortSignal, residentPlanes?: ReadonlyMap<string, GPUBuffer> | null): Promise<RoutedExpertWeights>;
  getResidentSlots(layer: number, ids: number[]): Map<number, number>;
  encodeResidentCopies(plan: RoutedExpertBankPlan, slots: ReadonlyMap<number, number>,
    destinations: ReadonlyMap<string, GPUBuffer>, encoder: GPUCommandEncoder): void;
  clearStaging(): void;
  hasLayer(layer: number): boolean;
  readonly residentBytes: number;
  getStats(): { cacheBytes: number; workingBytes: number; experts: number; compactLayers: Record<string, CompactLayerStats & { capacity: number }> };
  destroy(): Promise<void>;
}

export interface CompactExpertPool {
  stats: CompactLayerStats;
  capacity: number;
  planes: Map<string, GPUBuffer>;
  map: GPUBuffer;
  entries: Map<number, number>;
  free: number[];
}
