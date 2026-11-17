import type { RoutedExpertCacheOptions, RoutedExpertCacheStats, RoutedExpertWeights } from './routed-expert-cache.js';
import type { RoutedExpertBankPlan } from './routed-expert-plan.js';
import type { MappedExpertUploadStats } from './mapped-expert-upload-cache.js';
export declare class TieredExpertCache {
  constructor(options: RoutedExpertCacheOptions, plans: readonly RoutedExpertBankPlan[], minimumSlots: number);
  preload(signal?: AbortSignal): Promise<void>;
  getResidentWeights(layer: number): RoutedExpertWeights | null;
  capacity(layer: number): number | null;
  prepare(plan: RoutedExpertBankPlan, ids: number[], signal?: AbortSignal, phase?: 'prefill' | 'decode'): Promise<RoutedExpertWeights>;
  getStats(): RoutedExpertCacheStats & { tiered: { residentLayers: number[]; ready: boolean; mapped: MappedExpertUploadStats | null } };
  destroy(): Promise<void>;
}
