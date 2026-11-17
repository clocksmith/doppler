import type { RoutedExpertBankPlan } from './routed-expert-plan.js';
import type { Qwen35RoutedConfig } from '../../config/schema/runtime/loading.schema.js';
export interface TieredExpertLayer {
  plan: RoutedExpertBankPlan;
  mode: 'resident' | 'parked' | 'streamed';
  capacity: number;
  quotaBytes: number;
}
export interface TieredExpertPlacement {
  layers: ReadonlyMap<number, TieredExpertLayer>;
  residentBytes: number;
  parkedBytes: number;
  mappedCacheBytes: number;
  compactBytes: number;
  cacheUpperBytes: number;
  compactLayerBytes: Record<string, number>;
}
/** Pure admission: original layer indices and source packed extents determine all storage. */
export function planTieredExpertPlacement(plans: readonly RoutedExpertBankPlan[], policy: Qwen35RoutedConfig,
  limits: Pick<GPUSupportedLimits, 'maxBufferSize' | 'maxStorageBufferBindingSize'>,
  minimumSlots: number): TieredExpertPlacement;
