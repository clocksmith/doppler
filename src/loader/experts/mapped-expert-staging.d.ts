import type { CompactLayerStats, RoutedExpertCacheOptions } from './routed-expert-cache.js';
import type { RoutedExpertBankPlan, RoutedExpertPlane } from './routed-expert-plan.js';
import type { RoutedExpertCounters } from './routed-expert-source.js';

/** A ring of mapped upload buffers fed by direct verified reads; one per storage port. */
export interface MappedExpertStaging {
  readonly slotBytes: number;
  readonly slotCount: number;
  /** GPU bytes the ring currently owns. */
  readonly bytes: number;
  /** Resolves false when nothing was staged and nothing was submitted. */
  upload(parts: ReadonlyArray<{ shard: number; offset: number; length: number; at: number }>,
    copies: ReadonlyArray<{ at: number; length: number; target: GPUBuffer; targetOffset: number }>,
    signal: AbortSignal, beforeSubmit: (() => void) | null): Promise<boolean>;
  destroy(): Promise<void>;
}

/** null unless policy.uploadStaging is 'mapped', reads are batched leases, and the port serves direct verified reads. */
export function getMappedExpertStaging(options: RoutedExpertCacheOptions, plan: RoutedExpertBankPlan): MappedExpertStaging | null;
export function destroyMappedExpertStaging(options: RoutedExpertCacheOptions): Promise<void>;
/** Resolves false when the caller must load these experts through its lease path instead. */
export function stageRoutedExperts(staging: MappedExpertStaging, stats: RoutedExpertCounters, plan: RoutedExpertBankPlan,
  experts: ReadonlyArray<number>, targetOf: (expert: number, plane: RoutedExpertPlane) => { buffer: GPUBuffer; offset: number },
  signal: AbortSignal, beforeSubmit: (() => void) | null, layerStats: CompactLayerStats | null): Promise<boolean>;
