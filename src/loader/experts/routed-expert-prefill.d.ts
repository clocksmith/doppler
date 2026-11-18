import type { RoutedExpertCacheOptions } from './routed-expert-cache.js';
import type { RoutedExpertBankPlan } from './routed-expert-plan.js';
import type { RoutedExpertCounters } from './routed-expert-source.js';
export interface BatchedDirectPrefillState {
  misses: number[];
  batchSize: number;
  taskCount: number;
  nextTask: () => number;
}
export function createBatchedDirectPrefillState(options: RoutedExpertCacheOptions, stats: RoutedExpertCounters,
  plan: RoutedExpertBankPlan, ids: number[], copiedHits: ReadonlySet<number>, loaded: ReadonlySet<number>,
  laneCount: number, onWorkingHit: (expert: number) => void): BatchedDirectPrefillState | null;
export function runBatchedDirectPrefillLane(state: BatchedDirectPrefillState, options: RoutedExpertCacheOptions,
  stats: RoutedExpertCounters, plan: RoutedExpertBankPlan, signal: AbortSignal, working: ReadonlyMap<string, GPUBuffer>,
  loaded: Set<number>, beforeBatch: () => void, beforeUpload: () => void): Promise<void>;
export function writeRoutedPrefillExpert(options: RoutedExpertCacheOptions, stats: RoutedExpertCounters,
  plan: RoutedExpertBankPlan, expert: number, signal: AbortSignal, working: ReadonlyMap<string, GPUBuffer>,
  scratch: Uint8Array | null, beforeUpload: () => void): Promise<void>;
export function writeRoutedPrefillExpertsBatch(options: RoutedExpertCacheOptions, stats: RoutedExpertCounters,
  plan: RoutedExpertBankPlan, experts: number[], signal: AbortSignal, working: ReadonlyMap<string, GPUBuffer>,
  beforeUpload: () => void): Promise<void>;
export function selectPrefillSeedExperts(ids: number[], capacity: number): number[];
