import type { RoutedExpertCacheOptions } from './routed-expert-cache.js';
import type { RoutedExpertBankPlan } from './routed-expert-plan.js';
import type { RoutedExpertCounters } from './routed-expert-source.js';
export function writeRoutedPrefillExpert(options: RoutedExpertCacheOptions, stats: RoutedExpertCounters,
  plan: RoutedExpertBankPlan, expert: number, signal: AbortSignal, working: ReadonlyMap<string, GPUBuffer>,
  scratch: Uint8Array | null, beforeUpload: () => void): Promise<void>;
export function selectPrefillSeedExperts(ids: number[], capacity: number): number[];
