import type { Qwen35RoutedConfig } from "../../config/schema/runtime/loading.schema.js";
import type { WeightBuffer } from '../../gpu/weight-buffer.js';
import type { RoutedExpertBankPlan } from './routed-expert-plan.js';
export interface RoutedExpertCacheOptions {
  device: GPUDevice;
  compactLayerCount?: number;
  policy: Qwen35RoutedConfig;
  numLayers?: number;
  numExperts?: number;
  readRange: (shard: number, offset: number, length: number, signal: AbortSignal) => Promise<ArrayBuffer>;
  readRangeInto?: (shard: number, offset: number, destination: Uint8Array, signal: AbortSignal) => Promise<number>;
  /** Authenticated owned ranges; the consumer releases only after upload copies bytes. */
  rangeLease?: import('../../storage/verified-piece-gather.js').VerifiedRangeLeasePort;
  otherGpuBytes: () => number;
}
export interface RoutedExpertEntry { buffer: GPUBuffer; bytes: number; users: number; lastUseSerial: number }
export interface RoutedHostExpertEntry { bytes: number; chunks: Array<{ offset: number; data: Uint8Array }> }
export interface RoutedExpertWeights {
  sourceNumExperts?: number;
  sourceExpertToSlot?: GPUBuffer;
  expertIdx: number;
  expertFormat: string;
  numExperts: number;
  expertIntermediateSize: number;
  gateUp?: WeightBuffer;
  gate?: WeightBuffer;
  up?: WeightBuffer;
  down: WeightBuffer;
}
/** Optional profiler observations. Operation wait sums overlap and are not a wall-time decomposition. */
export interface RoutedExpertPhaseProfile {
  preparations: number;
  failures: number;
  prepareWallMs: number;
  hostLookupMs: number;
  synchronizationWaitMs: number;
  peakGpuResidentBytes: number;
  peakHostOwnedOrReservedBytes: number;
  peakStagingBytes: number;
  counters: Record<string, number>;
}
export interface CompactLayerStats {
  cacheHits: number; cacheMisses: number; evictions: number; sourceReadBytes: number;
  uploadBytes: number; sourceReadWaitMs: number; uploadEnqueueMs: number;
}
export interface RoutedExpertCacheStats {
  prefillSeedBytes: number;
  prefillSeedSubmissions: number;
  prefillGpuReuseBytes: number;
  prefillGpuReuseExperts: number;
  prefillGpuReuseSubmissions: number;
  compactLayers?: Record<string, CompactLayerStats & { capacity: number }>;
  evictionPolicy: 'lru' | 'lfu';
  frequencyHistoryEntries: number;
  frequencyResidentEntries: number;
  sourceReadIntoBytes: number;
  stagingAllocations: number;
  stagingReuses: number;
  phaseProfiles: Record<string, RoutedExpertPhaseProfile>;
  sourceReadBytes: number;
  sourceReads: number;
  /** Coverage is required when interpreting the following per-job timing sums. */
  sourceTimingJobs: number;
  sourceUntimedJobs: number;
  /** Worker/read/verification/round-trip sums overlap; not a phase wall-time decomposition. */
  sourceJobWallMs: number;
  sourceJobReadWaitMs: number;
  sourceJobVerificationMs: number;
  sourceJobCopyMs: number;
  sourceJobRoundTripMs: number;
  uploadBytes: number;
  cacheHits: number;
  cacheMisses: number;
  workingHits: number;
  workingBufferAllocations: number;
  workingBufferAllocatedBytes: number;
  workingBufferReuses: number;
  workingBufferReusedBytes: number;
  evictions: number;
  evictionFenceWaits: number;
  evictionFenceSkips: number;
  bufferReuses: number;
  expertCopySubmissions: number;
  hitBatchSubmissions: number;
  batchedHitExperts: number;
  residentBytes: number;
  peakResidentBytes: number;
  peakAccountedGpuBytes: number;
  cacheBytes: number;
  workingBytes: number;
  peakStagingBytes: number;
  ioWaitMs: number;
  sourceReadWaitMs: number;
  uploadEnqueueMs: number;
  /** First proven client-queue idle gap per preparation; a lower bound, not global GPU utilization. */
  initialQueueIdleForWeightsMs: number;
  experts: number;
  hostPrefillAdmissionSkips: number;
  hostPrefillAdmissions: number;
  hostDecodeAdmissions: number;
  hostCacheHits: number;
  hostCacheMisses: number;
  hostEvictions: number;
  hostAdmissionSkips: number;
  peakHostOwnedOrReservedBytes: number;
  hostCacheBytes: number;
  hostReservedBytes: number;
  hostExperts: number;
}
export declare class RoutedExpertCache {
  constructor(options: RoutedExpertCacheOptions);
  prepare(plan: RoutedExpertBankPlan, expertIds: number[], signal?: AbortSignal, phase?: 'prefill' | 'decode'): Promise<RoutedExpertWeights>;
  getStats(): RoutedExpertCacheStats;
  destroy(): Promise<void>;
}
