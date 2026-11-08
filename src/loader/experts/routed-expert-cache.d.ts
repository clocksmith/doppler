import type { Qwen35RoutedConfig } from '../../config/schema/loading.schema.js';
import type { WeightBuffer } from '../../gpu/weight-buffer.js';
import type { RoutedExpertBankPlan } from './routed-expert-plan.js';
export interface RoutedExpertCacheOptions {
  device: GPUDevice;
  policy: Qwen35RoutedConfig;
  readRange: (shard: number, offset: number, length: number, signal: AbortSignal) => Promise<ArrayBuffer>;
  otherGpuBytes: () => number;
}
export interface RoutedExpertEntry { buffer: GPUBuffer; bytes: number; users: number }
export interface RoutedHostExpertEntry { bytes: number; chunks: Array<{ offset: number; data: Uint8Array }> }
export interface RoutedExpertWeights {
  expertIdx: number;
  expertFormat: string;
  numExperts: number;
  expertIntermediateSize: number;
  gateUp?: WeightBuffer;
  gate?: WeightBuffer;
  up?: WeightBuffer;
  down: WeightBuffer;
}
export interface RoutedExpertCacheStats {
  sourceReadBytes: number;
  sourceReads: number;
  uploadBytes: number;
  cacheHits: number;
  cacheMisses: number;
  workingHits: number;
  evictions: number;
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
  prepare(plan: RoutedExpertBankPlan, expertIds: number[], signal?: AbortSignal): Promise<RoutedExpertWeights>;
  getStats(): RoutedExpertCacheStats;
  destroy(): Promise<void>;
}
