/**
 * Kernel Tuner Types
 *
 * Type definitions and interfaces for the kernel auto-tuner system.
 */

/**
 * Device information for cache keys
 */
export interface DeviceInfo {
  vendor: string;
  architecture: string;
  device: string;
  description?: string;
}

/**
 * Tuning result for a kernel
 */
export interface TuneResult {
  optimalWorkgroupSize: [number, number, number];
  optimalTileSize: number;
  throughput: number;
  timeMs: number;
  deviceInfo: DeviceInfo | undefined;
}

/**
 * Tuning record stored in cache
 */
export interface TuneRecord {
  optimalWorkgroupSize: [number, number, number];
  optimalTileSize: number;
  throughput: number;
  timeMs: number;
  deviceInfo: DeviceInfo | undefined;
}

/**
 * Tuning configuration options
 */
export interface TuneConfig {
  warmup?: number;
  iterations?: number;
  forceRetune?: boolean;
}

/**
 * Input sizes for kernel tuning
 */
export interface InputSizes {
  M?: number;
  N?: number;
  K?: number;
  seqLen?: number;
  numHeads?: number;
  headDim?: number;
  innerSize?: number;
  outerSize?: number;
  hiddenSize?: number;
  numTokens?: number;
  numBlocks?: number;
}

/**
 * Workgroup size candidate
 */
export type WorkgroupSize = [number, number, number];

/**
 * Device limits from GPU
 */
export interface DeviceLimits {
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
  maxComputeWorkgroupSizeX: number;
  maxComputeWorkgroupSizeY: number;
  maxComputeWorkgroupSizeZ: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupStorageSize: number;
  maxStorageBuffersPerShaderStage: number;
}

/**
 * Kernel capabilities from GPU
 */
export interface KernelCapabilities {
  hasSubgroups: boolean;
  hasSubgroupsF16: boolean;
  hasF16: boolean;
  hasTimestampQuery: boolean;
  maxBufferSize: number;
  maxWorkgroupSize: number;
  maxWorkgroupStorageSize: number;
  adapterInfo: DeviceInfo;
}

/**
 * Cache storage key type
 */
export type CacheKey = string;
