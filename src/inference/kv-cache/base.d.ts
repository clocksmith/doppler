/**
 * KV Cache Base - Core KV cache implementation
 *
 * Implements efficient key-value cache for transformer inference.
 * Supports both contiguous and paged memory layouts.
 * GPU-native storage to avoid CPU readbacks during inference.
 *
 * @module inference/kv-cache/base
 */

import {
  type KVCacheConfig,
  type ContiguousLayerCache,
  type PagedLayerCache,
  type LayerCache,
  type PageLocation,
  type KVGetResult,
  type GPUBuffersResult,
  type MemoryStats,
  type GPUContext,
} from './types.js';

// ============================================================================
// KVCache Class
// ============================================================================

export class KVCache {
  readonly numLayers: number;
  readonly numHeads: number;
  readonly headDim: number;
  readonly maxSeqLen: number;
  readonly layout: 'contiguous' | 'paged';
  readonly pageSize: number;
  readonly kvDtype: 'f16' | 'f32';
  readonly bytesPerElem: number;
  readonly kvSize: number;
  readonly windowSize?: number;

  useGPU: boolean;
  layers: LayerCache[];
  currentSeqLen: number;
  memoryUsage: number;
  gpuContext: GPUContext | null;

  /**
   * @param config - KV cache configuration
   */
  constructor(config: KVCacheConfig);

  /**
   * Update cache with new key-value pairs for a layer
   * @param layerIdx - Layer index
   * @param keys - New keys [batchSize, numHeads, headDim]
   * @param values - New values [batchSize, numHeads, headDim]
   * @param startPos - Starting position in sequence
   */
  update(
    layerIdx: number,
    keys: Float32Array | GPUBuffer,
    values: Float32Array | GPUBuffer,
    startPos?: number
  ): void;

  /**
   * Update cache directly from GPU buffers (zero-copy)
   */
  updateFromGPU(
    layerIdx: number,
    keysBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    startPos: number,
    numTokens: number
  ): void;

  /**
   * Record KV cache update to an external encoder (for batched GPU operations).
   * Does NOT submit - caller is responsible for submitting the encoder.
   */
  recordUpdateFromGPU(
    encoder: GPUCommandEncoder,
    layerIdx: number,
    keysBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    startPos: number,
    numTokens: number
  ): void;

  /**
   * Get cached keys and values for a layer
   */
  get(layerIdx: number, startPos?: number, endPos?: number): KVGetResult;

  /**
   * Get key cache buffer (GPU or CPU)
   */
  getKeyCache(layerIdx: number): GPUBuffer | Float32Array | null;

  /**
   * Get value cache buffer (GPU or CPU)
   */
  getValueCache(layerIdx: number): GPUBuffer | Float32Array | null;

  /**
   * Get GPU buffers for a layer (for GPU-native attention)
   */
  getGPUBuffers(layerIdx: number): GPUBuffersResult | null;

  /**
   * Check if GPU cache is available
   */
  hasGPUCache(): boolean;

  /**
   * Clear cache for all layers
   */
  clear(): void;

  /**
   * Clone the cache (for speculative decoding rollback)
   */
  clone(): KVCache;

  /**
   * Truncate cache to a specific length (for rollback)
   */
  truncate(length: number): void;

  /**
   * Get memory usage statistics
   */
  getMemoryStats(): MemoryStats;

  /**
   * Set GPU context for GPU-based caching
   */
  setGPUContext(gpuContext: GPUContext): void;

  /**
   * Sync GPU cache back to CPU (for debugging or fallback)
   */
  syncToCPU(): Promise<void>;

  /**
   * Destroy GPU resources
   */
  destroy(): void;
}
