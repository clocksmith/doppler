/**
 * Sliding Window KV Cache - For long sequences with limited memory
 *
 * Only keeps the most recent N tokens in the cache, using ring-buffer
 * semantics for GPU storage.
 *
 * @module inference/kv-cache/sliding-window
 */

import { KVCache } from './base.js';
import {
  type KVCacheConfig,
  type MemoryStats,
} from './types.js';

// ============================================================================
// SlidingWindowKVCache Class
// ============================================================================

/**
 * Sliding Window KV Cache for long sequences
 * Only keeps the most recent N tokens
 */
export class SlidingWindowKVCache extends KVCache {
  readonly windowSize: number;
  totalTokensSeen: number;

  /**
   * @param config - Configuration with windowSize
   */
  constructor(config: KVCacheConfig & { windowSize?: number });

  /**
   * Update with sliding window logic
   */
  update(
    layerIdx: number,
    keys: Float32Array | GPUBuffer,
    values: Float32Array | GPUBuffer,
    startPos?: number
  ): void;

  /**
   * GPU-native update with ring-buffer semantics.
   * Keeps the last `windowSize` tokens in GPU memory while allowing
   * unbounded absolute positions for RoPE.
   */
  updateFromGPU(
    layerIdx: number,
    keysBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    startPos: number,
    numTokens: number
  ): void;

  /**
   * Record KV cache update with ring-buffer semantics to an external encoder.
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

  getMemoryStats(): MemoryStats & { windowSize: number; totalTokensSeen: number };
}
