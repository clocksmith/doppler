/**
 * Tensor Reader - Low-level tensor data assembly from shards.
 *
 * @module loader/tensor-reader
 */

import type { ShardLoadOptions, TensorLocation } from '../loader-types.js';

/**
 * Assemble tensor data from single or multiple shards.
 *
 * @param location - Tensor location info (shard index, offset, size, spans)
 * @param name - Tensor name (for logging)
 * @param loadShard - Callback to load a shard by index
 * @returns Uint8Array containing the full tensor data
 */
export declare function assembleShardData(
  location: TensorLocation,
  name: string,
  loadShard: (index: number, options?: ShardLoadOptions) => Promise<ArrayBuffer>,
  loadShardRange?: (index: number, offset: number, length: number) => Promise<ArrayBuffer>,
  options?: {
    materializeSourceTransform?: boolean;
  }
): Promise<Uint8Array>;

export declare function loadTensorRange(
  location: TensorLocation,
  name: string,
  byteOffset: number,
  byteLength: number,
  loadShardRange: (index: number, offset: number, length: number) => Promise<ArrayBuffer>,
  /** Optional caller-owned destination fill. Source must not retain or mutate it after resolution. */
  loadShardRangeInto?: ((index: number, offset: number, destination: Uint8Array) => Promise<number>) | null
): Promise<Uint8Array>;
