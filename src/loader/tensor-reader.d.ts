/**
 * Tensor Reader - Low-level tensor data assembly from shards.
 *
 * @module loader/tensor-reader
 */

import type { TensorLocation } from './loader-types.js';

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
  loadShard: (index: number) => Promise<ArrayBuffer>
): Promise<Uint8Array>;
