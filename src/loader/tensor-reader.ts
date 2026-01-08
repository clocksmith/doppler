/**
 * Tensor Reader - Low-level tensor data assembly from shards.
 *
 * @module loader/tensor-reader
 */

import type { TensorLocation } from './loader-types.js';
import { trace } from '../debug/index.js';

/**
 * Assemble tensor data from single or multiple shards.
 *
 * @param location - Tensor location info (shard index, offset, size, spans)
 * @param name - Tensor name (for logging)
 * @param loadShard - Callback to load a shard by index
 * @returns Uint8Array containing the full tensor data
 */
export async function assembleShardData(
  location: TensorLocation,
  name: string,
  loadShard: (index: number) => Promise<ArrayBuffer>
): Promise<Uint8Array> {
  if (location.spans) {
    trace.loader(`Assembling tensor "${name}" from ${location.spans.length} spans`);
    const chunks: Uint8Array[] = [];
    for (const span of location.spans) {
      const data = await loadShard(span.shardIndex);
      if (span.offset + span.size > data.byteLength) {
        throw new Error(
          `[DopplerLoader] Shard ${span.shardIndex} too small for tensor "${name}" span.`
        );
      }
      chunks.push(new Uint8Array(data, span.offset, span.size));
    }
    const totalSize = chunks.reduce((s, c) => s + c.length, 0);
    const combined = new Uint8Array(totalSize);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }
    return combined;
  }

  // Single shard - use view to avoid copying
  const fullShard = await loadShard(location.shardIndex);
  // Boundary check
  if (location.offset + location.size > fullShard.byteLength) {
     throw new Error(
        `[DopplerLoader] Shard ${location.shardIndex} too small for tensor "${name}" (offset=${location.offset}, size=${location.size}, shard=${fullShard.byteLength})`
      );
  }
  return new Uint8Array(fullShard, location.offset, location.size);
}
