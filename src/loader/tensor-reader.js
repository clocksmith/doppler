

import { trace } from '../debug/index.js';


export async function assembleShardData(location, name, loadShard, loadShardRange = null) {
  if (location.spans) {
    trace.loader(`Assembling tensor "${name}" from ${location.spans.length} spans`);
    
    const chunks = [];
    for (const span of location.spans) {
      if (loadShardRange) {
        const data = await loadShardRange(span.shardIndex, span.offset, span.size);
        if (span.size > data.byteLength) {
          throw new Error(
            `[DopplerLoader] Shard ${span.shardIndex} too small for tensor "${name}" span.`
          );
        }
        chunks.push(new Uint8Array(data, 0, span.size));
      } else {
        const data = await loadShard(span.shardIndex);
        if (span.offset + span.size > data.byteLength) {
          throw new Error(
            `[DopplerLoader] Shard ${span.shardIndex} too small for tensor "${name}" span.`
          );
        }
        chunks.push(new Uint8Array(data, span.offset, span.size));
      }
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
  if (loadShardRange) {
    const slice = await loadShardRange(location.shardIndex, location.offset, location.size);
    if (location.size > slice.byteLength) {
      throw new Error(
        `[DopplerLoader] Shard ${location.shardIndex} too small for tensor "${name}" (offset=${location.offset}, size=${location.size}, shard=${slice.byteLength})`
      );
    }
    return new Uint8Array(slice, 0, location.size);
  }

  const fullShard = await loadShard(location.shardIndex);
  if (location.offset + location.size > fullShard.byteLength) {
    throw new Error(
      `[DopplerLoader] Shard ${location.shardIndex} too small for tensor "${name}" (offset=${location.offset}, size=${location.size}, shard=${fullShard.byteLength})`
    );
  }
  return new Uint8Array(fullShard, location.offset, location.size);
}
