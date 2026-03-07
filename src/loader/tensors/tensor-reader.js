

import { trace } from '../../debug/index.js';

function resolveSpanShardIndex(span, name, spanIndex) {
  const shardIndex = typeof span?.shardIndex === 'number'
    ? span.shardIndex
    : span?.shard;
  if (!Number.isInteger(shardIndex) || shardIndex < 0) {
    throw new Error(
      `[DopplerLoader] Tensor "${name}" span[${spanIndex}] has invalid shard index.`
    );
  }
  return shardIndex;
}

function validateSpanField(value, field, name, spanIndex) {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(
      `[DopplerLoader] Tensor "${name}" span[${spanIndex}] has invalid ${field}.`
    );
  }
  return value;
}

function getLocationSpans(location) {
  if (!Array.isArray(location?.spans) || location.spans.length === 0) {
    return null;
  }
  return location.spans;
}

function resolveLocationShardIndex(location, name) {
  const shardIndex = typeof location?.shardIndex === 'number'
    ? location.shardIndex
    : location?.shard;
  if (!Number.isInteger(shardIndex) || shardIndex < 0) {
    throw new Error(`[DopplerLoader] Tensor "${name}" has invalid shard index.`);
  }
  return shardIndex;
}

function validateLocationField(location, field, name) {
  const value = location?.[field];
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`[DopplerLoader] Tensor "${name}" has invalid ${field}.`);
  }
  return value;
}

export async function assembleShardData(location, name, loadShard, loadShardRange = null) {
  const spans = getLocationSpans(location);
  if (spans) {
    trace.loader(`Assembling tensor "${name}" from ${spans.length} spans`);

    const chunks = await Promise.all(spans.map(async (span, spanIndex) => {
      const shardIndex = resolveSpanShardIndex(span, name, spanIndex);
      const offset = validateSpanField(span.offset, 'offset', name, spanIndex);
      const size = validateSpanField(span.size, 'size', name, spanIndex);
      if (loadShardRange) {
        const data = await loadShardRange(shardIndex, offset, size);
        if (size > data.byteLength) {
          throw new Error(
            `[DopplerLoader] Shard ${shardIndex} too small for tensor "${name}" span.`
          );
        }
        return new Uint8Array(data, 0, size);
      }
      const data = await loadShard(shardIndex);
      if (offset + size > data.byteLength) {
        throw new Error(
          `[DopplerLoader] Shard ${shardIndex} too small for tensor "${name}" span.`
        );
      }
      return new Uint8Array(data, offset, size);
    }));
    const totalSize = chunks.reduce((s, c) => s + c.length, 0);
    if (Number.isInteger(location?.size) && totalSize !== location.size) {
      throw new Error(
        `[DopplerLoader] Tensor "${name}" spans total ${totalSize} bytes, expected ${location.size}.`
      );
    }
    const combined = new Uint8Array(totalSize);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }
    return combined;
  }

  // Single shard - use view to avoid copying
  const shardIndex = resolveLocationShardIndex(location, name);
  const offset = validateLocationField(location, 'offset', name);
  const size = validateLocationField(location, 'size', name);
  if (loadShardRange) {
    const slice = await loadShardRange(shardIndex, offset, size);
    if (size > slice.byteLength) {
      throw new Error(
        `[DopplerLoader] Shard ${shardIndex} too small for tensor "${name}" (offset=${offset}, size=${size}, shard=${slice.byteLength})`
      );
    }
    return new Uint8Array(slice, 0, size);
  }

  const fullShard = await loadShard(shardIndex);
  if (offset + size > fullShard.byteLength) {
    throw new Error(
      `[DopplerLoader] Shard ${shardIndex} too small for tensor "${name}" (offset=${offset}, size=${size}, shard=${fullShard.byteLength})`
    );
  }
  return new Uint8Array(fullShard, offset, size);
}
