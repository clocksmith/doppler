

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

function getPhysicalChunks(location, name) {
  const spans = getLocationSpans(location);
  if (spans) {
    return spans.map((span, spanIndex) => ({
      shardIndex: resolveSpanShardIndex(span, name, spanIndex),
      offset: validateSpanField(span.offset, 'offset', name, spanIndex),
      size: validateSpanField(span.size, 'size', name, spanIndex),
    }));
  }
  return [{
    shardIndex: resolveLocationShardIndex(location, name),
    offset: validateLocationField(location, 'offset', name),
    size: validateLocationField(location, 'size', name),
  }];
}

export async function assembleShardData(location, name, loadShard, loadShardRange = null) {
  const spans = getLocationSpans(location);
  if (spans) {
    trace.loader(`Assembling tensor "${name}" from ${spans.length} spans`);

    const chunks = await Promise.all(getPhysicalChunks(location, name).map(async (chunk) => {
      if (loadShardRange) {
        const data = await loadShardRange(chunk.shardIndex, chunk.offset, chunk.size);
        if (chunk.size > data.byteLength) {
          throw new Error(
            `[DopplerLoader] Shard ${chunk.shardIndex} too small for tensor "${name}" span.`
          );
        }
        return new Uint8Array(data, 0, chunk.size);
      }
      const data = await loadShard(chunk.shardIndex);
      if (chunk.offset + chunk.size > data.byteLength) {
        throw new Error(
          `[DopplerLoader] Shard ${chunk.shardIndex} too small for tensor "${name}" span.`
        );
      }
      return new Uint8Array(data, chunk.offset, chunk.size);
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

export async function loadTensorRange(location, name, byteOffset, byteLength, loadShardRange) {
  if (typeof loadShardRange !== 'function') {
    throw new Error(`[DopplerLoader] Tensor "${name}" range loading requires loadShardRange().`);
  }
  if (!Number.isInteger(byteOffset) || byteOffset < 0) {
    throw new Error(`[DopplerLoader] Tensor "${name}" has invalid byteOffset ${byteOffset}.`);
  }
  if (!Number.isInteger(byteLength) || byteLength < 0) {
    throw new Error(`[DopplerLoader] Tensor "${name}" has invalid byteLength ${byteLength}.`);
  }
  if (byteLength === 0) {
    return new Uint8Array(0);
  }

  const chunks = getPhysicalChunks(location, name);
  const totalSize = chunks.reduce((sum, chunk) => sum + chunk.size, 0);
  if (byteOffset + byteLength > totalSize) {
    throw new Error(
      `[DopplerLoader] Tensor "${name}" range (${byteOffset}..${byteOffset + byteLength}) exceeds size ${totalSize}.`
    );
  }

  const combined = new Uint8Array(byteLength);
  let logicalOffset = 0;
  let writeOffset = 0;
  const rangeEnd = byteOffset + byteLength;

  for (const chunk of chunks) {
    const chunkStart = logicalOffset;
    const chunkEnd = chunkStart + chunk.size;
    logicalOffset = chunkEnd;

    if (rangeEnd <= chunkStart || byteOffset >= chunkEnd) {
      continue;
    }

    const start = Math.max(byteOffset, chunkStart);
    const end = Math.min(rangeEnd, chunkEnd);
    const localOffset = chunk.offset + (start - chunkStart);
    const localSize = end - start;
    const data = await loadShardRange(chunk.shardIndex, localOffset, localSize);
    if (localSize > data.byteLength) {
      throw new Error(
        `[DopplerLoader] Shard ${chunk.shardIndex} too small for tensor "${name}" range.`
      );
    }
    combined.set(new Uint8Array(data, 0, localSize), writeOffset);
    writeOffset += localSize;

    if (writeOffset === byteLength) {
      break;
    }
  }

  if (writeOffset !== byteLength) {
    throw new Error(
      `[DopplerLoader] Tensor "${name}" short range read: got ${writeOffset}, expected ${byteLength}.`
    );
  }

  return combined;
}
