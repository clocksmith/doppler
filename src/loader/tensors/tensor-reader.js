

import { trace } from '../../debug/index.js';
import {
  getSourceTransformSpec,
  hasSourceTransform,
  materializeTensorSourceTransform,
} from './source-transform.js';

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

async function assembleLocationBytes(location, name, label, loadShard, loadShardRange = null) {
  const chunks = getPhysicalChunks(location, name);
  const parts = await Promise.all(chunks.map(async (chunk) => {
    if (loadShardRange) {
      const data = await loadShardRange(chunk.shardIndex, chunk.offset, chunk.size);
      if (chunk.size > data.byteLength) {
        throw new Error(
          `[DopplerLoader] Shard ${chunk.shardIndex} too small for tensor "${name}" ${label}.`
        );
      }
      return new Uint8Array(data, 0, chunk.size);
    }
    const data = await loadShard(chunk.shardIndex);
    if (chunk.offset + chunk.size > data.byteLength) {
      throw new Error(
        `[DopplerLoader] Shard ${chunk.shardIndex} too small for tensor "${name}" ${label}.`
      );
    }
    return new Uint8Array(data, chunk.offset, chunk.size);
  }));
  const totalSize = parts.reduce((sum, part) => sum + part.byteLength, 0);
  const combined = new Uint8Array(totalSize);
  let offset = 0;
  for (const part of parts) {
    combined.set(part, offset);
    offset += part.byteLength;
  }
  return combined;
}

async function loadLocationRange(location, name, label, byteOffset, byteLength, loadShardRange) {
  const chunks = getPhysicalChunks(location, name);
  const totalSize = chunks.reduce((sum, chunk) => sum + chunk.size, 0);
  if (byteOffset + byteLength > totalSize) {
    throw new Error(
      `[DopplerLoader] Tensor "${name}" ${label} range (${byteOffset}..${byteOffset + byteLength}) exceeds size ${totalSize}.`
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
        `[DopplerLoader] Shard ${chunk.shardIndex} too small for tensor "${name}" ${label} range.`
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
      `[DopplerLoader] Tensor "${name}" short ${label} range read: got ${writeOffset}, expected ${byteLength}.`
    );
  }

  return combined;
}

function getSourceTransformScaleLocation(location, name) {
  const scaleLocation = location?.sourceTransform?.scaleSource;
  if (!scaleLocation || typeof scaleLocation !== 'object') {
    throw new Error(
      `[DopplerLoader] Tensor "${name}" sourceTransform is missing scaleSource.`
    );
  }
  return scaleLocation;
}

function getSourceTransformSumLocation(location) {
  const transform = location?.sourceTransform;
  const sumLocation = transform?.kind === 'litert_axis_dequant'
    ? transform?.sumSource
    : transform?.rowSumSource;
  if (!sumLocation || typeof sumLocation !== 'object') {
    return null;
  }
  return sumLocation;
}

async function materializeLocationBytes(rawBytes, location, name, loadShard, loadShardRange = null, options = {}) {
  if (!hasSourceTransform(location)) {
    return rawBytes;
  }
  if (
    location.sourceTransform.kind === 'litert_rowwise_dequant'
    || location.sourceTransform.kind === 'litert_axis_dequant'
  ) {
    const scaleLocation = getSourceTransformScaleLocation(location, name);
    const scaleBytes = options.scaleBytes instanceof Uint8Array
      ? options.scaleBytes
      : await assembleLocationBytes(scaleLocation, name, 'scale companion', loadShard, loadShardRange);
    const sumLocation = getSourceTransformSumLocation(location);
    const sumBytes = options.sumBytes instanceof Uint8Array
      ? options.sumBytes
      : options.rowSumBytes instanceof Uint8Array
      ? options.rowSumBytes
      : (
        sumLocation
          ? await assembleLocationBytes(sumLocation, name, 'sum companion', loadShard, loadShardRange)
          : null
      );
    return materializeTensorSourceTransform(rawBytes, location, name, {
      scaleBytes,
      sumBytes,
      rowSumBytes: sumBytes,
      rowStart: options.rowStart ?? null,
      rowCount: options.rowCount ?? null,
      storageColumnStart: options.storageColumnStart ?? null,
    });
  }
  return materializeTensorSourceTransform(rawBytes, location, name);
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
    return materializeLocationBytes(combined, location, name, loadShard, loadShardRange);
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
    const bytes = new Uint8Array(slice, 0, size);
    return materializeLocationBytes(bytes, location, name, loadShard, loadShardRange);
  }

  const fullShard = await loadShard(shardIndex);
  if (offset + size > fullShard.byteLength) {
    throw new Error(
      `[DopplerLoader] Shard ${shardIndex} too small for tensor "${name}" (offset=${offset}, size=${size}, shard=${fullShard.byteLength})`
    );
  }
  const bytes = new Uint8Array(fullShard, offset, size);
  return materializeLocationBytes(bytes, location, name, loadShard, loadShardRange);
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

  if (hasSourceTransform(location)) {
    const spec = getSourceTransformSpec(location, name);
    if (
      (spec?.kind === 'litert_rowwise_dequant' || spec?.kind === 'litert_axis_dequant')
      && (byteOffset % spec.targetRowBytes) === 0
      && (byteLength % spec.targetRowBytes) === 0
    ) {
      const rowStart = byteOffset / spec.targetRowBytes;
      const rowCount = byteLength / spec.targetRowBytes;
      const scaleBytes = await loadLocationRange(
        getSourceTransformScaleLocation(location, name),
        name,
        'LiteRT transformed scale',
        rowStart * spec.scaleRowBytes,
        rowCount * spec.scaleRowBytes,
        loadShardRange
      );
      const sumLocation = getSourceTransformSumLocation(location);
      const sumBytes = sumLocation
        ? await loadLocationRange(
          sumLocation,
          name,
          'LiteRT transformed sum',
          rowStart * 4,
          rowCount * 4,
          loadShardRange
        )
        : null;
      if (spec.kind === 'litert_rowwise_dequant') {
        const raw = await loadLocationRange(
          location,
          name,
          'LiteRT transformed raw',
          rowStart * spec.rawRowBytes,
          rowCount * spec.rawRowBytes,
          loadShardRange
        );
        return materializeTensorSourceTransform(raw, location, name, {
          scaleBytes,
          rowSumBytes: sumBytes,
          rowStart,
          rowCount,
        });
      }

      if (spec.quantAxis === 1) {
        const raw = await loadLocationRange(
          location,
          name,
          'LiteRT transformed raw',
          rowStart * spec.rawStorageRowBytes,
          rowCount * spec.rawStorageRowBytes,
          loadShardRange
        );
        return materializeTensorSourceTransform(raw, location, name, {
          scaleBytes,
          sumBytes,
          rowStart,
          rowCount,
        });
      }

      const storageColumnStart = Math.floor(rowStart / spec.storageValuesPerByte) * spec.storageValuesPerByte;
      const storageColumnEnd = Math.ceil((rowStart + rowCount) / spec.storageValuesPerByte) * spec.storageValuesPerByte;
      const rawSliceRowBytes = (storageColumnEnd - storageColumnStart) / spec.storageValuesPerByte;
      const raw = new Uint8Array(spec.storageRows * rawSliceRowBytes);
      for (let storageRow = 0; storageRow < spec.storageRows; storageRow++) {
        const rowBytes = await loadLocationRange(
          location,
          name,
          `LiteRT transformed raw storage row ${storageRow}`,
          storageRow * spec.rawStorageRowBytes + (storageColumnStart / spec.storageValuesPerByte),
          rawSliceRowBytes,
          loadShardRange
        );
        raw.set(rowBytes, storageRow * rawSliceRowBytes);
      }
      return materializeTensorSourceTransform(raw, location, name, {
        scaleBytes,
        sumBytes,
        rowStart,
        rowCount,
        storageColumnStart,
      });
    }
    const raw = await assembleLocationBytes(location, name, 'transformed tensor', null, loadShardRange);
    const transformed = await materializeLocationBytes(raw, location, name, null, loadShardRange);
    if (byteOffset + byteLength > transformed.byteLength) {
      throw new Error(
        `[DopplerLoader] Tensor "${name}" transformed range (${byteOffset}..${byteOffset + byteLength}) ` +
        `exceeds size ${transformed.byteLength}.`
      );
    }
    return transformed.slice(byteOffset, byteOffset + byteLength);
  }

  return loadLocationRange(location, name, 'tensor', byteOffset, byteLength, loadShardRange);
}
