function resolveTensorPrimarySpans(location) {
  if (Array.isArray(location?.spans) && location.spans.length > 0) {
    return location.spans.map((span) => ({
      shardIndex: span.shardIndex ?? span.shard,
      offset: span.offset,
      size: span.size,
    }));
  }
  return [{
    shardIndex: location?.shardIndex ?? location?.shard,
    offset: location?.offset,
    size: location?.size,
  }];
}

export function isRequestedRangeInsideTensor(location, shardIndex, offset, length) {
  const start = Math.max(
    0,
    Number.isFinite(Number(offset)) ? Math.floor(Number(offset)) : 0
  );
  const size = length == null
    ? null
    : Math.max(0, Number.isFinite(Number(length)) ? Math.floor(Number(length)) : 0);
  const end = size == null ? Number.POSITIVE_INFINITY : start + size;
  const spans = resolveTensorPrimarySpans(location);
  return spans.some((span) => {
    const spanShardIndex = span.shardIndex;
    const spanStart = span.offset;
    const spanEnd = span.offset + span.size;
    return spanShardIndex === shardIndex && start >= spanStart && end <= spanEnd;
  });
}

export async function checkFileExistsInBackend(storageBackend, filename) {
  return (await getFileSizeInBackend(storageBackend, filename)) !== null;
}

export async function getFileSizeInBackend(storageBackend, filename) {
  if (!storageBackend || typeof storageBackend !== 'object') {
    throw new Error('getFileSizeInBackend requires a storage backend object.');
  }
  if (!filename || typeof filename !== 'string') {
    throw new Error('getFileSizeInBackend requires a filename.');
  }

  try {
    if (typeof storageBackend.getFileSize === 'function') {
      const size = await storageBackend.getFileSize(filename);
      return Number.isFinite(size) ? Math.max(0, Math.floor(size)) : null;
    }
    const buffer = await storageBackend.readFile(filename);
    return buffer.byteLength;
  } catch (error) {
    const message = String(error?.message || '');
    if (error?.name === 'NotFoundError' || message.toLowerCase().includes('not found')) {
      return null;
    }
    throw error;
  }
}
