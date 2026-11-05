/** @param {import('./index.js').TensorLocation} location */
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

/** @type {typeof import('./index.js').isRequestedRangeInsideTensor} */
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

/** @type {typeof import('./index.js').checkFileExistsInBackend} */
export async function checkFileExistsInBackend(storageBackend, filename) {
  return (await getFileSizeInBackend(storageBackend, filename)) !== null;
}

/** @type {typeof import('./index.js').getFileSizeInBackend} */
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

/** @type {typeof import('./index.js').readFileRangeInBackend} */
export async function readFileRangeInBackend(storageBackend, filename, offset, length, requireRange) {
  const start = Math.max(0, offset);
  const want = length == null ? null : Math.max(0, length);
  if (typeof storageBackend.readFileRange === 'function') return storageBackend.readFileRange(filename, start, want);
  if (requireRange) throw new Error('A bounded backend range reader is required; whole-file fallback is disabled.');
  const buffer = await storageBackend.readFile(filename);
  const view = new Uint8Array(buffer);
  const end = want == null ? view.length : Math.min(view.length, start + want);
  return view.slice(start, end).buffer;
}
