export function normalizeShardWriterOptions(options = {}) {
  const append = options?.append === true;
  const expectedOffsetRaw = options?.expectedOffset;
  const expectedOffset = expectedOffsetRaw == null
    ? null
    : Number(expectedOffsetRaw);
  if (
    expectedOffset != null
    && (!Number.isInteger(expectedOffset) || expectedOffset < 0)
  ) {
    throw new Error('Shard writer expectedOffset must be a non-negative integer');
  }
  return { append, expectedOffset };
}

export function createStorageWriteStream(storageBackend, filename, options = {}, onCreate = null) {
  if (!storageBackend?.createWriteStream) {
    throw new Error('Storage backend does not support streaming writes');
  }
  onCreate?.();
  return storageBackend.createWriteStream(filename, normalizeShardWriterOptions(options));
}
