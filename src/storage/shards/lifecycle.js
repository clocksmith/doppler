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
