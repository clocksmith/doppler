export function getTensorShardIndices(location) {
  if (Array.isArray(location?.spans) && location.spans.length > 0) {
    return Array.from(
      new Set(location.spans.map((span) => span?.shardIndex).filter(Number.isInteger))
    );
  }
  return Number.isInteger(location?.shardIndex) ? [location.shardIndex] : [];
}
