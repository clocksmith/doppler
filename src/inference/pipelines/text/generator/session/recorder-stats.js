/** @type {import('./recorder-stats.js').mergeRecordOpLabelCounts} */
export function mergeRecordOpLabelCounts(target, source) {
  const merged = target && typeof target === 'object' && !Array.isArray(target)
    ? target
    : {};
  if (!source || typeof source !== 'object' || Array.isArray(source)) {
    return merged;
  }
  for (const [label, rawCount] of Object.entries(source)) {
    const count = Number(rawCount);
    if (typeof label !== 'string' || label.length === 0 || !Number.isFinite(count) || count <= 0) {
      continue;
    }
    merged[label] = (Number.isFinite(merged[label]) ? merged[label] : 0) + count;
  }
  return merged;
}
