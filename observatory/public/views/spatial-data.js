// A coordinate projection, not a semantic embedding or dimensionality-reduction
// fit. Axes are the three original channels with greatest across-layer variance.
// Missing layers stay null; no segment may cross a missing layer.
export function coordinateJourney(vectors) {
  const available = vectors.filter((v) => v && v.length && Array.from(v).every(Number.isFinite));
  if (!available.length) return null;
  const width = available[0].length;
  if (available.some((v) => v.length !== width)) return null;
  const means = Array(width).fill(0);
  for (const v of available) for (let i = 0; i < width; i++) means[i] += v[i] / available.length;
  const variance = Array(width).fill(0);
  for (const v of available) for (let i = 0; i < width; i++) variance[i] += (v[i] - means[i]) ** 2;
  const axes = variance.map((value, channel) => ({ value, channel })).sort((a, b) => b.value - a.value || a.channel - b.channel).slice(0, 3).map((v) => v.channel);
  const points = vectors.map((v, layer) => !v || v.length !== width || !Array.from(v).every(Number.isFinite) ? null : { layer, values: axes.map((axis) => v[axis] - means[axis]) });
  const extent = Math.max(0, ...points.filter(Boolean).flatMap((p) => p.values.map(Math.abs)));
  const total = variance.reduce((a, b) => a + b, 0);
  return { axes, means: axes.map((i) => means[i]), points, extent, varianceShare: total > 0 ? axes.reduce((sum, i) => sum + variance[i], 0) / total : null };
}

export function rotatePoint(values, angle, spatial = true) {
  const [x = 0, y = 0, z = 0] = values;
  return spatial ? [x * Math.cos(angle) + z * Math.sin(angle), y * 0.85 - (z * Math.cos(angle) - x * Math.sin(angle)) * 0.35] : [x, y];
}

// Each cell is a recorded next-token probability, not embedding similarity.
// Unlisted candidates remain null, not zero. Bounded rows keep large runs usable.
export function candidateAtlas(run, limit = 24, { center = 0, count = 64 } = {}) {
  const windowSize = Math.max(1, Math.min(64, Number.isInteger(count) ? count : 64));
  const focus = Number.isInteger(center) ? Math.max(0, Math.min(run.tokens.length - 1, center)) : 0;
  const start = Math.max(0, Math.min(Math.max(0, run.tokens.length - windowSize), focus - Math.floor(windowSize / 2)));
  const tokens = run.tokens.slice(start, start + windowSize);
  const rows = new Map();
  tokens.forEach((token, index) => {
    const seen = new Set();
    for (const candidate of token.candidates ?? []) {
      if (!Number.isInteger(candidate.tokenId) || seen.has(candidate.tokenId) || !Number.isFinite(candidate.probability) || candidate.probability < 0 || candidate.probability > 1) continue;
      seen.add(candidate.tokenId);
      if (!rows.has(candidate.tokenId)) rows.set(candidate.tokenId, { tokenId: candidate.tokenId, text: candidate.text, cells: new Map(), peak: 0 });
      const row = rows.get(candidate.tokenId);
      row.cells.set(index, candidate.probability);
      row.peak = Math.max(row.peak, candidate.probability);
    }
  });
  const all = [...rows.values()].sort((a, b) => b.peak - a.peak || a.tokenId - b.tokenId);
  const shown = all.slice(0, Math.max(0, Math.min(24, limit))).map(({ cells, ...row }) => ({ ...row, probabilities: tokens.map((_, index) => cells.get(index) ?? null) }));
  return { rows: shown, omitted: all.length - shown.length, total: all.length, start, count: tokens.length, omittedColumns: run.tokens.length - tokens.length };
}
