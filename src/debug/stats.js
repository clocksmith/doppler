export function percentile(sorted, p) {
  if (!sorted.length) return 0;
  if (sorted.length === 1) return sorted[0];
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

export function median(sorted) {
  if (!sorted.length) return 0;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

export function removeOutliersIQR(values, multiplier = 1.5) {
  if (values.length < 4) return { filtered: values, removed: 0, lower: -Infinity, upper: Infinity };
  const sorted = [...values].sort((a, b) => a - b);
  const q1 = percentile(sorted, 25);
  const q3 = percentile(sorted, 75);
  const iqr = q3 - q1;
  const lower = q1 - multiplier * iqr;
  const upper = q3 + multiplier * iqr;
  const filtered = values.filter((v) => v >= lower && v <= upper);
  return { filtered, removed: values.length - filtered.length, lower, upper };
}

export function sampleStdDev(values, meanValue) {
  const n = values.length;
  if (n < 2) return 0;
  const variance = values.reduce((sum, v) => sum + (v - meanValue) ** 2, 0) / (n - 1);
  return Math.sqrt(variance);
}

export function confidenceInterval95(stdDev, n) {
  if (n < 2) return 0;
  const tValue = n >= 30 ? 1.96 : 2.0 + 3.0 / n;
  return tValue * (stdDev / Math.sqrt(n));
}

export function computeSampleStats(values, options = {}) {
  const { outlierIqrMultiplier = 1.5 } = options;
  const { filtered, removed } = removeOutliersIQR(values, outlierIqrMultiplier);
  const sorted = [...filtered].sort((a, b) => a - b);
  const n = sorted.length;

  if (n === 0) {
    return {
      mean: 0,
      median: 0,
      min: 0,
      max: 0,
      p95: 0,
      p99: 0,
      stdDev: 0,
      ci95: 0,
      samples: values.length,
      samplesAfterOutlierRemoval: 0,
      outliersRemoved: removed,
    };
  }

  const meanValue = filtered.reduce((a, b) => a + b, 0) / n;
  const stdDev = sampleStdDev(filtered, meanValue);

  return {
    mean: meanValue,
    median: median(sorted),
    min: sorted[0],
    max: sorted[n - 1],
    p95: percentile(sorted, 95),
    p99: percentile(sorted, 99),
    stdDev,
    ci95: confidenceInterval95(stdDev, n),
    samples: values.length,
    samplesAfterOutlierRemoval: n,
    outliersRemoved: removed,
  };
}

export function computeArrayStats(values, limit = values.length) {
  const stats = {
    min: Infinity,
    max: -Infinity,
    mean: 0,
    std: 0,
    maxAbs: 0,
    nanCount: 0,
    infCount: 0,
    zeroCount: 0,
    validCount: 0,
  };

  let sum = 0;
  let sumSq = 0;
  const count = Math.min(values.length, limit);

  for (let i = 0; i < count; i++) {
    const v = values[i];
    if (Number.isNaN(v)) {
      stats.nanCount++;
      continue;
    }
    if (!Number.isFinite(v)) {
      stats.infCount++;
      continue;
    }
    if (v === 0) stats.zeroCount++;
    stats.min = Math.min(stats.min, v);
    stats.max = Math.max(stats.max, v);
    sum += v;
    sumSq += v * v;
    stats.validCount++;
  }

  if (stats.validCount > 0) {
    stats.mean = sum / stats.validCount;
    const variance = sumSq / stats.validCount - stats.mean * stats.mean;
    stats.std = Math.sqrt(Math.max(0, variance));
    stats.maxAbs = Math.max(Math.abs(stats.min), Math.abs(stats.max));
  } else {
    stats.min = 0;
    stats.max = 0;
    stats.maxAbs = 0;
  }

  return stats;
}

export function computeBasicStats(values) {
  const count = values.length;
  if (!count) {
    return { mean: 0, min: 0, max: 0, total: 0, count: 0 };
  }

  let min = Infinity;
  let max = -Infinity;
  let total = 0;

  for (const v of values) {
    min = Math.min(min, v);
    max = Math.max(max, v);
    total += v;
  }

  return {
    mean: total / count,
    min,
    max,
    total,
    count,
  };
}
