export function buildSuiteSummary(suiteName, results, startTimeMs) {
  let passed = 0;
  let failed = 0;
  let skipped = 0;
  const safeResults = Array.isArray(results) ? results : [];
  for (const result of safeResults) {
    if (result.skipped) {
      skipped++;
    } else if (result.passed) {
      passed++;
    } else {
      failed++;
    }
  }
  const duration = Math.max(0, performance.now() - (Number.isFinite(startTimeMs) ? startTimeMs : performance.now()));
  return { suite: suiteName, passed, failed, skipped, duration, results: safeResults };
}

export function normalizeCacheMode(value) {
  return value === 'cold' || value === 'warm' ? value : 'warm';
}

export function normalizeLoadMode(value, hasModelUrl, modelUrl) {
  if (value === 'opfs' || value === 'http' || value === 'memory' || value === 'file') {
    return value;
  }
  if (!hasModelUrl) return 'opfs';
  if (typeof modelUrl === 'string' && modelUrl.startsWith('file://')) return 'file';
  return 'http';
}

export function normalizeWorkloadType(value) {
  const normalized = String(value || '').trim().toLowerCase();
  return normalized || null;
}

export function safeStatsValue(value) {
  return Number.isFinite(value) ? Number(value) : 0;
}
