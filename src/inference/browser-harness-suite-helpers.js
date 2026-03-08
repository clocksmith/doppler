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

export function normalizeLoadMode(value, hasModelUrl) {
  if (value === 'opfs' || value === 'http' || value === 'memory') {
    return value;
  }
  return hasModelUrl ? 'http' : 'opfs';
}

export function normalizeWorkloadType(value) {
  const normalized = String(value || '').trim().toLowerCase();
  return normalized || null;
}

export function safeStatsValue(value) {
  return Number.isFinite(value) ? Number(value) : 0;
}

export function calculateRatePerSecond(count, durationMs) {
  const safeCount = safeStatsValue(count);
  const safeDurationMs = safeStatsValue(durationMs);
  if (safeCount <= 0 || safeDurationMs <= 0) return 0;
  return Number(((safeCount * 1000) / safeDurationMs).toFixed(2));
}

export function buildDiffusionPerformanceArtifact({
  warmupRuns,
  timedRuns,
  width,
  height,
  steps,
  guidanceScale,
  avgPrefillTokens,
  avgDecodeTokens,
  cpuStats,
  gpuStats,
}) {
  const cpuPrefillMs = safeStatsValue(cpuStats?.prefillMs?.median);
  const cpuDenoiseMs = safeStatsValue(cpuStats?.denoiseMs?.median);
  const cpuVaeMs = safeStatsValue(cpuStats?.vaeMs?.median);
  const cpuTotalMs = safeStatsValue(cpuStats?.totalMs?.median);
  const gpuPrefillMs = safeStatsValue(gpuStats?.prefillMs?.median);
  const gpuDenoiseMs = safeStatsValue(gpuStats?.denoiseMs?.median);
  const gpuVaeMs = safeStatsValue(gpuStats?.vaeMs?.median);
  const gpuTotalMs = safeStatsValue(gpuStats?.totalMs?.median);
  const decodeStepsPerSec = calculateRatePerSecond(steps, cpuDenoiseMs);
  const decodeTokensPerSec = calculateRatePerSecond(avgDecodeTokens, cpuDenoiseMs);
  const prefillTokensPerSec = calculateRatePerSecond(avgPrefillTokens, cpuPrefillMs);

  return {
    schemaVersion: 1,
    warmupRuns,
    timedRuns,
    shape: {
      width,
      height,
    },
    scheduler: {
      steps,
      guidanceScale,
    },
    cpu: {
      totalMs: cpuTotalMs,
      prefillMs: cpuPrefillMs,
      denoiseMs: cpuDenoiseMs,
      vaeMs: cpuVaeMs,
    },
    gpu: {
      available: gpuStats?.available === true,
      totalMs: gpuStats?.available === true ? gpuTotalMs : null,
      prefillMs: gpuStats?.available === true ? gpuPrefillMs : null,
      denoiseMs: gpuStats?.available === true ? gpuDenoiseMs : null,
      vaeMs: gpuStats?.available === true ? gpuVaeMs : null,
    },
    throughput: {
      prefillTokensPerSec,
      decodeTokensPerSec,
      decodeStepsPerSec,
    },
    tokens: {
      avgPrefillTokens: safeStatsValue(avgPrefillTokens),
      avgDecodeTokens: safeStatsValue(avgDecodeTokens),
    },
  };
}

export function assertDiffusionPerformanceArtifact(metrics, contextLabel = 'diffusion') {
  const artifact = metrics?.performanceArtifact;
  if (!artifact || typeof artifact !== 'object') {
    throw new Error(`${contextLabel}: metrics.performanceArtifact is required.`);
  }
  if (artifact.schemaVersion !== 1) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.schemaVersion must be 1.`);
  }
  if (!Number.isInteger(artifact.warmupRuns) || artifact.warmupRuns < 0) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.warmupRuns must be a non-negative integer.`);
  }
  if (!Number.isInteger(artifact.timedRuns) || artifact.timedRuns < 1) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.timedRuns must be a positive integer.`);
  }
  if (!Number.isFinite(artifact?.cpu?.prefillMs)) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.cpu.prefillMs must be finite.`);
  }
  if (!Number.isFinite(artifact?.cpu?.denoiseMs)) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.cpu.denoiseMs must be finite.`);
  }
  if (!Number.isFinite(artifact?.cpu?.vaeMs)) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.cpu.vaeMs must be finite.`);
  }
  if (!Number.isFinite(artifact?.cpu?.totalMs)) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.cpu.totalMs must be finite.`);
  }
  if (!Number.isFinite(artifact?.throughput?.decodeStepsPerSec)) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.throughput.decodeStepsPerSec must be finite.`);
  }
}

function formatMetricNumber(value, fallback = 0, digits = 2) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return fallback;
  return Number(numericValue.toFixed(digits));
}

export function toTimingNumber(value, fallback = 0) {
  return formatMetricNumber(value, fallback, 2);
}

export function safeToFixed(value, fallback = 0, digits = 2) {
  return formatMetricNumber(value, fallback, digits);
}

export function sampleTimingNumber(stats, key, fallback = 0) {
  return formatMetricNumber(stats?.[key], fallback, 2);
}

export function buildCanonicalTiming(overrides = {}) {
  const cacheMode = normalizeCacheMode(overrides.cacheMode);
  const modelLoadMs = toTimingNumber(overrides.modelLoadMs, 0);
  const prefillMs = toTimingNumber(overrides.prefillMs, 0);
  const decodeMs = toTimingNumber(overrides.decodeMs, 0);
  const decodeMsPerTokenP50 = Number.isFinite(overrides.decodeMsPerTokenP50)
    ? toTimingNumber(overrides.decodeMsPerTokenP50)
    : null;
  const decodeMsPerTokenP95 = Number.isFinite(overrides.decodeMsPerTokenP95)
    ? toTimingNumber(overrides.decodeMsPerTokenP95)
    : null;
  const decodeMsPerTokenP99 = Number.isFinite(overrides.decodeMsPerTokenP99)
    ? toTimingNumber(overrides.decodeMsPerTokenP99)
    : null;
  const decodeTokensPerSec = Number.isFinite(overrides.decodeTokensPerSec)
    ? toTimingNumber(overrides.decodeTokensPerSec)
    : null;
  const prefillTokensPerSec = Number.isFinite(overrides.prefillTokensPerSec)
    ? toTimingNumber(overrides.prefillTokensPerSec)
    : null;
  const totalRunMs = toTimingNumber(
    overrides.totalRunMs,
    toTimingNumber(prefillMs + decodeMs)
  );
  const firstTokenMs = Number.isFinite(overrides.firstTokenMs)
    ? toTimingNumber(overrides.firstTokenMs)
    : null;
  const firstResponseMs = Number.isFinite(overrides.firstResponseMs)
    ? toTimingNumber(overrides.firstResponseMs)
    : toTimingNumber(modelLoadMs + totalRunMs);

  return {
    modelLoadMs,
    firstTokenMs,
    firstResponseMs,
    prefillMs,
    decodeMs,
    decodeMsPerTokenP50,
    decodeMsPerTokenP95,
    decodeMsPerTokenP99,
    decodeTokensPerSec,
    prefillTokensPerSec,
    totalRunMs,
    cacheMode,
    loadMode: overrides.loadMode,
  };
}

export function buildTimingDiagnostics(timing = {}, options = {}) {
  const prefillSemantics = String(options.prefillSemantics || 'internal_prefill_phase');
  const source = String(options.source || 'doppler');
  const modelLoadMs = Number.isFinite(timing.modelLoadMs) ? toTimingNumber(timing.modelLoadMs) : null;
  const firstTokenMs = Number.isFinite(timing.firstTokenMs) ? toTimingNumber(timing.firstTokenMs) : null;
  const firstResponseMs = Number.isFinite(timing.firstResponseMs) ? toTimingNumber(timing.firstResponseMs) : null;
  const prefillMs = Number.isFinite(timing.prefillMs) ? toTimingNumber(timing.prefillMs) : null;
  const decodeMs = Number.isFinite(timing.decodeMs) ? toTimingNumber(timing.decodeMs) : null;
  const totalRunMs = Number.isFinite(timing.totalRunMs) ? toTimingNumber(timing.totalRunMs) : null;

  const firstResponseFromLoadAndFirstTokenMs = (
    Number.isFinite(modelLoadMs) && Number.isFinite(firstTokenMs)
  )
    ? toTimingNumber(modelLoadMs + firstTokenMs)
    : null;
  const runFromPrefillAndDecodeMs = (
    Number.isFinite(prefillMs) && Number.isFinite(decodeMs)
  )
    ? toTimingNumber(prefillMs + decodeMs)
    : null;

  const firstResponseResidualMs = (
    Number.isFinite(firstResponseMs) && Number.isFinite(firstResponseFromLoadAndFirstTokenMs)
  )
    ? toTimingNumber(firstResponseMs - firstResponseFromLoadAndFirstTokenMs)
    : null;
  const runResidualMs = (
    Number.isFinite(totalRunMs) && Number.isFinite(runFromPrefillAndDecodeMs)
  )
    ? toTimingNumber(totalRunMs - runFromPrefillAndDecodeMs)
    : null;

  return {
    schemaVersion: 1,
    source,
    semantics: {
      modelLoadMs: 'model initialization/load before generation',
      firstTokenMs: 'ttft from generation start',
      firstResponseMs: 'modelLoadMs + firstTokenMs',
      prefillMs: prefillSemantics,
      decodeMs: 'time after first token',
      totalRunMs: 'prefillMs + decodeMs',
    },
    componentsMs: {
      modelLoadMs,
      firstTokenMs,
      firstResponseMs,
      prefillMs,
      decodeMs,
      totalRunMs,
    },
    sumsMs: {
      firstResponseFromLoadAndFirstTokenMs,
      runFromPrefillAndDecodeMs,
    },
    residualsMs: {
      firstResponseResidualMs,
      runResidualMs,
    },
    consistent: {
      firstResponse: Number.isFinite(firstResponseResidualMs) ? Math.abs(firstResponseResidualMs) <= 2 : null,
      totalRun: Number.isFinite(runResidualMs) ? Math.abs(runResidualMs) <= 2 : null,
    },
  };
}
