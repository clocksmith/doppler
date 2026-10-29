import { safeStatsValue } from './suite-summary.js';

function calculateRatePerSecond(count, durationMs) {
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
  modality = 'image',
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
    modality,
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
