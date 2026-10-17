import { computeSampleStats } from '../../debug/stats.js';

/** @param {number} start @param {number} durationMs @param {number[]} tokenTimestamps */
export function buildRequestTiming(start, durationMs, tokenTimestamps) {
  const firstTokenMs = tokenTimestamps.length > 0 ? tokenTimestamps[0] - start : null;
  const decodeMs = firstTokenMs === null ? null : durationMs - firstTokenMs;
  const decodeTokens = Math.max(0, tokenTimestamps.length - 1);
  return {
    scope: /** @type {const} */ ('generation-call-through-completion'),
    firstTokenMs, totalRunMs: durationMs, decodeMs, decodeTokens,
    tokensGenerated: tokenTimestamps.length,
    decodeTokensPerSec: decodeMs !== null && decodeMs > 0 ? (decodeTokens / decodeMs) * 1000 : 0,
    perTokenMs: tokenTimestamps.slice(1).map((time, index) => time - tokenTimestamps[index]),
  };
}

/**
 * @param {import('./request-timing.js').RequestTimingSample[]} samples
 * @param {number[]} prefillTokens
 * @param {number} timedRuns
 * @param {number} modelLoadMs
 * @param {{ outlierPolicy: 'none' | 'iqr', outlierIqrMultiplier?: number }} samplePolicy
 */
export function summarizeRequestTiming(samples, prefillTokens, timedRuns, modelLoadMs, samplePolicy) {
  /** @type {Record<string, import('../../debug/stats.js').SampleStats>} */
  const latency = {};
  for (const key of /** @type {const} */ (['firstTokenMs', 'totalRunMs', 'decodeMs', 'decodeTokensPerSec'])) {
    latency[key] = computeSampleStats(samples.map((sample) => sample[key] ?? NaN), samplePolicy);
  }
  const perTokenMs = computeSampleStats(samples.flatMap((sample) => sample.perTokenMs), samplePolicy);
  const prefill = computeSampleStats(samples.map((sample, index) => (
    sample.firstTokenMs !== null && sample.firstTokenMs > 0 ? (prefillTokens[index] / sample.firstTokenMs) * 1000 : 0
  )), samplePolicy);
  const complete = samples.length === timedRuns && samples.every((sample) => Number.isFinite(sample.firstTokenMs));
  return {
    metrics: { scope: 'generation-call-through-completion', samples, latency, perTokenMs },
    scope: complete ? 'generation-call-through-completion' : 'internal-pipeline',
    timing: complete ? {
      firstTokenMs: latency.firstTokenMs.median,
      firstResponseMs: modelLoadMs + latency.firstTokenMs.median,
      totalRunMs: latency.totalRunMs.median,
      decodeMs: latency.decodeMs.median,
      decodeTokensPerSec: latency.decodeTokensPerSec.median,
      decodeMsPerTokenP50: perTokenMs.median,
      decodeMsPerTokenP95: perTokenMs.p95,
      decodeMsPerTokenP99: perTokenMs.p99,
      prefillTokensPerSecTtft: prefill.median,
    } : {},
  };
}
