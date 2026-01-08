/**
 * Submit Tracker - Measures GPU submit overhead for optimization benchmarking.
 *
 * Usage:
 *   // Before forward pass:
 *   resetSubmitStats();
 *
 *   // Run forward pass...
 *
 *   // After forward pass:
 *   const stats = getSubmitStats();
 *   log.info('SubmitTracker', `Submits: ${stats.count}, Total time: ${stats.totalMs.toFixed(2)}ms`);
 *
 * To enable tracking, set TRACK_SUBMITS = true and wrap queue.submit calls.
 *
 * @module gpu/submit-tracker
 */

import { trackSubmit } from './perf-guards.js';
import { log, trace } from '../debug/index.js';

/** Whether to track submits (disable in production for perf) */
export let TRACK_SUBMITS = false;

/** Internal tracking state */
let submitCount = 0;
/** @type {number[]} */
let submitTimes = [];
let totalSubmitMs = 0;
let maxSubmitMs = 0;
let minSubmitMs = Infinity;
/** @type {Map<string, number>} */
let submitSources = new Map();

/** @typedef {'prefill' | 'decode' | 'other'} SubmitPhase */

/** @type {SubmitPhase} Current phase for submit tracking */
let currentPhase = 'other';

/** @type {Record<SubmitPhase, { count: number; times: number[]; totalMs: number; maxMs: number; minMs: number; sources: Map<string, number> }>} */
const phaseStats = {
  prefill: { count: 0, times: [], totalMs: 0, maxMs: 0, minMs: Infinity, sources: new Map() },
  decode: { count: 0, times: [], totalMs: 0, maxMs: 0, minMs: Infinity, sources: new Map() },
  other: { count: 0, times: [], totalMs: 0, maxMs: 0, minMs: Infinity, sources: new Map() },
};

/**
 * Enable/disable submit tracking.
 * @param {boolean} enabled - Whether to track submits
 * @returns {void}
 */
export function setTrackSubmits(enabled) {
  TRACK_SUBMITS = enabled;
  if (enabled) {
    resetSubmitStats();
    log.debug('SubmitTracker', 'Enabled');
  } else {
    log.debug('SubmitTracker', 'Disabled');
  }
}

/**
 * Reset submit statistics.
 * Call before starting a new measurement.
 * @returns {void}
 */
export function resetSubmitStats() {
  submitCount = 0;
  submitTimes = [];
  totalSubmitMs = 0;
  maxSubmitMs = 0;
  minSubmitMs = Infinity;
  submitSources = new Map();
  currentPhase = 'other';

  // Reset phase stats
  for (const phase of /** @type {const} */ (['prefill', 'decode', 'other'])) {
    phaseStats[phase] = { count: 0, times: [], totalMs: 0, maxMs: 0, minMs: Infinity, sources: new Map() };
  }
}

/**
 * Set the current phase for submit tracking.
 * @param {SubmitPhase} phase - The phase to track ('prefill', 'decode', or 'other')
 * @returns {void}
 */
export function setSubmitPhase(phase) {
  currentPhase = phase;
}

/**
 * Record a submit call.
 * Call this from a wrapper around queue.submit().
 * @param {number} durationMs - Time spent in this submit call
 * @param {string} [source] - Optional source identifier (e.g., "pipeline.ts:prefill", "layer.ts:attention")
 * @returns {void}
 */
export function recordSubmit(durationMs, source) {
  if (!TRACK_SUBMITS) return;

  // Global stats
  submitCount++;
  submitTimes.push(durationMs);
  totalSubmitMs += durationMs;
  maxSubmitMs = Math.max(maxSubmitMs, durationMs);
  minSubmitMs = Math.min(minSubmitMs, durationMs);

  // Track by source
  if (source) {
    submitSources.set(source, (submitSources.get(source) || 0) + 1);
  }

  // Phase-specific stats
  const ps = phaseStats[currentPhase];
  ps.count++;
  ps.times.push(durationMs);
  ps.totalMs += durationMs;
  ps.maxMs = Math.max(ps.maxMs, durationMs);
  ps.minMs = Math.min(ps.minMs, durationMs);

  // Track source in phase stats
  if (source) {
    ps.sources.set(source, (ps.sources.get(source) || 0) + 1);
  }
}

/**
 * Get current submit statistics.
 * @returns {import('./submit-tracker.js').SubmitStats}
 */
export function getSubmitStats() {
  return {
    count: submitCount,
    totalMs: totalSubmitMs,
    avgMs: submitCount > 0 ? totalSubmitMs / submitCount : 0,
    maxMs: maxSubmitMs,
    minMs: minSubmitMs === Infinity ? 0 : minSubmitMs,
    timestamps: [...submitTimes],
    bySource: new Map(submitSources),
  };
}

/**
 * Get submit statistics for a specific phase.
 * @param {SubmitPhase} phase - The phase to get stats for
 * @returns {import('./submit-tracker.js').SubmitStats}
 */
export function getPhaseSubmitStats(phase) {
  const ps = phaseStats[phase];
  return {
    count: ps.count,
    totalMs: ps.totalMs,
    avgMs: ps.count > 0 ? ps.totalMs / ps.count : 0,
    maxMs: ps.maxMs,
    minMs: ps.minMs === Infinity ? 0 : ps.minMs,
    timestamps: [...ps.times],
    bySource: new Map(ps.sources),
  };
}

/**
 * Get submit statistics for all phases.
 * @returns {import('./submit-tracker.js').PhaseSubmitStats}
 */
export function getAllPhaseSubmitStats() {
  return {
    prefill: getPhaseSubmitStats('prefill'),
    decode: getPhaseSubmitStats('decode'),
    other: getPhaseSubmitStats('other'),
  };
}

/**
 * Log submit statistics summary.
 * @param {string} [label] - Label for the log output
 * @returns {void}
 */
export function logSubmitStats(label = 'Forward pass') {
  const stats = getSubmitStats();
  trace.perf(
    `SubmitTracker ${label}: ${stats.count} submits, ` +
    `total=${stats.totalMs.toFixed(2)}ms, ` +
    `avg=${stats.avgMs.toFixed(3)}ms, ` +
    `range=[${stats.minMs.toFixed(3)}-${stats.maxMs.toFixed(3)}ms]`
  );

  // Log by source if available
  if (stats.bySource && stats.bySource.size > 0) {
    trace.perf('SubmitTracker: Submits by source:');
    const sorted = Array.from(stats.bySource.entries()).sort((a, b) => b[1] - a[1]);
    for (const [source, count] of sorted) {
      const pct = ((count / stats.count) * 100).toFixed(1);
      trace.perf(`  ${source}: ${count} (${pct}%)`);
    }
  }
}

/**
 * Log submit statistics for all phases.
 * @param {string} [label] - Label for the log output
 * @returns {void}
 */
export function logAllPhaseSubmitStats(label = 'All phases') {
  const allStats = getAllPhaseSubmitStats();
  trace.perf(`SubmitTracker ${label}:`);

  for (const phase of /** @type {const} */ (['prefill', 'decode', 'other'])) {
    const stats = allStats[phase];
    if (stats.count === 0) continue;

    trace.perf(
      `  ${phase}: ${stats.count} submits, ` +
      `total=${stats.totalMs.toFixed(2)}ms, ` +
      `avg=${stats.avgMs.toFixed(3)}ms`
    );

    // Log by source for this phase
    if (stats.bySource && stats.bySource.size > 0) {
      const sorted = Array.from(stats.bySource.entries()).sort((a, b) => b[1] - a[1]);
      for (const [source, count] of sorted) {
        const pct = ((count / stats.count) * 100).toFixed(1);
        trace.perf(`    ${source}: ${count} (${pct}%)`);
      }
    }
  }
}

/**
 * Extract source from stack trace.
 * @returns {string}
 */
function extractSourceFromStack() {
  const stack = new Error().stack;
  if (!stack) return 'unknown';

  const lines = stack.split('\n');
  // Skip first 3 lines: Error, extractSourceFromStack, queue.submit wrapper
  for (let i = 3; i < lines.length; i++) {
    const line = lines[i];
    // Match file:line pattern in stack trace
    // Example: "at functionName (http://localhost:8080/path/to/file.ts:123:45)"
    const match = line.match(/\/([^\/]+\.ts):(\d+):/);
    if (match) {
      return `${match[1]}:${match[2]}`;
    }
  }
  return 'unknown';
}

/**
 * Wrap a GPU queue to track submit calls.
 * @param {GPUQueue} queue - GPU queue to wrap
 * @returns {GPUQueue}
 */
export function wrapQueueForTracking(queue) {
  const originalSubmit = queue.submit.bind(queue);

  /** @type {any} */ (queue).submit = function(/** @type {Iterable<GPUCommandBuffer>} */ commandBuffers) {
    const start = TRACK_SUBMITS ? performance.now() : 0;
    const result = originalSubmit(commandBuffers);
    trackSubmit();

    if (!TRACK_SUBMITS) {
      return result;
    }

    const duration = performance.now() - start;
    recordSubmit(duration, extractSourceFromStack());
    return result;
  };

  return queue;
}

/**
 * Estimate submit overhead savings from batching.
 * @param {import('./submit-tracker.js').SubmitStats} currentStats - Current submit stats (unbatched)
 * @param {number} [targetSubmits] - Target number of submits after batching
 * @returns {{ savedSubmits: number; estimatedSavingsMs: number }}
 */
export function estimateBatchingSavings(
  currentStats,
  targetSubmits = 1
) {
  const savedSubmits = Math.max(0, currentStats.count - targetSubmits);
  // Each submit has overhead, estimate savings based on average submit time
  const estimatedSavingsMs = savedSubmits * currentStats.avgMs;

  return {
    savedSubmits,
    estimatedSavingsMs,
  };
}
