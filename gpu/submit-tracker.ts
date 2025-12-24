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
 *   console.log(`Submits: ${stats.count}, Total time: ${stats.totalMs.toFixed(2)}ms`);
 *
 * To enable tracking, set TRACK_SUBMITS = true and wrap queue.submit calls.
 *
 * @module gpu/submit-tracker
 */

import { trackSubmit } from './perf-guards.js';

/** Whether to track submits (disable in production for perf) */
export let TRACK_SUBMITS = false;

/** Submit statistics */
export interface SubmitStats {
  /** Number of queue.submit() calls */
  count: number;
  /** Total time spent in submit calls (ms) */
  totalMs: number;
  /** Average time per submit (ms) */
  avgMs: number;
  /** Max time for a single submit (ms) */
  maxMs: number;
  /** Min time for a single submit (ms) */
  minMs: number;
  /** Submit timestamps for detailed analysis */
  timestamps: number[];
  /** Submit counts by source */
  bySource?: Map<string, number>;
}

/** Phase-based submit statistics */
export interface PhaseSubmitStats {
  prefill: SubmitStats;
  decode: SubmitStats;
  other: SubmitStats;
}

/** Current phase for submit tracking */
export type SubmitPhase = 'prefill' | 'decode' | 'other';

/** Internal tracking state */
let submitCount = 0;
let submitTimes: number[] = [];
let totalSubmitMs = 0;
let maxSubmitMs = 0;
let minSubmitMs = Infinity;
let submitSources = new Map<string, number>();

/** Phase-based tracking state */
let currentPhase: SubmitPhase = 'other';
const phaseStats: Record<SubmitPhase, { count: number; times: number[]; totalMs: number; maxMs: number; minMs: number; sources: Map<string, number> }> = {
  prefill: { count: 0, times: [], totalMs: 0, maxMs: 0, minMs: Infinity, sources: new Map() },
  decode: { count: 0, times: [], totalMs: 0, maxMs: 0, minMs: Infinity, sources: new Map() },
  other: { count: 0, times: [], totalMs: 0, maxMs: 0, minMs: Infinity, sources: new Map() },
};

/**
 * Enable/disable submit tracking.
 * @param enabled - Whether to track submits
 */
export function setTrackSubmits(enabled: boolean): void {
  TRACK_SUBMITS = enabled;
  if (enabled) {
    resetSubmitStats();
    console.log('[SubmitTracker] Enabled');
  } else {
    console.log('[SubmitTracker] Disabled');
  }
}

/**
 * Reset submit statistics.
 * Call before starting a new measurement.
 */
export function resetSubmitStats(): void {
  submitCount = 0;
  submitTimes = [];
  totalSubmitMs = 0;
  maxSubmitMs = 0;
  minSubmitMs = Infinity;
  submitSources = new Map();
  currentPhase = 'other';

  // Reset phase stats
  for (const phase of ['prefill', 'decode', 'other'] as const) {
    phaseStats[phase] = { count: 0, times: [], totalMs: 0, maxMs: 0, minMs: Infinity, sources: new Map() };
  }
}

/**
 * Set the current phase for submit tracking.
 * @param phase - The phase to track ('prefill', 'decode', or 'other')
 */
export function setSubmitPhase(phase: SubmitPhase): void {
  currentPhase = phase;
}

/**
 * Record a submit call.
 * Call this from a wrapper around queue.submit().
 * @param durationMs - Time spent in this submit call
 * @param source - Optional source identifier (e.g., "pipeline.ts:prefill", "layer.ts:attention")
 */
export function recordSubmit(durationMs: number, source?: string): void {
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
 * @returns Submit statistics
 */
export function getSubmitStats(): SubmitStats {
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
 * @param phase - The phase to get stats for
 * @returns Submit statistics for the phase
 */
export function getPhaseSubmitStats(phase: SubmitPhase): SubmitStats {
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
 * @returns Submit statistics by phase
 */
export function getAllPhaseSubmitStats(): PhaseSubmitStats {
  return {
    prefill: getPhaseSubmitStats('prefill'),
    decode: getPhaseSubmitStats('decode'),
    other: getPhaseSubmitStats('other'),
  };
}

/**
 * Log submit statistics summary.
 * @param label - Label for the log output
 */
export function logSubmitStats(label: string = 'Forward pass'): void {
  const stats = getSubmitStats();
  console.log(
    `[SubmitTracker] ${label}: ${stats.count} submits, ` +
    `total=${stats.totalMs.toFixed(2)}ms, ` +
    `avg=${stats.avgMs.toFixed(3)}ms, ` +
    `range=[${stats.minMs.toFixed(3)}-${stats.maxMs.toFixed(3)}ms]`
  );

  // Log by source if available
  if (stats.bySource && stats.bySource.size > 0) {
    console.log('[SubmitTracker] Submits by source:');
    const sorted = Array.from(stats.bySource.entries()).sort((a, b) => b[1] - a[1]);
    for (const [source, count] of sorted) {
      const pct = ((count / stats.count) * 100).toFixed(1);
      console.log(`  ${source}: ${count} (${pct}%)`);
    }
  }
}

/**
 * Log submit statistics for all phases.
 * @param label - Label for the log output
 */
export function logAllPhaseSubmitStats(label: string = 'All phases'): void {
  const allStats = getAllPhaseSubmitStats();
  console.log(`[SubmitTracker] ${label}:`);

  for (const phase of ['prefill', 'decode', 'other'] as const) {
    const stats = allStats[phase];
    if (stats.count === 0) continue;

    console.log(
      `  ${phase}: ${stats.count} submits, ` +
      `total=${stats.totalMs.toFixed(2)}ms, ` +
      `avg=${stats.avgMs.toFixed(3)}ms`
    );

    // Log by source for this phase
    if (stats.bySource && stats.bySource.size > 0) {
      const sorted = Array.from(stats.bySource.entries()).sort((a, b) => b[1] - a[1]);
      for (const [source, count] of sorted) {
        const pct = ((count / stats.count) * 100).toFixed(1);
        console.log(`    ${source}: ${count} (${pct}%)`);
      }
    }
  }
}

/**
 * Extract source from stack trace.
 * @returns Source identifier (e.g., "pipeline.ts:456" or "unknown")
 */
function extractSourceFromStack(): string {
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
 * @param queue - GPU queue to wrap
 * @returns Wrapped queue with tracking
 */
export function wrapQueueForTracking(queue: GPUQueue): GPUQueue {
  const originalSubmit = queue.submit.bind(queue);

  (queue as any).submit = function(commandBuffers: Iterable<GPUCommandBuffer>): undefined {
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
 * @param currentStats - Current submit stats (unbatched)
 * @param targetSubmits - Target number of submits after batching
 * @returns Estimated time savings in ms
 */
export function estimateBatchingSavings(
  currentStats: SubmitStats,
  targetSubmits: number = 1
): { savedSubmits: number; estimatedSavingsMs: number } {
  const savedSubmits = Math.max(0, currentStats.count - targetSubmits);
  // Each submit has overhead, estimate savings based on average submit time
  const estimatedSavingsMs = savedSubmits * currentStats.avgMs;

  return {
    savedSubmits,
    estimatedSavingsMs,
  };
}
