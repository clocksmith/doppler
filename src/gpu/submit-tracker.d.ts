/**
 * Submit Tracker - Measures GPU submit overhead for optimization benchmarking.
 *
 * @module gpu/submit-tracker
 */

/** Whether to track submits (disable in production for perf) */
export let TRACK_SUBMITS: boolean;

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

/**
 * Enable/disable submit tracking.
 * @param enabled - Whether to track submits
 */
export function setTrackSubmits(enabled: boolean): void;

/**
 * Reset submit statistics.
 * Call before starting a new measurement.
 */
export function resetSubmitStats(): void;

/**
 * Set the current phase for submit tracking.
 * @param phase - The phase to track ('prefill', 'decode', or 'other')
 */
export function setSubmitPhase(phase: SubmitPhase): void;

/**
 * Record a submit call.
 * Call this from a wrapper around queue.submit().
 * @param durationMs - Time spent in this submit call
 * @param source - Optional source identifier (e.g., "pipeline.ts:prefill", "layer.ts:attention")
 */
export function recordSubmit(durationMs: number, source?: string): void;

/**
 * Get current submit statistics.
 * @returns Submit statistics
 */
export function getSubmitStats(): SubmitStats;

/**
 * Get submit statistics for a specific phase.
 * @param phase - The phase to get stats for
 * @returns Submit statistics for the phase
 */
export function getPhaseSubmitStats(phase: SubmitPhase): SubmitStats;

/**
 * Get submit statistics for all phases.
 * @returns Submit statistics by phase
 */
export function getAllPhaseSubmitStats(): PhaseSubmitStats;

/**
 * Log submit statistics summary.
 * @param label - Label for the log output
 */
export function logSubmitStats(label?: string): void;

/**
 * Log submit statistics for all phases.
 * @param label - Label for the log output
 */
export function logAllPhaseSubmitStats(label?: string): void;

/**
 * Wrap a GPU queue to track submit calls.
 * @param queue - GPU queue to wrap
 * @returns Wrapped queue with tracking
 */
export function wrapQueueForTracking(queue: GPUQueue): GPUQueue;

/**
 * Estimate submit overhead savings from batching.
 * @param currentStats - Current submit stats (unbatched)
 * @param targetSubmits - Target number of submits after batching
 * @returns Estimated time savings in ms
 */
export function estimateBatchingSavings(
  currentStats: SubmitStats,
  targetSubmits?: number
): { savedSubmits: number; estimatedSavingsMs: number };
