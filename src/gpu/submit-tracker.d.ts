/**
 * Submit Tracker - Measures GPU submit overhead for optimization benchmarking.
 *
 * @module gpu/submit-tracker
 */

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
 * Get current submit statistics.
 * @returns Submit statistics
 */
export function getSubmitStats(): SubmitStats;

/**
 * Log submit statistics summary.
 * @param label - Label for the log output
 */
export function logSubmitStats(label?: string): void;

/**
 * Wrap a GPU queue to track submit calls.
 * @param queue - GPU queue to wrap
 * @returns Wrapped queue with tracking
 */
export function wrapQueueForTracking(queue: GPUQueue): GPUQueue;
