/**
 * DOPPLER Debug Module - Performance Timing Utilities
 *
 * Tools for measuring and logging execution times.
 *
 * @module debug/perf
 */

// ============================================================================
// Performance Timing Interface
// ============================================================================

/**
 * Performance timing utilities.
 */
export declare const perf: {
  marks: Map<string, number>;

  /**
   * Start a timing mark.
   */
  mark(label: string): void;

  /**
   * End a timing mark and log duration.
   */
  measure(label: string, module?: string): number;

  /**
   * Time an async operation.
   */
  time<T>(label: string, fn: () => Promise<T>): Promise<{ result: T; durationMs: number }>;
};
