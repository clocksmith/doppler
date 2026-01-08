/**
 * DOPPLER Debug Module - Performance Timing Utilities
 *
 * Tools for measuring and logging execution times.
 *
 * @module debug/perf
 */

import { log } from './log.js';

// ============================================================================
// Performance Timing Interface
// ============================================================================

/**
 * Performance timing utilities.
 */
export const perf = {
  marks: new Map<string, number>(),

  /**
   * Start a timing mark.
   */
  mark(label: string): void {
    this.marks.set(label, performance.now());
  },

  /**
   * End a timing mark and log duration.
   */
  measure(label: string, module = 'Perf'): number {
    const start = this.marks.get(label);
    if (start === undefined) {
      log.warn(module, `No mark found for "${label}"`);
      return 0;
    }

    const duration = performance.now() - start;
    this.marks.delete(label);
    log.debug(module, `${label}: ${duration.toFixed(2)}ms`);
    return duration;
  },

  /**
   * Time an async operation.
   */
  async time<T>(label: string, fn: () => Promise<T>): Promise<{ result: T; durationMs: number }> {
    const start = performance.now();
    const result = await fn();
    const durationMs = performance.now() - start;
    log.debug('Perf', `${label}: ${durationMs.toFixed(2)}ms`);
    return { result, durationMs };
  },
};
