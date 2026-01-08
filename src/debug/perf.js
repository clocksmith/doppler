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
  marks: new Map(),

  /**
   * Start a timing mark.
   */
  mark(label) {
    this.marks.set(label, performance.now());
  },

  /**
   * End a timing mark and log duration.
   */
  measure(label, module = 'Perf') {
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
  async time(label, fn) {
    const start = performance.now();
    const result = await fn();
    const durationMs = performance.now() - start;
    log.debug('Perf', `${label}: ${durationMs.toFixed(2)}ms`);
    return { result, durationMs };
  },
};
