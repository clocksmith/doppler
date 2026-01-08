/**
 * Kernel Cache Warmth Tracker
 *
 * Tracks when the first forward pass completes so callers can gate
 * any warm-up behavior.
 *
 * @module gpu/kernel-selection-cache
 */

import { log } from '../debug/index.js';

let isWarmed = false;

/**
 * Mark the cache as warmed (first forward pass complete).
 * @returns {void}
 */
export function markWarmed() {
  if (!isWarmed) {
    isWarmed = true;
    log.debug('KernelCache', 'Warmed');
  }
}
