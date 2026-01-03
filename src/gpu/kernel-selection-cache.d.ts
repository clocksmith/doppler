/**
 * Kernel Cache Warmth Tracker
 *
 * Tracks when the first forward pass completes so callers can gate
 * any warm-up behavior.
 *
 * @module gpu/kernel-selection-cache
 */

/**
 * Mark the cache as warmed (first forward pass complete).
 */
export function markWarmed(): void;
