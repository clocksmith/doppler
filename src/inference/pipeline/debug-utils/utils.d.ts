/**
 * General utilities for pipeline debugging.
 *
 * Provides math helpers, buffer decoding, health checks, and preset configurations.
 *
 * @module inference/pipeline/debug-utils/utils
 */

import type { DebugCategory } from './config.js';

// ============================================================================
// Math Helpers
// ============================================================================

/**
 * Convert a 16-bit float (IEEE 754 half-precision) to 32-bit float.
 */
export function f16ToF32(h: number): number;

/**
 * Decode a GPU readback buffer to Float32Array.
 * Handles both f16 and f32 dtypes.
 */
export function decodeReadback(buffer: ArrayBuffer, dtype: 'f16' | 'f32'): Float32Array;

// ============================================================================
// Health Checks
// ============================================================================

/**
 * Analyze logits array for numerical issues.
 * Returns counts of NaN, Inf, non-zero values, and the max absolute value.
 */
export function getLogitsHealth(logits: Float32Array): {
  nanCount: number;
  infCount: number;
  nonZeroCount: number;
  maxAbs: number;
};

// ============================================================================
// Buffer Stats (Expensive)
// ============================================================================

/**
 * Read GPU buffer and compute stats. Only use when bufferStats is enabled.
 */
export function getBufferStats(
  buffer: GPUBuffer
): Promise<{ min: number; max: number; maxAbs: number; sample: number[]; nanCount: number } | null>;

// ============================================================================
// Debug Presets
// ============================================================================

/**
 * Preset debug configurations for common debugging scenarios.
 */
export const DEBUG_PRESETS: {
  /** Quick check: just embedding and final logits */
  quick: Partial<Record<DebugCategory, boolean>>;

  /** Layer tracing: watch values flow through layers */
  layers: Partial<Record<DebugCategory, boolean>>;

  /** Attention focus: debug attention computation */
  attention: Partial<Record<DebugCategory, boolean>>;

  /** Full trace: everything (very verbose) */
  full: Partial<Record<DebugCategory, boolean>>;

  /** Performance only: timing info */
  perf: Partial<Record<DebugCategory, boolean>>;

  /** Kernel step debugging: inspect tensor state after every kernel (very slow!) */
  kernelStep: Partial<Record<DebugCategory, boolean>>;
};
