/**
 * General utilities for pipeline debugging.
 *
 * Provides math helpers, buffer decoding, health checks, and preset configurations.
 *
 * @module inference/pipeline/debug-utils/utils
 */

import { readBuffer } from '../../../gpu/buffer-pool.js';
import { isBufferStatsEnabled } from './config.js';

// ============================================================================
// Math Helpers
// ============================================================================

/**
 * Convert a 16-bit float (IEEE 754 half-precision) to 32-bit float.
 * @param {number} h
 * @returns {number}
 */
export function f16ToF32(h) {
  const sign = (h >> 15) & 0x1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;

  if (exp === 0) {
    if (mant === 0) return sign ? -0 : 0;
    const f = mant / 1024 * Math.pow(2, -14);
    return sign ? -f : f;
  }
  if (exp === 31) {
    return mant ? NaN : (sign ? -Infinity : Infinity);
  }

  const f = (1 + mant / 1024) * Math.pow(2, exp - 15);
  return sign ? -f : f;
}

/**
 * Decode a GPU readback buffer to Float32Array.
 * Handles both f16 and f32 dtypes.
 * @param {ArrayBuffer} buffer
 * @param {'f16' | 'f32'} dtype
 * @returns {Float32Array}
 */
export function decodeReadback(buffer, dtype) {
  if (dtype === 'f32') {
    return new Float32Array(buffer);
  }
  const src = new Uint16Array(buffer);
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    out[i] = f16ToF32(src[i]);
  }
  return out;
}

// ============================================================================
// Health Checks
// ============================================================================

/**
 * Analyze logits array for numerical issues.
 * Returns counts of NaN, Inf, non-zero values, and the max absolute value.
 * @param {Float32Array} logits
 * @returns {{ nanCount: number; infCount: number; nonZeroCount: number; maxAbs: number }}
 */
export function getLogitsHealth(logits) {
  let nanCount = 0;
  let infCount = 0;
  let nonZeroCount = 0;
  let maxAbs = 0;

  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (Number.isNaN(v)) {
      nanCount++;
      continue;
    }
    if (!Number.isFinite(v)) {
      infCount++;
      continue;
    }
    if (v !== 0) {
      nonZeroCount++;
      const abs = Math.abs(v);
      if (abs > maxAbs) maxAbs = abs;
    }
  }

  return { nanCount, infCount, nonZeroCount, maxAbs };
}

// ============================================================================
// Buffer Stats (Expensive)
// ============================================================================

/**
 * Read GPU buffer and compute stats. Only use when bufferStats is enabled.
 * @param {GPUBuffer} buffer
 * @returns {Promise<{ min: number; max: number; maxAbs: number; sample: number[]; nanCount: number } | null>}
 */
export async function getBufferStats(buffer) {
  if (!isBufferStatsEnabled()) return null;

  try {
    const data = await readBuffer(buffer);
    const arr = new Float32Array(data);
    let min = Infinity;
    let max = -Infinity;
    let nanCount = 0;

    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (!Number.isFinite(v)) {
        nanCount++;
      } else {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }

    const maxAbs = Math.max(Math.abs(min), Math.abs(max));
    const sample = Array.from(arr.slice(0, 5));

    return { min, max, maxAbs, sample, nanCount };
  } catch {
    return null;
  }
}

// ============================================================================
// Debug Presets
// ============================================================================

/**
 * Preset debug configurations for common debugging scenarios.
 * @type {{ quick: Partial<Record<import('./config.js').DebugCategory, boolean>>; layers: Partial<Record<import('./config.js').DebugCategory, boolean>>; attention: Partial<Record<import('./config.js').DebugCategory, boolean>>; full: Partial<Record<import('./config.js').DebugCategory, boolean>>; perf: Partial<Record<import('./config.js').DebugCategory, boolean>>; kernelStep: Partial<Record<import('./config.js').DebugCategory, boolean>> }}
 */
export const DEBUG_PRESETS = {
  /** Quick check: just embedding and final logits */
  quick: { embed: true, logits: true, sample: true },

  /** Layer tracing: watch values flow through layers */
  layers: { layer: true },

  /** Attention focus: debug attention computation */
  attention: { attn: true, kv: true },

  /** Full trace: everything (very verbose) */
  full: { all: true },

  /** Performance only: timing info */
  perf: { perf: true },

  /** Kernel step debugging: inspect tensor state after every kernel (very slow!) */
  kernelStep: { kernel: true },
};
