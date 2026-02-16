

import { readBuffer } from '../../../../memory/buffer-pool.js';
import { isBufferStatsEnabled } from './config.js';

// ============================================================================
// Math Helpers
// ============================================================================


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


export function decodeReadback(buffer, dtype) {
  if (dtype === 'f32') {
    return new Float32Array(buffer);
  }
  const src = new Uint16Array(buffer);
  const out = new Float32Array(src.length);
  if (dtype === 'bf16') {
    const tmp = new Uint32Array(1);
    const f32View = new Float32Array(tmp.buffer);
    for (let i = 0; i < src.length; i++) {
      tmp[0] = src[i] << 16;
      out[i] = f32View[0];
    }
    return out;
  }
  for (let i = 0; i < src.length; i++) {
    out[i] = f16ToF32(src[i]);
  }
  return out;
}

// ============================================================================
// Health Checks
// ============================================================================


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


export const DEBUG_PRESETS = {
  
  quick: { embed: true, logits: true, sample: true },

  
  layers: { layer: true },

  
  attention: { attn: true, kv: true },

  
  full: { all: true },

  
  perf: { perf: true },

  
  kernelStep: { kernel: true },
};
