import { computeArrayStats } from '../../../../debug/stats.js';
import { readBuffer } from '../../../../memory/buffer-pool.js';
import { isBufferStatsEnabled } from './config.js';

export function getLogitsHealth(logits) {
  const stats = computeArrayStats(logits);
  return {
    nanCount: stats.nanCount,
    infCount: stats.infCount,
    nonZeroCount: stats.validCount - stats.zeroCount,
    maxAbs: stats.maxAbs,
  };
}

export async function getBufferStats(buffer) {
  if (!isBufferStatsEnabled()) return null;

  try {
    const data = await readBuffer(buffer);
    const values = new Float32Array(data);
    let min = Infinity;
    let max = -Infinity;
    let nanCount = 0;

    for (let i = 0; i < values.length; i++) {
      const value = values[i];
      if (!Number.isFinite(value)) {
        nanCount++;
      } else {
        if (value < min) min = value;
        if (value > max) max = value;
      }
    }

    const maxAbs = Math.max(Math.abs(min), Math.abs(max));
    const sample = Array.from(values.slice(0, 5));

    return { min, max, maxAbs, sample, nanCount };
  } catch {
    return null;
  }
}
