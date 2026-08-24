import { readBuffer } from '../../../../memory/buffer-pool.js';
import { isBufferStatsEnabled } from './config.js';

export function getLogitsHealth(logits) {
  let nanCount = 0;
  let infCount = 0;
  let nonZeroCount = 0;
  let maxAbs = 0;

  for (let i = 0; i < logits.length; i++) {
    const value = logits[i];
    if (Number.isNaN(value)) {
      nanCount++;
      continue;
    }
    if (!Number.isFinite(value)) {
      infCount++;
      continue;
    }
    if (value !== 0) {
      nonZeroCount++;
      const abs = Math.abs(value);
      if (abs > maxAbs) maxAbs = abs;
    }
  }

  return { nanCount, infCount, nonZeroCount, maxAbs };
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
