/**
 * Visual token scatter — injects vision encoder output into text decoder hidden states.
 *
 * findTokenPositions: locates image_pad token positions in the input token IDs.
 * scatterVisionTokens: overwrites hidden states at pad positions with visual tokens.
 *
 * For DeepStack models, deepstack tokens are injected at intermediate decoder
 * layers (e.g., layers 5, 11, 17) using separate merger projections.
 */

import { readBuffer } from '../memory/buffer-pool.js';
import { selectRuleValue } from '../rules/rule-registry.js';

export function findTokenPositions(tokenIds, targetId) {
  const positions = [];
  for (let i = 0; i < tokenIds.length; i++) {
    if (tokenIds[i] === targetId) positions.push(i);
  }
  return positions;
}

export async function scatterVisionTokens(hiddenStatesBuffer, visualTokens, padPositions, hiddenSize, activationDtype) {
  if (padPositions.length === 0 || visualTokens.length === 0) return;

  const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });
  const tokenBytes = hiddenSize * bytesPerElement;

  const device = hiddenStatesBuffer.device ?? null;
  if (!device) {
    throw new Error('[Vision] Cannot scatter: hiddenStatesBuffer has no device reference');
  }

  const numVisualTokens = visualTokens.length / hiddenSize;
  const count = Math.min(padPositions.length, numVisualTokens);

  for (let i = 0; i < count; i++) {
    const pos = padPositions[i];
    const offset = pos * tokenBytes;
    const srcOffset = i * hiddenSize;

    if (activationDtype === 'f16') {
      const f16 = new Uint16Array(hiddenSize);
      for (let j = 0; j < hiddenSize; j++) {
        f16[j] = f32ToF16(visualTokens[srcOffset + j]);
      }
      device.queue.writeBuffer(hiddenStatesBuffer, offset, f16.buffer);
    } else {
      const slice = visualTokens.subarray(srcOffset, srcOffset + hiddenSize);
      device.queue.writeBuffer(hiddenStatesBuffer, offset, slice.buffer, slice.byteOffset, slice.byteLength);
    }
  }
}

function f32ToF16(value) {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = value;
  const bits = u32[0];
  const sign = (bits >>> 31) & 1;
  const exp = (bits >>> 23) & 0xff;
  const frac = bits & 0x7fffff;

  if (exp === 0xff) {
    return (sign << 15) | 0x7c00 | (frac ? 0x200 : 0);
  }
  if (exp === 0) {
    return sign << 15;
  }
  const newExp = exp - 127 + 15;
  if (newExp >= 31) {
    return (sign << 15) | 0x7c00;
  }
  if (newExp <= 0) {
    if (newExp < -10) return sign << 15;
    const mant = (frac | 0x800000) >> (1 - newExp);
    return (sign << 15) | (mant >> 13);
  }
  return (sign << 15) | (newExp << 10) | (frac >> 13);
}
