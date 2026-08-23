import { recordScale } from '../../../../gpu/kernel-selector.js';
import { resolveQueryScale } from './heterogeneous-contract.js';

export async function applyAttentionQueryScale({ recorder, tensor, scale, count, release, observe }) {
  const resolvedScale = resolveQueryScale(scale);
  if (resolvedScale === 1) return tensor;
  const output = await recordScale(recorder, tensor, resolvedScale, { count });
  release(tensor.buffer);
  if (observe) await observe(output);
  return output;
}
