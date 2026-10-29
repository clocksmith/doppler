import { recordScale } from '../../../../gpu/kernels/scale.js';
import { resolveQueryScale } from '../../../../config/heterogeneous-attention-contract.js';

export async function applyAttentionQueryScale({ recorder, tensor, scale, count, release, observe }) {
  const resolvedScale = resolveQueryScale(scale);
  if (resolvedScale === 1) return tensor;
  const output = await recordScale(recorder, tensor, resolvedScale, { count });
  release(tensor.buffer);
  if (observe) await observe(output);
  return output;
}
