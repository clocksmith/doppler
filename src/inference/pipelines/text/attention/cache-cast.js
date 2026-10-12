import { recordCastPairedToF16 } from '../../../../gpu/kernels/paired-cast.js';
import { recordCastF32ToF16 } from '../../../../gpu/kernels/cast.js';

// Register each successful conversion before starting the next conversion or
// cache write, so failures cannot orphan the first temporary.
/** @type {typeof import('./cache-cast.js').recordAttentionCacheCast} */
export async function recordAttentionCacheCast(recorder, tensor, scope, label) {
  const metadata = tensor.precision !== undefined || tensor.components !== undefined || tensor.elementStrideBytes !== undefined;
  if (!metadata && tensor.dtype === 'f16') return tensor;
  const converted = metadata
    ? await recordCastPairedToF16(recorder, tensor)
    : await recordCastF32ToF16(recorder, tensor);
  scope.register(converted.buffer, label, 'submitOwned');
  return converted;
}
