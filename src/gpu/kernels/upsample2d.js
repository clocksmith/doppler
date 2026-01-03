import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { unifiedKernelWrapper } from './utils.js';
import { selectRuleValue } from './rule-registry.js';
import { WORKGROUP_SIZES } from './constants.js';

function selectUpsample2DVariant(isF16) {
  return selectRuleValue('upsample2d', 'variant', { isF16 });
}

async function _upsample2d(target, input, options = {}) {
  const {
    channels,
    height,
    width,
    inHeight,
    inWidth,
    scale = 2,
    outputBuffer = null,
  } = options;

  const resolvedHeight = Number.isFinite(height) ? height : inHeight;
  const resolvedWidth = Number.isFinite(width) ? width : inWidth;

  if (!Number.isFinite(channels) || !Number.isFinite(resolvedHeight) || !Number.isFinite(resolvedWidth)) {
    throw new Error('Upsample2D requires channels/height/width.');
  }
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error('Upsample2D requires scale > 0.');
  }

  const outHeight = resolvedHeight * scale;
  const outWidth = resolvedWidth * scale;
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = channels * outHeight * outWidth * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'upsample2d_output');

  await unifiedKernelWrapper(
    'upsample2d', target, selectUpsample2DVariant(input.dtype === 'f16'),
    [input, output],
    {
      channels, in_height: resolvedHeight, in_width: resolvedWidth,
      out_height: outHeight, out_width: outWidth, scale,
      _pad0: 0, _pad1: 0,
    },
    Math.ceil((channels * outHeight * outWidth) / WORKGROUP_SIZES.DEFAULT)
  );

  return createTensor(output, input.dtype, [channels, outHeight, outWidth], 'upsample2d_output');
}

export async function runUpsample2D(input, options = {}) {
  return _upsample2d(null, input, options);
}

export async function recordUpsample2D(recorder, input, options = {}) {
  return _upsample2d(recorder, input, options);
}
