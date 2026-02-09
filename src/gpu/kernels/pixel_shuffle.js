import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { unifiedKernelWrapper } from './utils.js';
import { selectRuleValue } from './rule-registry.js';

function selectPixelShuffleVariant(dtype) {
  return selectRuleValue('pixel_shuffle', 'variant', { dtype });
}

async function _pixelShuffle(target, input, options = {}) {
  const {
    outChannels, outHeight, outWidth,
    gridWidth, gridHeight, patchSize,
    patchChannels, outputBuffer = null,
  } = options;

  if (!Number.isFinite(outChannels) || !Number.isFinite(outHeight) || !Number.isFinite(outWidth) ||
      !Number.isFinite(gridWidth) || !Number.isFinite(gridHeight) || !Number.isFinite(patchSize)) {
    throw new Error('PixelShuffle requires explicit dimensions.');
  }

  const inferredPatchChannels = patchChannels ?? outChannels * patchSize * patchSize;
  const variant = selectPixelShuffleVariant(input.dtype);
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = outChannels * outHeight * outWidth * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'pixel_shuffle_output');

  await unifiedKernelWrapper(
    'pixel_shuffle', target, variant,
    [input, output],
    {
      out_channels: outChannels, out_height: outHeight, out_width: outWidth,
      grid_width: gridWidth, grid_height: gridHeight, patch_size: patchSize,
      patch_channels: inferredPatchChannels, _pad0: 0,
    },
    Math.ceil((outChannels * outHeight * outWidth) / 256)
  );

  return createTensor(output, input.dtype, [outChannels, outHeight, outWidth], 'pixel_shuffle_output');
}

export async function runPixelShuffle(input, options = {}) {
  return _pixelShuffle(null, input, options);
}

export async function recordPixelShuffle(recorder, input, options = {}) {
  return _pixelShuffle(recorder, input, options);
}
