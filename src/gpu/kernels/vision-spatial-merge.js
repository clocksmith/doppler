import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './kernel-execution.js';

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Vision spatial merge requires ${label} to be a positive integer.`);
  }
}

export async function runVisionSpatialMerge(input, geometry) {
  const { gridHeight, gridWidth, hiddenSize, mergeSize } = geometry;
  for (const [label, value] of Object.entries({ gridHeight, gridWidth, hiddenSize, mergeSize })) {
    requirePositiveInteger(value, label);
  }
  if (!input?.buffer || input.dtype !== 'f32') {
    throw new Error('Vision spatial merge requires an f32 input tensor.');
  }
  if (gridHeight % mergeSize !== 0 || gridWidth % mergeSize !== 0) {
    throw new Error(
      `Vision spatial merge grid ${gridHeight}x${gridWidth} must be divisible by mergeSize=${mergeSize}.`
    );
  }
  const inputElements = gridHeight * gridWidth * hiddenSize;
  const declaredInputElements = input.shape.reduce((product, value) => product * value, 1);
  if (declaredInputElements !== inputElements) {
    throw new Error(
      `Vision spatial merge input shape mismatch: got ${declaredInputElements} elements, expected ${inputElements}.`
    );
  }

  const mergedHeight = gridHeight / mergeSize;
  const mergedWidth = gridWidth / mergeSize;
  const concatDim = mergeSize * mergeSize * hiddenSize;
  const outputElements = mergedHeight * mergedWidth * concatDim;
  const outputBuffer = acquireBuffer(outputElements * Float32Array.BYTES_PER_ELEMENT, undefined, 'vision_spatial_merge_output');
  let succeeded = false;
  try {
    await unifiedKernelWrapper(
      'vision_spatial_merge',
      null,
      'default',
      [input, outputBuffer],
      {
        grid_height: gridHeight,
        grid_width: gridWidth,
        hidden_size: hiddenSize,
        merge_size: mergeSize,
        merged_height: mergedHeight,
        merged_width: mergedWidth,
        output_elements: outputElements,
        _pad0: 0,
      },
      Math.ceil(outputElements / WORKGROUP_SIZES.DEFAULT)
    );
    succeeded = true;
    return createTensor(
      outputBuffer,
      'f32',
      [mergedHeight * mergedWidth, concatDim],
      'vision_spatial_merge_output'
    );
  } finally {
    if (!succeeded) {
      releaseBuffer(outputBuffer);
    }
  }
}
