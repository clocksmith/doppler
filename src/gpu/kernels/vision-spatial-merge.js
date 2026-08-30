import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { GPU_LIMITS, WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './kernel-execution.js';
import { selectRuleValue } from './rule-registry.js';

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Vision spatial merge requires ${label} to be a positive integer.`);
  }
}

function requireBoolean(value, label) {
  if (typeof value !== 'boolean') {
    throw new Error(`Vision spatial merge requires ${label} to be boolean.`);
  }
}

export function planVisionSpatialMergeDispatch(outputElements) {
  requirePositiveInteger(outputElements, 'outputElements');
  const workgroups = Math.ceil(outputElements / WORKGROUP_SIZES.DEFAULT);
  const x = Math.min(workgroups, GPU_LIMITS.MAX_WORKGROUPS);
  const y = Math.ceil(workgroups / x);
  if (y > GPU_LIMITS.MAX_WORKGROUPS) {
    throw new Error(
      `Vision spatial merge requires ${workgroups} workgroups, exceeding the two-axis WebGPU capacity.`
    );
  }
  return [x, y, 1];
}

export async function runVisionSpatialMerge(input, geometry) {
  const {
    gridHeight,
    gridWidth,
    hiddenSize,
    mergeSize,
    channelFirst,
    inputBlockMajor,
  } = geometry;
  for (const [label, value] of Object.entries({ gridHeight, gridWidth, hiddenSize, mergeSize })) {
    requirePositiveInteger(value, label);
  }
  requireBoolean(channelFirst, 'channelFirst');
  requireBoolean(inputBlockMajor, 'inputBlockMajor');
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
  const variant = selectRuleValue('visionSpatialMerge', 'variant', { inputDtype: input.dtype });
  const outputBuffer = acquireBuffer(outputElements * Float32Array.BYTES_PER_ELEMENT, undefined, 'vision_spatial_merge_output');
  let succeeded = false;
  try {
    await unifiedKernelWrapper(
      'vision_spatial_merge',
      null,
      variant,
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
      planVisionSpatialMergeDispatch(outputElements),
      { CHANNEL_FIRST: channelFirst, INPUT_BLOCK_MAJOR: inputBlockMajor }
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
