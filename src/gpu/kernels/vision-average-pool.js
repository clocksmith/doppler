import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './kernel-execution.js';
import { selectRuleValue } from './rule-registry.js';

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Vision average pool requires ${label} to be a positive integer.`);
  }
}

export async function runVisionAveragePool(input, geometry) {
  const { gridHeight, gridWidth, hiddenSize, poolingSize } = geometry;
  for (const [label, value] of Object.entries({ gridHeight, gridWidth, hiddenSize, poolingSize })) {
    requirePositiveInteger(value, label);
  }
  if (!input?.buffer || input.dtype !== 'f32') {
    throw new Error('Vision average pool requires an f32 input tensor.');
  }
  if (gridHeight % poolingSize !== 0 || gridWidth % poolingSize !== 0) {
    throw new Error(
      `Vision average pool grid ${gridHeight}x${gridWidth} must be divisible by poolingSize=${poolingSize}.`
    );
  }
  const inputElements = input.shape.reduce((product, value) => product * value, 1);
  const expectedElements = gridHeight * gridWidth * hiddenSize;
  if (inputElements !== expectedElements) {
    throw new Error(`Vision average pool input has ${inputElements} elements, expected ${expectedElements}.`);
  }

  const pooledHeight = gridHeight / poolingSize;
  const pooledWidth = gridWidth / poolingSize;
  const outputElements = pooledHeight * pooledWidth * hiddenSize;
  const variant = selectRuleValue('visionAveragePool', 'variant', { inputDtype: input.dtype });
  const outputBuffer = acquireBuffer(outputElements * Float32Array.BYTES_PER_ELEMENT, undefined, 'vision_average_pool_output');
  let succeeded = false;
  try {
    await unifiedKernelWrapper(
      'vision_average_pool',
      null,
      variant,
      [input, outputBuffer],
      {
        grid_height: gridHeight,
        grid_width: gridWidth,
        hidden_size: hiddenSize,
        pooling_size: poolingSize,
        pooled_height: pooledHeight,
        pooled_width: pooledWidth,
        output_elements: outputElements,
        _pad0: 0,
      },
      Math.ceil(outputElements / WORKGROUP_SIZES.DEFAULT)
    );
    succeeded = true;
    return createTensor(
      outputBuffer,
      'f32',
      [pooledHeight * pooledWidth, hiddenSize],
      'vision_average_pool_output'
    );
  } finally {
    if (!succeeded) {
      releaseBuffer(outputBuffer);
    }
  }
}
