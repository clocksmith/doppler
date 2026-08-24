import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { getDevice } from '../device.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './kernel-execution.js';
import { selectRuleValue } from './rule-registry.js';

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Vision patch embedding requires ${label} to be a positive integer.`);
  }
}

function requireTensor(tensor, label) {
  if (!tensor?.buffer || !Array.isArray(tensor.shape) || typeof tensor.dtype !== 'string') {
    throw new Error(`Vision patch embedding requires ${label} to be a typed GPU tensor.`);
  }
}

export async function runVisionPatchEmbed(imageData, weight, bias, geometry) {
  const {
    gridHeight,
    gridWidth,
    channels,
    patchSize,
    temporalPatchSize,
    hiddenSize,
  } = geometry;
  for (const [label, value] of Object.entries({
    gridHeight,
    gridWidth,
    channels,
    patchSize,
    temporalPatchSize,
    hiddenSize,
  })) {
    requirePositiveInteger(value, label);
  }
  if (!(imageData instanceof Float32Array)) {
    throw new Error('Vision patch embedding requires Float32Array image data.');
  }
  const imageElements = channels * gridHeight * patchSize * gridWidth * patchSize;
  if (imageData.length !== imageElements) {
    throw new Error(
      `Vision patch embedding image length mismatch: got ${imageData.length}, expected ${imageElements}.`
    );
  }

  requireTensor(weight, 'weight');
  if (weight.dtype !== 'f32' && weight.dtype !== 'f16') {
    throw new Error(`Vision patch embedding requires f32 or f16 weights, got ${weight.dtype}.`);
  }
  const expectedWeightElements = hiddenSize * channels * temporalPatchSize * patchSize * patchSize;
  const weightElements = weight.shape.reduce((product, value) => product * value, 1);
  if (weightElements !== expectedWeightElements) {
    throw new Error(
      `Vision patch embedding weight shape mismatch: got ${weightElements} elements, expected ${expectedWeightElements}.`
    );
  }
  if (weight.buffer.size < expectedWeightElements * dtypeBytes(weight.dtype)) {
    throw new Error('Vision patch embedding weight buffer is smaller than its declared shape.');
  }
  if (bias != null) {
    requireTensor(bias, 'bias');
    if (bias.dtype !== weight.dtype) {
      throw new Error(`Vision patch embedding bias dtype ${bias.dtype} does not match weight dtype ${weight.dtype}.`);
    }
    const biasElements = bias.shape.reduce((product, value) => product * value, 1);
    if (biasElements !== hiddenSize) {
      throw new Error(`Vision patch embedding bias must contain ${hiddenSize} elements, got ${biasElements}.`);
    }
  }

  const variant = selectRuleValue('visionPatchEmbed', 'variant', { weightDtype: weight.dtype });
  const outputElements = gridHeight * gridWidth * hiddenSize;
  const inputBuffer = acquireBuffer(imageData.byteLength, undefined, 'vision_patch_embed_input');
  const outputBuffer = acquireBuffer(outputElements * Float32Array.BYTES_PER_ELEMENT, undefined, 'vision_patch_embed_output');
  let succeeded = false;
  try {
    const inputTensor = createTensor(
      inputBuffer,
      'f32',
      [channels, gridHeight * patchSize, gridWidth * patchSize],
      'vision_patch_embed_input'
    );
    getDevice().queue.writeBuffer(inputBuffer, 0, imageData);
    await unifiedKernelWrapper(
      'vision_patch_embed',
      null,
      variant,
      [inputTensor, weight, bias ?? weight, outputBuffer],
      {
        grid_height: gridHeight,
        grid_width: gridWidth,
        channels,
        patch_size: patchSize,
        temporal_patch_size: temporalPatchSize,
        hidden_size: hiddenSize,
        has_bias: bias == null ? 0 : 1,
        output_elements: outputElements,
      },
      Math.ceil(outputElements / WORKGROUP_SIZES.DEFAULT)
    );
    succeeded = true;
    return createTensor(
      outputBuffer,
      'f32',
      [gridHeight * gridWidth, hiddenSize],
      'vision_patch_embed_output'
    );
  } finally {
    releaseBuffer(inputBuffer);
    if (!succeeded) {
      releaseBuffer(outputBuffer);
    }
  }
}
