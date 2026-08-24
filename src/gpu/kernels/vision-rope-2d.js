import { createTensor } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './kernel-execution.js';
import { selectRuleValue } from './rule-registry.js';

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Vision 2D RoPE requires ${label} to be a positive integer.`);
  }
}

export async function runVisionRope2D(input, geometry) {
  const { numTokens, numHeads, headDim, gridHeight, gridWidth, ropeTheta } = geometry;
  for (const [label, value] of Object.entries({
    numTokens,
    numHeads,
    headDim,
    gridHeight,
    gridWidth,
  })) {
    requirePositiveInteger(value, label);
  }
  if (!Number.isFinite(ropeTheta) || ropeTheta <= 0) {
    throw new Error('Vision 2D RoPE requires ropeTheta to be a positive number.');
  }
  if (headDim % 4 !== 0) {
    throw new Error(`Vision 2D RoPE requires headDim divisible by 4, got ${headDim}.`);
  }
  if (numTokens !== gridHeight * gridWidth) {
    throw new Error(
      `Vision 2D RoPE geometry mismatch: numTokens=${numTokens}, grid=${gridWidth}x${gridHeight}.`
    );
  }
  if (!input?.buffer || input.dtype !== 'f32') {
    throw new Error('Vision 2D RoPE requires an f32 input tensor.');
  }
  const expectedElements = numTokens * numHeads * headDim;
  const inputElements = input.shape.reduce((product, value) => product * value, 1);
  if (inputElements !== expectedElements) {
    throw new Error(`Vision 2D RoPE input has ${inputElements} elements, expected ${expectedElements}.`);
  }

  const variant = selectRuleValue('visionRope2d', 'variant', { inputDtype: input.dtype });
  const totalPairs = numTokens * numHeads * (headDim / 2);
  await unifiedKernelWrapper(
    'vision_rope_2d',
    null,
    variant,
    [input],
    {
      num_tokens: numTokens,
      num_heads: numHeads,
      head_dim: headDim,
      grid_height: gridHeight,
      grid_width: gridWidth,
      rope_theta: ropeTheta,
      total_pairs: totalPairs,
      _pad0: 0,
    },
    Math.ceil(totalPairs / WORKGROUP_SIZES.DEFAULT)
  );
  return createTensor(input.buffer, input.dtype, [...input.shape], 'vision_rope_2d_output');
}
