import { runRoPEPrecompute } from '../../../gpu/kernels/rope-precompute.js';
import { releaseBuffer } from '../../../memory/buffer-pool.js';

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`[GLM-OCR MRoPE] ${label} must be a positive integer, got ${value}.`);
  }
  return value;
}

export function buildGlmOcrRopePositionPlan(options) {
  const promptLength = requirePositiveInteger(Number(options.promptLength), 'promptLength');
  const capacity = requirePositiveInteger(Number(options.capacity), 'capacity');
  const imageStartOffset = Number(options.imageStartOffset);
  const imageTokenLength = requirePositiveInteger(Number(options.imageTokenLength), 'imageTokenLength');
  const gridHeight = requirePositiveInteger(Number(options.gridHeight), 'gridHeight');
  const gridWidth = requirePositiveInteger(Number(options.gridWidth), 'gridWidth');
  const mergeSize = requirePositiveInteger(Number(options.mergeSize), 'mergeSize');
  if (!Number.isInteger(imageStartOffset) || imageStartOffset < 0) {
    throw new Error('[GLM-OCR MRoPE] imageStartOffset must be a non-negative integer.');
  }
  if (capacity < promptLength) {
    throw new Error(
      `[GLM-OCR MRoPE] capacity=${capacity} must be at least promptLength=${promptLength}.`
    );
  }
  if (imageStartOffset + imageTokenLength > promptLength) {
    throw new Error('[GLM-OCR MRoPE] image token span exceeds the prompt.');
  }
  if (gridHeight % mergeSize !== 0 || gridWidth % mergeSize !== 0) {
    throw new Error(
      `[GLM-OCR MRoPE] grid ${gridWidth}x${gridHeight} must be divisible by mergeSize=${mergeSize}.`
    );
  }
  const mergedHeight = gridHeight / mergeSize;
  const mergedWidth = gridWidth / mergeSize;
  if (imageTokenLength !== mergedHeight * mergedWidth) {
    throw new Error(
      `[GLM-OCR MRoPE] imageTokenLength=${imageTokenLength} does not match ` +
      `merged grid ${mergedWidth}x${mergedHeight}.`
    );
  }

  const temporal = new Int32Array(capacity);
  const height = new Int32Array(capacity);
  const width = new Int32Array(capacity);
  for (let index = 0; index < imageStartOffset; index++) {
    temporal[index] = index;
    height[index] = index;
    width[index] = index;
  }

  let currentPosition = imageStartOffset;
  for (let row = 0; row < mergedHeight; row++) {
    for (let column = 0; column < mergedWidth; column++) {
      const index = imageStartOffset + row * mergedWidth + column;
      temporal[index] = currentPosition;
      height[index] = currentPosition + row;
      width[index] = currentPosition + column;
    }
  }
  currentPosition += Math.max(mergedHeight, mergedWidth);

  const suffixStart = imageStartOffset + imageTokenLength;
  for (let index = suffixStart; index < capacity; index++) {
    const position = currentPosition + index - suffixStart;
    temporal[index] = position;
    height[index] = position;
    width[index] = position;
  }

  const promptMaximumPosition = Math.max(
    temporal[promptLength - 1],
    height[promptLength - 1],
    width[promptLength - 1]
  );
  return {
    temporal,
    height,
    width,
    promptLength,
    capacity,
    ropeDelta: promptMaximumPosition + 1 - promptLength,
  };
}

export async function uploadGlmOcrRopeFrequencies(positionPlan, options) {
  const { cos, sin } = await runRoPEPrecompute({
    maxSeqLen: positionPlan.capacity,
    rotaryDim: options.rotaryDim,
    frequencyBaseDim: options.frequencyBaseDim,
    theta: options.ropeTheta,
    ropeScale: 1,
    positionPlan,
    mropeSection: options.mropeSection,
  });
  return {
    cos,
    sin,
    positionPlan,
    release() {
      releaseBuffer(cos);
      releaseBuffer(sin);
    },
  };
}
