import { createTensor } from '../../../gpu/tensor.js';
import { runVisionSpatialMerge } from '../../../gpu/kernels/vision-spatial-merge.js';
import { releaseBuffer } from '../../../memory/buffer-pool.js';
import { doGelu, doMatmul } from './ops.js';

function getMergerWeight(weights, field, tensorName) {
  const value = weights[field] ?? weights[tensorName];
  if (!value) {
    throw new Error(`Vision spatial merger tensor "${tensorName}" is missing.`);
  }
  return value;
}

export async function spatialMergeProject(params) {
  const {
    input,
    gridHeight,
    gridWidth,
    hiddenSize,
    outHiddenSize,
    spatialMergeSize,
    weights,
  } = params;
  const numPatches = gridHeight * gridWidth;
  const mergedHeight = gridHeight / spatialMergeSize;
  const mergedWidth = gridWidth / spatialMergeSize;
  const mergedCount = mergedHeight * mergedWidth;
  const concatDim = spatialMergeSize * spatialMergeSize * hiddenSize;
  const inputTensor = createTensor(input, 'f32', [numPatches, hiddenSize], 'vision_spatial_merge_input');
  let concatenated = await runVisionSpatialMerge(inputTensor, {
    gridHeight,
    gridWidth,
    hiddenSize,
    mergeSize: spatialMergeSize,
    channelFirst: false,
    inputBlockMajor: false,
  });

  let projected = null;
  let activated = null;
  try {
    projected = await doMatmul(
      concatenated.buffer,
      getMergerWeight(weights, 'mergerMlp0Weight', 'visual.merger.mlp.0.weight'),
      {
        M: mergedCount,
        K: concatDim,
        N: outHiddenSize,
        bias: getMergerWeight(weights, 'mergerMlp0Bias', 'visual.merger.mlp.0.bias'),
      }
    );
    releaseBuffer(concatenated.buffer);
    concatenated = null;

    activated = await doGelu(projected, { count: mergedCount * outHiddenSize });
    releaseBuffer(projected);
    projected = null;

    const output = await doMatmul(
      activated,
      getMergerWeight(weights, 'mergerMlp2Weight', 'visual.merger.mlp.2.weight'),
      {
        M: mergedCount,
        K: outHiddenSize,
        N: outHiddenSize,
        bias: getMergerWeight(weights, 'mergerMlp2Bias', 'visual.merger.mlp.2.bias'),
      }
    );
    releaseBuffer(activated);
    activated = null;
    return output;
  } catch (error) {
    if (activated) releaseBuffer(activated);
    if (projected) releaseBuffer(projected);
    if (concatenated) releaseBuffer(concatenated.buffer);
    throw error;
  }
}
