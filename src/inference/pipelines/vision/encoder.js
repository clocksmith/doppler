import { log } from '../../../debug/index.js';
import { releaseBuffer } from '../../../memory/buffer-pool.js';
import { computeVisionAttention } from './attention.js';
import { doGelu, doLayerNorm, doMatmul, doResidualAdd } from './ops.js';
import { spatialMergeProject } from './spatial-merge.js';

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Vision config ${label} must be a positive integer.`);
  }
}

function getLayerWeights(weights, layerIndex) {
  const owned = weights.layers?.[layerIndex];
  if (owned) {
    return owned;
  }
  const prefix = `visual.blocks.${layerIndex}`;
  return {
    norm1Weight: weights[`${prefix}.norm1.weight`],
    norm1Bias: weights[`${prefix}.norm1.bias`] ?? null,
    norm2Weight: weights[`${prefix}.norm2.weight`],
    norm2Bias: weights[`${prefix}.norm2.bias`] ?? null,
    qkvWeight: weights[`${prefix}.attn.qkv.weight`],
    qkvBias: weights[`${prefix}.attn.qkv.bias`] ?? null,
    projWeight: weights[`${prefix}.attn.proj.weight`],
    projBias: weights[`${prefix}.attn.proj.bias`] ?? null,
    fc1Weight: weights[`${prefix}.mlp.fc1.weight`],
    fc1Bias: weights[`${prefix}.mlp.fc1.bias`] ?? null,
    fc2Weight: weights[`${prefix}.mlp.fc2.weight`],
    fc2Bias: weights[`${prefix}.mlp.fc2.bias`] ?? null,
  };
}

function requireLayerWeights(layerWeights, layerIndex) {
  for (const field of [
    'norm1Weight',
    'norm2Weight',
    'qkvWeight',
    'projWeight',
    'fc1Weight',
    'fc2Weight',
  ]) {
    if (!layerWeights[field]) {
      throw new Error(`Vision encoder layer ${layerIndex} is missing ${field}.`);
    }
  }
}

async function visionSelfAttention(params) {
  const {
    input,
    seqLen,
    hiddenSize,
    numHeads,
    headDim,
    layerWeights,
  } = params;
  let qkv = null;
  let attention = null;
  try {
    qkv = await doMatmul(input, layerWeights.qkvWeight, {
      M: seqLen,
      K: hiddenSize,
      N: 3 * hiddenSize,
      bias: layerWeights.qkvBias,
    });
    attention = await computeVisionAttention({
      qkv,
      seqLen,
      numHeads,
      headDim,
      hiddenSize,
    });
    releaseBuffer(qkv);
    qkv = null;

    const output = await doMatmul(attention, layerWeights.projWeight, {
      M: seqLen,
      K: hiddenSize,
      N: hiddenSize,
      bias: layerWeights.projBias,
    });
    releaseBuffer(attention);
    attention = null;
    return output;
  } finally {
    if (attention) releaseBuffer(attention);
    if (qkv) releaseBuffer(qkv);
  }
}

async function visionFfn(params) {
  const { input, seqLen, hiddenSize, intermediateSize, layerWeights } = params;
  let first = null;
  let activated = null;
  try {
    first = await doMatmul(input, layerWeights.fc1Weight, {
      M: seqLen,
      K: hiddenSize,
      N: intermediateSize,
      bias: layerWeights.fc1Bias,
    });
    activated = await doGelu(first, { count: seqLen * intermediateSize });
    releaseBuffer(first);
    first = null;

    const output = await doMatmul(activated, layerWeights.fc2Weight, {
      M: seqLen,
      K: intermediateSize,
      N: hiddenSize,
      bias: layerWeights.fc2Bias,
    });
    releaseBuffer(activated);
    activated = null;
    return output;
  } finally {
    if (activated) releaseBuffer(activated);
    if (first) releaseBuffer(first);
  }
}

async function runVisionBlock(input, params) {
  const {
    layerIndex,
    layerWeights,
    numPatches,
    hiddenSize,
    intermediateSize,
    numHeads,
    headDim,
    eps,
  } = params;
  let normed = null;
  let attention = null;
  let attentionResidual = null;
  let ffn = null;
  try {
    normed = await doLayerNorm(input, layerWeights.norm1Weight, layerWeights.norm1Bias, {
      seqLen: numPatches,
      hiddenSize,
      eps,
    });
    attention = await visionSelfAttention({
      input: normed,
      seqLen: numPatches,
      hiddenSize,
      numHeads,
      headDim,
      layerWeights,
    });
    releaseBuffer(normed);
    normed = null;

    attentionResidual = await doResidualAdd(input, attention, {
      count: numPatches * hiddenSize,
    });
    releaseBuffer(attention);
    attention = null;

    normed = await doLayerNorm(
      attentionResidual,
      layerWeights.norm2Weight,
      layerWeights.norm2Bias,
      { seqLen: numPatches, hiddenSize, eps }
    );
    ffn = await visionFfn({
      input: normed,
      seqLen: numPatches,
      hiddenSize,
      intermediateSize,
      layerWeights,
    });
    releaseBuffer(normed);
    normed = null;

    const output = await doResidualAdd(attentionResidual, ffn, {
      count: numPatches * hiddenSize,
    });
    releaseBuffer(attentionResidual);
    attentionResidual = null;
    releaseBuffer(ffn);
    ffn = null;
    log.debug('Vision', `block ${layerIndex + 1}/${params.depth} done`);
    return output;
  } finally {
    if (ffn) releaseBuffer(ffn);
    if (attentionResidual) releaseBuffer(attentionResidual);
    if (attention) releaseBuffer(attention);
    if (normed) releaseBuffer(normed);
  }
}

export async function runVisionEncoder(params) {
  const {
    patchBuffer,
    numPatches,
    gridHeight,
    gridWidth,
    visionConfig,
    weights,
  } = params;
  const {
    depth,
    hiddenSize,
    intermediateSize,
    numHeads,
    headDim,
    outHiddenSize,
    spatialMergeSize,
    eps,
  } = visionConfig;
  for (const [label, value] of Object.entries({
    depth,
    hiddenSize,
    intermediateSize,
    numHeads,
    headDim,
    outHiddenSize,
    spatialMergeSize,
    numPatches,
    gridHeight,
    gridWidth,
  })) {
    requirePositiveInteger(value, label);
  }
  if (!Number.isFinite(eps) || eps <= 0) {
    throw new Error('Vision config eps must be a positive number.');
  }
  if (hiddenSize !== numHeads * headDim) {
    throw new Error(
      `Vision config geometry mismatch: hiddenSize=${hiddenSize}, ` +
      `numHeads=${numHeads}, headDim=${headDim}.`
    );
  }
  if (numPatches !== gridHeight * gridWidth) {
    throw new Error(
      `Vision patch geometry mismatch: numPatches=${numPatches}, ` +
      `gridHeight=${gridHeight}, gridWidth=${gridWidth}.`
    );
  }
  if (gridHeight % spatialMergeSize !== 0 || gridWidth % spatialMergeSize !== 0) {
    throw new Error(
      `Vision grid ${gridHeight}x${gridWidth} must be divisible by spatialMergeSize=${spatialMergeSize}.`
    );
  }

  log.debug(
    'Vision',
    `encoder: depth=${depth} hidden=${hiddenSize} heads=${numHeads} patches=${numPatches}`
  );
  let hidden = patchBuffer;
  try {
    for (let layerIndex = 0; layerIndex < depth; layerIndex++) {
      const layerWeights = getLayerWeights(weights, layerIndex);
      requireLayerWeights(layerWeights, layerIndex);
      const next = await runVisionBlock(hidden, {
        layerIndex,
        depth,
        layerWeights,
        numPatches,
        hiddenSize,
        intermediateSize,
        numHeads,
        headDim,
        eps,
      });
      releaseBuffer(hidden);
      hidden = next;
    }

    const features = await spatialMergeProject({
      input: hidden,
      gridHeight,
      gridWidth,
      hiddenSize,
      outHiddenSize,
      spatialMergeSize,
      weights,
    });
    releaseBuffer(hidden);
    hidden = null;
    const numTokens = (gridHeight / spatialMergeSize) * (gridWidth / spatialMergeSize);
    log.debug(
      'Vision',
      `encoder done: ${numPatches} patches -> ${numTokens} tokens (${outHiddenSize}d)`
    );
    return { features, numTokens };
  } finally {
    if (hidden) releaseBuffer(hidden);
  }
}
