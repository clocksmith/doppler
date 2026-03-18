/**
 * Qwen3-VL ViT encoder — CPU-side forward pass.
 *
 * Runs the full vision transformer on preprocessed image patches:
 *   patch_embed → pos_embed → N transformer blocks → merger → visual tokens
 *
 * The encoder produces visual token embeddings that get scattered into the
 * text decoder hidden states at image_pad token positions.
 *
 * For DeepStack models, intermediate ViT hidden states at specific layers
 * are also projected through deepstack_merger_list and injected at
 * corresponding decoder layers.
 */

import { log } from '../debug/index.js';
import { toF32 } from './dequant-cpu.js';

function getWeightF32(tensor, numElements) {
  if (tensor instanceof Float32Array) return tensor;
  if (tensor && typeof tensor === 'object' && tensor.data !== undefined) {
    return toF32(tensor.data, tensor.dtype, numElements);
  }
  return toF32(tensor, 'F32', numElements);
}

function getTensorInfo(tensor) {
  if (tensor instanceof Float32Array) {
    return { data: tensor, dtype: 'F32', numElements: tensor.length };
  }
  if (tensor && typeof tensor === 'object' && tensor.dtype !== undefined) {
    const shape = tensor.shape ?? [];
    const numElements = shape.reduce((a, b) => a * b, 1);
    return { data: tensor.data ?? tensor, dtype: tensor.dtype, numElements };
  }
  if (ArrayBuffer.isView(tensor)) {
    return { data: tensor, dtype: 'F32', numElements: tensor.byteLength / 4 };
  }
  throw new Error('[Vision] Cannot determine tensor info');
}

function matmul(a, b, M, K, N) {
  const out = new Float32Array(M * N);
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += a[m * K + k] * b[k * N + n];
      }
      out[m * N + n] = sum;
    }
  }
  return out;
}

function addBias(a, bias, M, N) {
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      a[m * N + n] += bias[n];
    }
  }
}

function layerNorm(x, weight, bias, numTokens, dim, eps) {
  const out = new Float32Array(numTokens * dim);
  for (let t = 0; t < numTokens; t++) {
    const off = t * dim;
    let mean = 0;
    for (let i = 0; i < dim; i++) mean += x[off + i];
    mean /= dim;
    let variance = 0;
    for (let i = 0; i < dim; i++) {
      const d = x[off + i] - mean;
      variance += d * d;
    }
    variance /= dim;
    const invStd = 1.0 / Math.sqrt(variance + eps);
    for (let i = 0; i < dim; i++) {
      out[off + i] = (x[off + i] - mean) * invStd * weight[i] + (bias ? bias[i] : 0);
    }
  }
  return out;
}

function gelu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    out[i] = 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v * v * v)));
  }
  return out;
}

function softmax(arr, len) {
  let max = -Infinity;
  for (let i = 0; i < len; i++) {
    if (arr[i] > max) max = arr[i];
  }
  let sum = 0;
  for (let i = 0; i < len; i++) {
    arr[i] = Math.exp(arr[i] - max);
    sum += arr[i];
  }
  for (let i = 0; i < len; i++) arr[i] /= sum;
}

function selfAttention(hidden, qkvWeight, qkvBias, projWeight, projBias, numTokens, dim, numHeads) {
  const headDim = dim / numHeads;
  const qkvDim = dim * 3;
  const qkvInfo = getTensorInfo(qkvWeight);
  const qkvW = getWeightF32(qkvWeight, qkvInfo.numElements);
  const qkv = matmul(hidden, qkvW, numTokens, dim, qkvDim);
  if (qkvBias) {
    const biasInfo = getTensorInfo(qkvBias);
    addBias(qkv, getWeightF32(qkvBias, biasInfo.numElements), numTokens, qkvDim);
  }

  const attnOut = new Float32Array(numTokens * dim);
  const scale = 1.0 / Math.sqrt(headDim);

  for (let h = 0; h < numHeads; h++) {
    const qOff = h * headDim;
    const kOff = dim + h * headDim;
    const vOff = dim * 2 + h * headDim;

    const scores = new Float32Array(numTokens * numTokens);
    for (let qi = 0; qi < numTokens; qi++) {
      for (let ki = 0; ki < numTokens; ki++) {
        let dot = 0;
        for (let d = 0; d < headDim; d++) {
          dot += qkv[qi * qkvDim + qOff + d] * qkv[ki * qkvDim + kOff + d];
        }
        scores[qi * numTokens + ki] = dot * scale;
      }
      softmax(scores.subarray(qi * numTokens, qi * numTokens + numTokens), numTokens);
    }

    for (let qi = 0; qi < numTokens; qi++) {
      for (let d = 0; d < headDim; d++) {
        let sum = 0;
        for (let ki = 0; ki < numTokens; ki++) {
          sum += scores[qi * numTokens + ki] * qkv[ki * qkvDim + vOff + d];
        }
        attnOut[qi * dim + qOff + d] = sum;
      }
    }
  }

  const projInfo = getTensorInfo(projWeight);
  const projW = getWeightF32(projWeight, projInfo.numElements);
  const projected = matmul(attnOut, projW, numTokens, dim, dim);
  if (projBias) {
    const biasInfo = getTensorInfo(projBias);
    addBias(projected, getWeightF32(projBias, biasInfo.numElements), numTokens, dim);
  }
  return projected;
}

function ffn(hidden, fc1Weight, fc1Bias, fc2Weight, fc2Bias, numTokens, dim, intermediateDim) {
  const fc1Info = getTensorInfo(fc1Weight);
  const fc1W = getWeightF32(fc1Weight, fc1Info.numElements);
  let h = matmul(hidden, fc1W, numTokens, dim, intermediateDim);
  if (fc1Bias) {
    const biasInfo = getTensorInfo(fc1Bias);
    addBias(h, getWeightF32(fc1Bias, biasInfo.numElements), numTokens, intermediateDim);
  }
  h = gelu(h);

  const fc2Info = getTensorInfo(fc2Weight);
  const fc2W = getWeightF32(fc2Weight, fc2Info.numElements);
  const out = matmul(h, fc2W, numTokens, intermediateDim, dim);
  if (fc2Bias) {
    const biasInfo = getTensorInfo(fc2Bias);
    addBias(out, getWeightF32(fc2Bias, biasInfo.numElements), numTokens, dim);
  }
  return out;
}

function residualAdd(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

function spatialMerge(hidden, numTokens, dim, gridH, gridW, spatialMergeSize, mergerWeights, outDim, eps) {
  const mergeH = Math.floor(gridH / spatialMergeSize);
  const mergeW = Math.floor(gridW / spatialMergeSize);
  const mergedTokens = mergeH * mergeW;
  const mergeInputDim = dim * spatialMergeSize * spatialMergeSize;

  const merged = new Float32Array(mergedTokens * mergeInputDim);
  for (let mh = 0; mh < mergeH; mh++) {
    for (let mw = 0; mw < mergeW; mw++) {
      const outIdx = mh * mergeW + mw;
      let offset = 0;
      for (let sh = 0; sh < spatialMergeSize; sh++) {
        for (let sw = 0; sw < spatialMergeSize; sw++) {
          const srcH = mh * spatialMergeSize + sh;
          const srcW = mw * spatialMergeSize + sw;
          const srcIdx = srcH * gridW + srcW;
          for (let d = 0; d < dim; d++) {
            merged[outIdx * mergeInputDim + offset] = hidden[srcIdx * dim + d];
            offset++;
          }
        }
      }
    }
  }

  let normed = merged;
  if (mergerWeights.normWeight) {
    const normWInfo = getTensorInfo(mergerWeights.normWeight);
    const normW = getWeightF32(mergerWeights.normWeight, normWInfo.numElements);
    const normB = mergerWeights.normBias
      ? getWeightF32(mergerWeights.normBias, getTensorInfo(mergerWeights.normBias).numElements)
      : null;
    normed = layerNorm(merged, normW, normB, mergedTokens, mergeInputDim, eps);
  }

  const fc1Info = getTensorInfo(mergerWeights.fc1Weight);
  const fc1NumElements = fc1Info.numElements;
  const mlpHidden = fc1NumElements / mergeInputDim;
  const fc1W = getWeightF32(mergerWeights.fc1Weight, fc1NumElements);
  let h = matmul(normed, fc1W, mergedTokens, mergeInputDim, mlpHidden);
  if (mergerWeights.fc1Bias) {
    const biasInfo = getTensorInfo(mergerWeights.fc1Bias);
    addBias(h, getWeightF32(mergerWeights.fc1Bias, biasInfo.numElements), mergedTokens, mlpHidden);
  }
  h = gelu(h);

  const fc2Info = getTensorInfo(mergerWeights.fc2Weight);
  const fc2W = getWeightF32(mergerWeights.fc2Weight, fc2Info.numElements);
  const out = matmul(h, fc2W, mergedTokens, mlpHidden, outDim);
  if (mergerWeights.fc2Bias) {
    const biasInfo = getTensorInfo(mergerWeights.fc2Bias);
    addBias(out, getWeightF32(mergerWeights.fc2Bias, biasInfo.numElements), mergedTokens, outDim);
  }

  return { tokens: out, numTokens: mergedTokens };
}

export function encodeVision(patches, visionConfig, weights) {
  const depth = visionConfig.depth ?? 24;
  const hiddenSize = visionConfig.hiddenSize ?? 1024;
  const numHeads = visionConfig.numHeads ?? 16;
  const intermediateSize = visionConfig.intermediateSize ?? 4096;
  const outHiddenSize = visionConfig.outHiddenSize ?? hiddenSize;
  const eps = visionConfig.eps ?? 1e-6;
  const spatialMergeSize = visionConfig.spatialMergeSize ?? 2;
  const deepstackIndexes = visionConfig.deepstackVisualIndexes ?? [];

  const { patches: patchData, gridH, gridW, numPatches, patchDim } = patches;

  log.info('Vision', `Encoding ${numPatches} patches (${gridH}x${gridW}), depth=${depth}, hidden=${hiddenSize}`);

  const patchEmbedInfo = getTensorInfo(weights.patchEmbed.projWeight);
  const patchW = getWeightF32(weights.patchEmbed.projWeight, patchEmbedInfo.numElements);
  let hidden = matmul(patchData, patchW, numPatches, patchDim, hiddenSize);
  if (weights.patchEmbed.projBias) {
    const biasInfo = getTensorInfo(weights.patchEmbed.projBias);
    addBias(hidden, getWeightF32(weights.patchEmbed.projBias, biasInfo.numElements), numPatches, hiddenSize);
  }

  if (weights.posEmbed) {
    const posInfo = getTensorInfo(weights.posEmbed);
    const posData = getWeightF32(weights.posEmbed, posInfo.numElements);
    const usablePositions = Math.min(numPatches, posData.length / hiddenSize);
    for (let i = 0; i < usablePositions * hiddenSize; i++) {
      hidden[i] += posData[i];
    }
  }

  const deepstackHiddenStates = new Map();
  let deepstackIdx = 0;

  for (let l = 0; l < depth; l++) {
    const block = weights.blocks[l];

    const n1wInfo = getTensorInfo(block.norm1Weight);
    const n1w = getWeightF32(block.norm1Weight, n1wInfo.numElements);
    const n1b = block.norm1Bias
      ? getWeightF32(block.norm1Bias, getTensorInfo(block.norm1Bias).numElements)
      : null;
    const normed1 = layerNorm(hidden, n1w, n1b, numPatches, hiddenSize, eps);

    const attnOut = selfAttention(
      normed1, block.qkvWeight, block.qkvBias,
      block.projWeight, block.projBias,
      numPatches, hiddenSize, numHeads
    );
    hidden = residualAdd(hidden, attnOut);

    const n2wInfo = getTensorInfo(block.norm2Weight);
    const n2w = getWeightF32(block.norm2Weight, n2wInfo.numElements);
    const n2b = block.norm2Bias
      ? getWeightF32(block.norm2Bias, getTensorInfo(block.norm2Bias).numElements)
      : null;
    const normed2 = layerNorm(hidden, n2w, n2b, numPatches, hiddenSize, eps);

    const ffnOut = ffn(normed2, block.mlpFc1Weight, block.mlpFc1Bias,
      block.mlpFc2Weight, block.mlpFc2Bias,
      numPatches, hiddenSize, intermediateSize);
    hidden = residualAdd(hidden, ffnOut);

    if (deepstackIdx < deepstackIndexes.length && l === deepstackIndexes[deepstackIdx]) {
      const dsWeights = weights.deepstackMergers[deepstackIdx];
      if (dsWeights) {
        const dsMerged = spatialMerge(
          hidden, numPatches, hiddenSize, gridH, gridW,
          spatialMergeSize, dsWeights, outHiddenSize, eps
        );
        deepstackHiddenStates.set(deepstackIndexes[deepstackIdx], dsMerged.tokens);
        log.info('Vision', `DeepStack layer ${l}: ${dsMerged.numTokens} merged tokens`);
      }
      deepstackIdx++;
    }

    if ((l + 1) % 6 === 0 || l === depth - 1) {
      log.info('Vision', `ViT block ${l + 1}/${depth} done`);
    }
  }

  const finalMerged = spatialMerge(
    hidden, numPatches, hiddenSize, gridH, gridW,
    spatialMergeSize, weights.merger, outHiddenSize, eps
  );

  log.info('Vision', `Final merge: ${finalMerged.numTokens} visual tokens, dim=${outHiddenSize}`);

  return {
    visualTokens: finalMerged.tokens,
    numVisualTokens: finalMerged.numTokens,
    outHiddenSize,
    deepstackHiddenStates,
  };
}
