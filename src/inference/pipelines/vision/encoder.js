

import { log } from '../../../debug/index.js';
import { getDevice, getKernelCapabilities } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import {
  doLayerNorm, doMatmul, doGelu, doResidualAdd,
} from './ops.js';

/**
 * Run the Qwen3-VL vision encoder on preprocessed image patches.
 *
 * Architecture:
 *   patch_embed (conv2d 3->hidden, stride=patchSize) -> [numPatches, hiddenSize]
 *   for each ViT block:
 *     x = layerNorm(x)
 *     x = x + selfAttention(x)    (no KV cache — full prefill attention)
 *     x = layerNorm(x)
 *     x = x + FFN(x)              (gelu activation)
 *   spatialMerge(x) -> [numMergedPatches, outHiddenSize]
 *
 * @param {object} params
 * @param {GPUBuffer}  params.patchBuffer    Preprocessed patches [numPatches, hiddenSize] on GPU
 * @param {number}     params.numPatches     Total number of patches
 * @param {object}     params.visionConfig   Vision config from manifest
 * @param {object}     params.weights        Vision encoder weight buffers keyed by tensor name
 * @param {object}     params.pipelineState  Shared pipeline state for buffer tracking
 * @returns {Promise<{ features: GPUBuffer, numTokens: number }>}
 */
export async function runVisionEncoder(params) {
  const {
    patchBuffer,
    numPatches,
    visionConfig,
    weights,
    pipelineState,
  } = params;

  const {
    depth,
    hiddenSize,
    intermediateSize,
    numHeads,
    outHiddenSize,
    spatialMergeSize,
    eps = 1e-6,
  } = visionConfig;

  const headDim = Math.floor(hiddenSize / numHeads);
  const device = getDevice();

  log.debug('Vision', `encoder: depth=${depth} hidden=${hiddenSize} heads=${numHeads} patches=${numPatches}`);

  let hidden = patchBuffer;

  // Run ViT transformer blocks.
  for (let i = 0; i < depth; i++) {
    const prefix = `visual.blocks.${i}`;

    // Pre-attention layer norm.
    const normed1 = await doLayerNorm(hidden, weights[`${prefix}.norm1.weight`], weights[`${prefix}.norm1.bias`], {
      seqLen: numPatches, hiddenSize, eps,
    });

    // Self-attention (full, no KV cache).
    const attnOut = await visionSelfAttention({
      input: normed1,
      seqLen: numPatches,
      hiddenSize,
      numHeads,
      headDim,
      qkvWeight: weights[`${prefix}.attn.qkv.weight`],
      qkvBias: weights[`${prefix}.attn.qkv.bias`],
      projWeight: weights[`${prefix}.attn.proj.weight`],
      projBias: weights[`${prefix}.attn.proj.bias`],
    });

    releaseBuffer(normed1);

    // Residual add.
    const residual1 = await doResidualAdd(hidden, attnOut, { count: numPatches * hiddenSize });
    releaseBuffer(hidden);
    releaseBuffer(attnOut);

    // Pre-FFN layer norm.
    const normed2 = await doLayerNorm(residual1, weights[`${prefix}.norm2.weight`], weights[`${prefix}.norm2.bias`], {
      seqLen: numPatches, hiddenSize, eps,
    });

    // FFN: linear -> gelu -> linear.
    const ffnOut = await visionFFN({
      input: normed2,
      seqLen: numPatches,
      hiddenSize,
      intermediateSize,
      fc1Weight: weights[`${prefix}.mlp.fc1.weight`],
      fc1Bias: weights[`${prefix}.mlp.fc1.bias`],
      fc2Weight: weights[`${prefix}.mlp.fc2.weight`],
      fc2Bias: weights[`${prefix}.mlp.fc2.bias`],
    });

    releaseBuffer(normed2);

    // Residual add.
    hidden = await doResidualAdd(residual1, ffnOut, { count: numPatches * hiddenSize });
    releaseBuffer(residual1);
    releaseBuffer(ffnOut);

    log.debug('Vision', `block ${i}/${depth} done`);
  }

  // Spatial merge projector: merge 2x2 patches -> outHiddenSize.
  const mergedTokens = Math.floor(numPatches / (spatialMergeSize * spatialMergeSize));
  const merged = await spatialMergeProject({
    input: hidden,
    numPatches,
    hiddenSize,
    outHiddenSize,
    spatialMergeSize,
    weights,
  });

  releaseBuffer(hidden);

  log.debug('Vision', `encoder done: ${numPatches} patches -> ${mergedTokens} tokens (${outHiddenSize}d)`);

  return { features: merged, numTokens: mergedTokens };
}

/**
 * Vision self-attention (full prefill, no KV cache).
 * QKV are fused into one weight matrix [3*hiddenSize, hiddenSize].
 */
async function visionSelfAttention(params) {
  const {
    input, seqLen, hiddenSize, numHeads, headDim,
    qkvWeight, qkvBias, projWeight, projBias,
  } = params;

  // QKV projection: [seqLen, hiddenSize] @ [hiddenSize, 3*hiddenSize] -> [seqLen, 3*hiddenSize]
  const qkv = await doMatmul(input, qkvWeight, {
    M: seqLen, K: hiddenSize, N: 3 * hiddenSize, bias: qkvBias,
  });

  // Split Q, K, V and compute scaled dot-product attention on GPU.
  // This uses the existing attention kernel infrastructure in prefill mode.
  const attnResult = await computeVisionAttention({
    qkv, seqLen, numHeads, headDim, hiddenSize,
  });

  releaseBuffer(qkv);

  // Output projection: [seqLen, hiddenSize] @ [hiddenSize, hiddenSize] -> [seqLen, hiddenSize]
  const output = await doMatmul(attnResult, projWeight, {
    M: seqLen, K: hiddenSize, N: hiddenSize, bias: projBias,
  });

  releaseBuffer(attnResult);

  return output;
}

/**
 * Compute scaled dot-product attention for vision encoder.
 * No KV cache, no causal mask — full bidirectional attention.
 *
 * Input: fused QKV buffer [seqLen, 3*hiddenSize]
 * Output: attention output [seqLen, hiddenSize]
 */
async function computeVisionAttention(params) {
  const { qkv, seqLen, numHeads, headDim, hiddenSize } = params;
  const device = getDevice();
  const scale = 1.0 / Math.sqrt(headDim);

  // For the initial implementation, read QKV back to CPU, compute attention,
  // and upload the result. This will be replaced with a GPU kernel.
  //
  // TODO(perf): Replace with GPU-native vision attention kernel.
  // The text decoder attention kernels assume causal masking and KV cache,
  // which don't apply to the vision encoder's bidirectional full attention.
  const qkvSize = seqLen * 3 * hiddenSize;
  const qkvData = new Float32Array(qkvSize);
  {
    const staging = device.createBuffer({
      size: qkvSize * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(qkv, 0, staging, 0, qkvSize * 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    qkvData.set(new Float32Array(staging.getMappedRange()));
    staging.unmap();
    staging.destroy();
  }

  // Split into Q, K, V: each [numHeads, seqLen, headDim]
  const Q = new Float32Array(numHeads * seqLen * headDim);
  const K = new Float32Array(numHeads * seqLen * headDim);
  const V = new Float32Array(numHeads * seqLen * headDim);

  for (let s = 0; s < seqLen; s++) {
    for (let h = 0; h < numHeads; h++) {
      for (let d = 0; d < headDim; d++) {
        const srcBase = s * 3 * hiddenSize;
        const headOffset = h * headDim + d;
        Q[(h * seqLen + s) * headDim + d] = qkvData[srcBase + headOffset];
        K[(h * seqLen + s) * headDim + d] = qkvData[srcBase + hiddenSize + headOffset];
        V[(h * seqLen + s) * headDim + d] = qkvData[srcBase + 2 * hiddenSize + headOffset];
      }
    }
  }

  // Compute attention: softmax(Q @ K^T / sqrt(d)) @ V per head.
  const output = new Float32Array(seqLen * hiddenSize);

  for (let h = 0; h < numHeads; h++) {
    // Scores: [seqLen, seqLen]
    const scores = new Float32Array(seqLen * seqLen);
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        let dot = 0;
        for (let d = 0; d < headDim; d++) {
          dot += Q[(h * seqLen + i) * headDim + d] * K[(h * seqLen + j) * headDim + d];
        }
        scores[i * seqLen + j] = dot * scale;
      }
    }

    // Softmax per row.
    for (let i = 0; i < seqLen; i++) {
      let maxVal = -Infinity;
      for (let j = 0; j < seqLen; j++) {
        if (scores[i * seqLen + j] > maxVal) maxVal = scores[i * seqLen + j];
      }
      let sumExp = 0;
      for (let j = 0; j < seqLen; j++) {
        scores[i * seqLen + j] = Math.exp(scores[i * seqLen + j] - maxVal);
        sumExp += scores[i * seqLen + j];
      }
      for (let j = 0; j < seqLen; j++) {
        scores[i * seqLen + j] /= sumExp;
      }
    }

    // Weighted sum: [seqLen, headDim]
    for (let i = 0; i < seqLen; i++) {
      for (let d = 0; d < headDim; d++) {
        let val = 0;
        for (let j = 0; j < seqLen; j++) {
          val += scores[i * seqLen + j] * V[(h * seqLen + j) * headDim + d];
        }
        output[i * hiddenSize + h * headDim + d] = val;
      }
    }
  }

  // Upload result to GPU.
  const outBuffer = acquireBuffer(seqLen * hiddenSize * 4, 'vision-attn-output');
  device.queue.writeBuffer(outBuffer, 0, output);

  return outBuffer;
}

/**
 * Vision FFN: fc1 -> gelu -> fc2.
 */
async function visionFFN(params) {
  const {
    input, seqLen, hiddenSize, intermediateSize,
    fc1Weight, fc1Bias, fc2Weight, fc2Bias,
  } = params;

  // fc1: [seqLen, hiddenSize] -> [seqLen, intermediateSize]
  const fc1Out = await doMatmul(input, fc1Weight, {
    M: seqLen, K: hiddenSize, N: intermediateSize, bias: fc1Bias,
  });

  // GELU activation.
  const activated = await doGelu(fc1Out, { count: seqLen * intermediateSize });
  releaseBuffer(fc1Out);

  // fc2: [seqLen, intermediateSize] -> [seqLen, hiddenSize]
  const fc2Out = await doMatmul(activated, fc2Weight, {
    M: seqLen, K: intermediateSize, N: hiddenSize, bias: fc2Bias,
  });
  releaseBuffer(activated);

  return fc2Out;
}

/**
 * Spatial merge projector.
 *
 * Takes [numPatches, hiddenSize] vision features and merges spatialMergeSize x spatialMergeSize
 * adjacent patches into single tokens via concatenation + linear projection.
 *
 * Input:  [numPatches, hiddenSize] where numPatches = gridH * gridW
 * Output: [mergedPatches, outHiddenSize] where mergedPatches = (gridH/m) * (gridW/m), m = spatialMergeSize
 */
async function spatialMergeProject(params) {
  const {
    input, numPatches, hiddenSize, outHiddenSize, spatialMergeSize, weights,
  } = params;

  const device = getDevice();
  const m = spatialMergeSize;
  const concatDim = m * m * hiddenSize;

  // Read vision features back for spatial rearrangement.
  // TODO(perf): GPU kernel for spatial merge gather.
  const inputSize = numPatches * hiddenSize;
  const inputData = new Float32Array(inputSize);
  {
    const staging = device.createBuffer({
      size: inputSize * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(input, 0, staging, 0, inputSize * 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    inputData.set(new Float32Array(staging.getMappedRange()));
    staging.unmap();
    staging.destroy();
  }

  // Assume patches are laid out as [gridH, gridW, hiddenSize].
  // We need gridH and gridW — derive from numPatches assuming square-ish grid.
  // The actual grid dimensions should be passed in; for now infer from sqrt.
  const gridSide = Math.round(Math.sqrt(numPatches));
  const gridH = gridSide;
  const gridW = Math.floor(numPatches / gridH);

  const mergedH = Math.floor(gridH / m);
  const mergedW = Math.floor(gridW / m);
  const mergedCount = mergedH * mergedW;

  // Concatenate m x m patches into single vectors of dimension concatDim.
  const concatenated = new Float32Array(mergedCount * concatDim);
  for (let mh = 0; mh < mergedH; mh++) {
    for (let mw = 0; mw < mergedW; mw++) {
      const outIdx = mh * mergedW + mw;
      let offset = 0;
      for (let dh = 0; dh < m; dh++) {
        for (let dw = 0; dw < m; dw++) {
          const srcH = mh * m + dh;
          const srcW = mw * m + dw;
          const srcIdx = srcH * gridW + srcW;
          for (let d = 0; d < hiddenSize; d++) {
            concatenated[outIdx * concatDim + offset] = inputData[srcIdx * hiddenSize + d];
            offset++;
          }
        }
      }
    }
  }

  // Upload concatenated data.
  const concatBuffer = acquireBuffer(mergedCount * concatDim * 4, 'vision-merge-concat');
  device.queue.writeBuffer(concatBuffer, 0, concatenated);

  // Linear projection: [mergedCount, concatDim] @ [concatDim, outHiddenSize] -> [mergedCount, outHiddenSize]
  const projected = await doMatmul(concatBuffer, weights['visual.merger.mlp.0.weight'], {
    M: mergedCount,
    K: concatDim,
    N: outHiddenSize,
    bias: weights['visual.merger.mlp.0.bias'],
  });

  releaseBuffer(concatBuffer);

  // GELU + second linear layer.
  const activated = await doGelu(projected, { count: mergedCount * outHiddenSize });
  releaseBuffer(projected);

  const output = await doMatmul(activated, weights['visual.merger.mlp.2.weight'], {
    M: mergedCount,
    K: outHiddenSize,
    N: outHiddenSize,
    bias: weights['visual.merger.mlp.2.bias'],
  });
  releaseBuffer(activated);

  return output;
}
