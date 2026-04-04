/**
 * Gemma 4 Conformer Audio Encoder
 *
 * Architecture:
 *   mel spectrogram -> 2-stage conv subsampling -> linear projection
 *   -> 12 conformer layers (half-FFN + self-attn + depthwise conv1d + half-FFN + norm)
 *   -> output projection -> audio embedding projection
 *
 * All linears use clipped ranges (input_min/max/output_min/max).
 * Self-attention uses relative position encoding with per_dim_scale and logit capping.
 */

import { log } from '../../../debug/index.js';
import { getDevice } from '../../../gpu/device.js';
import { createTensor } from '../../../gpu/tensor.js';
import {
  runMatmul,
  runRMSNorm,
  runResidualAdd,
  runSiLU,
} from '../../../gpu/kernel-selector.js';
import { runConv2D } from '../../../gpu/kernels/conv2d.js';
import { runDepthwiseConv1D } from '../../../gpu/kernels/depthwise_conv1d.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../../memory/buffer-pool.js';
import { runClippableLinear } from '../shared/clipped-linear.js';

function reshapeTensor(tensor, shape, label) {
  return createTensor(tensor.buffer, tensor.dtype, shape, label ?? tensor.label);
}

// ---------------------------------------------------------------------------
// Subsampling: 2-stage Conv2D + LayerNorm + Linear projection
// ---------------------------------------------------------------------------

async function runSubsampling(melFeatures, numFrames, nMels, weights, audioConfig) {
  const hiddenSize = audioConfig.hiddenSize;
  const convChannels = audioConfig.subsamplingConvChannels;

  // Reshape mel [numFrames, nMels] -> [1, numFrames, nMels] for conv2d (treat as single-channel image)
  const inputBuffer = await uploadData(melFeatures, 'audio_mel_input');
  let currentTensor = createTensor(inputBuffer, 'f32', [1, numFrames, nMels], 'audio_mel');

  let currentHeight = numFrames;
  let currentWidth = nMels;
  let currentChannels = 1;

  // Conv layer 0: [1, H, W] -> [convChannels[0], H/2, W/2] with stride=2, kernel=3, pad=1
  const conv0Result = await runConv2D(
    currentTensor,
    weights.subsampleConv0Weight,
    null,
    {
      inChannels: currentChannels,
      outChannels: convChannels[0],
      height: currentHeight,
      width: currentWidth,
      kernelH: 3,
      kernelW: 3,
      stride: 2,
      pad: 1,
    }
  );
  releaseBuffer(currentTensor.buffer);
  currentHeight = Math.floor((currentHeight + 2 - 3) / 2) + 1;
  currentWidth = Math.floor((currentWidth + 2 - 3) / 2) + 1;
  currentChannels = convChannels[0];

  // Norm after conv0
  const norm0Elements = currentChannels * currentHeight * currentWidth;
  const norm0Result = await runRMSNorm(
    reshapeTensor(conv0Result, [norm0Elements, 1], 'audio_conv0_flat'),
    weights.subsampleNorm0Weight,
    audioConfig.rmsNormEps,
    { batchSize: currentHeight * currentWidth, hiddenSize: currentChannels }
  );
  releaseBuffer(conv0Result.buffer);

  // Conv layer 1: [convChannels[0], H/2, W/2] -> [convChannels[1], H/4, W/4]
  const conv1Input = reshapeTensor(norm0Result, [currentChannels, currentHeight, currentWidth], 'audio_conv1_in');
  const conv1Result = await runConv2D(
    conv1Input,
    weights.subsampleConv1Weight,
    null,
    {
      inChannels: currentChannels,
      outChannels: convChannels[1],
      height: currentHeight,
      width: currentWidth,
      kernelH: 3,
      kernelW: 3,
      stride: 2,
      pad: 1,
    }
  );
  releaseBuffer(norm0Result.buffer);
  currentHeight = Math.floor((currentHeight + 2 - 3) / 2) + 1;
  currentWidth = Math.floor((currentWidth + 2 - 3) / 2) + 1;
  currentChannels = convChannels[1];

  // Norm after conv1
  const norm1Result = await runRMSNorm(
    reshapeTensor(conv1Result, [currentHeight * currentWidth, currentChannels], 'audio_conv1_flat'),
    weights.subsampleNorm1Weight,
    audioConfig.rmsNormEps,
    { batchSize: currentHeight * currentWidth, hiddenSize: currentChannels }
  );
  releaseBuffer(conv1Result.buffer);

  // Flatten to [seqLen, flatDim] where seqLen = currentHeight, flatDim = currentChannels * currentWidth
  const seqLen = currentHeight;
  const flatDim = currentChannels * currentWidth;

  // Linear projection: [seqLen, flatDim] -> [seqLen, hiddenSize]
  const projInput = reshapeTensor(norm1Result, [seqLen, flatDim], 'audio_subsample_flat');
  const projected = await runMatmul(
    projInput,
    weights.subsampleInputProjWeight,
    seqLen,
    hiddenSize,
    flatDim,
    { outputDtype: 'f32', transposeB: 'auto' }
  );
  releaseBuffer(norm1Result.buffer);

  return { tensor: projected, seqLen };
}

// ---------------------------------------------------------------------------
// Conformer Half-Step FFN (Macaron style)
// ---------------------------------------------------------------------------

async function runConformerFFN(hiddenTensor, layerWeights, audioConfig, seqLen, prefix) {
  const hiddenSize = audioConfig.hiddenSize;
  const intermediateSize = hiddenSize * 4; // Conformer FFN is typically 4x
  const residualWeight = audioConfig.residualWeight;

  // Pre layer norm
  const normed = await runRMSNorm(
    hiddenTensor,
    layerWeights.preLayerNorm,
    audioConfig.rmsNormEps,
    { batchSize: seqLen, hiddenSize }
  );

  // FFW layer 1: [seqLen, hiddenSize] -> [seqLen, intermediateSize]
  const ffn1 = await runClippableLinear(
    normed,
    layerWeights.ffwLayer1Weight,
    seqLen,
    intermediateSize,
    hiddenSize,
    layerWeights.ffwLayer1Clip,
    `${prefix}_ffw1`
  );
  releaseBuffer(normed.buffer);

  // SiLU activation
  const activated = await runSiLU(ffn1, { size: seqLen * intermediateSize });
  releaseBuffer(ffn1.buffer);

  // FFW layer 2: [seqLen, intermediateSize] -> [seqLen, hiddenSize]
  const ffn2 = await runClippableLinear(
    activated,
    layerWeights.ffwLayer2Weight,
    seqLen,
    hiddenSize,
    intermediateSize,
    layerWeights.ffwLayer2Clip,
    `${prefix}_ffw2`
  );
  releaseBuffer(activated.buffer);

  // Post layer norm
  const postNormed = await runRMSNorm(
    ffn2,
    layerWeights.postLayerNorm,
    audioConfig.rmsNormEps,
    { batchSize: seqLen, hiddenSize }
  );
  releaseBuffer(ffn2.buffer);

  // Scale by residualWeight (0.5 for Macaron half-step)
  if (residualWeight !== 1.0) {
    const data = await readBuffer(postNormed.buffer, seqLen * hiddenSize * 4);
    const f32 = new Float32Array(data instanceof ArrayBuffer ? data : data.buffer);
    for (let i = 0; i < f32.length; i++) {
      f32[i] *= residualWeight;
    }
    const device = getDevice();
    device.queue.writeBuffer(postNormed.buffer, 0, f32);
  }

  // Residual add
  const result = await runResidualAdd(hiddenTensor, postNormed, seqLen * hiddenSize);
  releaseBuffer(postNormed.buffer);

  return result;
}

// ---------------------------------------------------------------------------
// Conformer Self-Attention with Relative Position Encoding
// ---------------------------------------------------------------------------

async function runConformerAttention(hiddenTensor, layerWeights, audioConfig, seqLen) {
  const hiddenSize = audioConfig.hiddenSize;
  const numHeads = audioConfig.numAttentionHeads;
  const headDim = audioConfig.headDim;
  const logitCap = audioConfig.attentionLogitCap;
  const contextLeft = audioConfig.attentionContextLeft;
  const contextRight = audioConfig.attentionContextRight;

  // Pre-attention norm
  const normed = await runRMSNorm(
    hiddenTensor,
    layerWeights.normPreAttn,
    audioConfig.rmsNormEps,
    { batchSize: seqLen, hiddenSize }
  );

  // Q/K/V projections with clipping
  let qTensor = await runClippableLinear(
    normed, layerWeights.qProj, seqLen, hiddenSize, hiddenSize,
    layerWeights.qProjClip, 'audio_q_proj'
  );
  let kTensor = await runClippableLinear(
    normed, layerWeights.kProj, seqLen, hiddenSize, hiddenSize,
    layerWeights.kProjClip, 'audio_k_proj'
  );
  let vTensor = await runClippableLinear(
    normed, layerWeights.vProj, seqLen, hiddenSize, hiddenSize,
    layerWeights.vProjClip, 'audio_v_proj'
  );
  releaseBuffer(normed.buffer);

  // Apply per_dim_scale to queries: Q *= softplus(per_dim_scale)
  // per_dim_scale is [headDim], applied per-head
  const perDimScaleData = await readBuffer(layerWeights.perDimScale, headDim * 4);
  const perDimScaleF32 = new Float32Array(
    perDimScaleData instanceof ArrayBuffer ? perDimScaleData : perDimScaleData.buffer
  );
  const qData = await readBuffer(qTensor.buffer, seqLen * hiddenSize * 4);
  const qF32 = new Float32Array(qData instanceof ArrayBuffer ? qData : qData.buffer);
  for (let s = 0; s < seqLen; s++) {
    for (let h = 0; h < numHeads; h++) {
      for (let d = 0; d < headDim; d++) {
        const idx = s * hiddenSize + h * headDim + d;
        const scale = Math.log(1 + Math.exp(perDimScaleF32[d])); // softplus
        qF32[idx] *= scale;
      }
    }
  }

  // Compute attention scores: [seqLen, numHeads, seqLen]
  // Score = Q @ K^T / sqrt(headDim)
  const scaleFactor = 1.0 / Math.sqrt(headDim);
  const scores = new Float32Array(numHeads * seqLen * seqLen);

  const kData = await readBuffer(kTensor.buffer, seqLen * hiddenSize * 4);
  const kF32 = new Float32Array(kData instanceof ArrayBuffer ? kData : kData.buffer);

  for (let h = 0; h < numHeads; h++) {
    for (let qi = 0; qi < seqLen; qi++) {
      for (let ki = 0; ki < seqLen; ki++) {
        let dot = 0;
        for (let d = 0; d < headDim; d++) {
          dot += qF32[qi * hiddenSize + h * headDim + d] * kF32[ki * hiddenSize + h * headDim + d];
        }
        scores[h * seqLen * seqLen + qi * seqLen + ki] = dot * scaleFactor;
      }
    }
  }

  // Add relative position bias via relative_k_proj
  // This is a simplified approach: compute relative position embeddings and add to scores
  const relKProjData = await readBuffer(layerWeights.relativeKProj, hiddenSize * hiddenSize * 4);
  const relKProjF32 = new Float32Array(
    relKProjData instanceof ArrayBuffer ? relKProjData : relKProjData.buffer
  );

  // Build relative position vectors and project
  for (let h = 0; h < numHeads; h++) {
    for (let qi = 0; qi < seqLen; qi++) {
      for (let ki = 0; ki < seqLen; ki++) {
        const relPos = ki - qi;
        // Compute relative bias from q and relative_k_proj
        let relBias = 0;
        for (let d = 0; d < headDim; d++) {
          const qVal = qF32[qi * hiddenSize + h * headDim + d];
          // Use distance-based relative encoding
          relBias += qVal * relKProjF32[(h * headDim + d) * hiddenSize + (Math.abs(relPos) % hiddenSize)];
        }
        scores[h * seqLen * seqLen + qi * seqLen + ki] += relBias * scaleFactor;
      }
    }
  }

  // Apply logit capping: scores = cap * tanh(scores / cap)
  for (let i = 0; i < scores.length; i++) {
    scores[i] = logitCap * Math.tanh(scores[i] / logitCap);
  }

  // Apply chunked attention mask (contextLeft, contextRight)
  const invalidValue = audioConfig.attentionInvalidLogitsValue;
  for (let h = 0; h < numHeads; h++) {
    for (let qi = 0; qi < seqLen; qi++) {
      for (let ki = 0; ki < seqLen; ki++) {
        const dist = ki - qi;
        if (dist < -contextLeft || dist > contextRight) {
          scores[h * seqLen * seqLen + qi * seqLen + ki] = invalidValue;
        }
      }
    }
  }

  // Softmax per query position per head
  for (let h = 0; h < numHeads; h++) {
    for (let qi = 0; qi < seqLen; qi++) {
      const offset = h * seqLen * seqLen + qi * seqLen;
      let maxVal = -Infinity;
      for (let ki = 0; ki < seqLen; ki++) {
        if (scores[offset + ki] > maxVal) maxVal = scores[offset + ki];
      }
      let sumExp = 0;
      for (let ki = 0; ki < seqLen; ki++) {
        scores[offset + ki] = Math.exp(scores[offset + ki] - maxVal);
        sumExp += scores[offset + ki];
      }
      for (let ki = 0; ki < seqLen; ki++) {
        scores[offset + ki] /= sumExp;
      }
    }
  }

  // Apply attention: output = scores @ V
  const vData = await readBuffer(vTensor.buffer, seqLen * hiddenSize * 4);
  const vF32 = new Float32Array(vData instanceof ArrayBuffer ? vData : vData.buffer);
  const attnOutput = new Float32Array(seqLen * hiddenSize);

  for (let h = 0; h < numHeads; h++) {
    for (let qi = 0; qi < seqLen; qi++) {
      for (let d = 0; d < headDim; d++) {
        let sum = 0;
        for (let ki = 0; ki < seqLen; ki++) {
          sum += scores[h * seqLen * seqLen + qi * seqLen + ki] * vF32[ki * hiddenSize + h * headDim + d];
        }
        attnOutput[qi * hiddenSize + h * headDim + d] = sum;
      }
    }
  }

  releaseBuffer(qTensor.buffer);
  releaseBuffer(kTensor.buffer);
  releaseBuffer(vTensor.buffer);

  // Upload attention output
  const attnBuffer = await uploadData(attnOutput, 'audio_attn_output');
  const attnTensor = createTensor(attnBuffer, 'f32', [seqLen, hiddenSize], 'audio_attn_output');

  // Output projection with clipping
  const projected = await runClippableLinear(
    attnTensor,
    layerWeights.postProj,
    seqLen,
    hiddenSize,
    hiddenSize,
    layerWeights.postProjClip,
    'audio_attn_post'
  );
  releaseBuffer(attnTensor.buffer);

  // Post-attention norm
  const postNormed = await runRMSNorm(
    projected,
    layerWeights.normPostAttn,
    audioConfig.rmsNormEps,
    { batchSize: seqLen, hiddenSize }
  );
  releaseBuffer(projected.buffer);

  // Residual add
  const result = await runResidualAdd(hiddenTensor, postNormed, seqLen * hiddenSize);
  releaseBuffer(postNormed.buffer);

  return result;
}

// ---------------------------------------------------------------------------
// Conformer Convolution Module (LConv1D)
// ---------------------------------------------------------------------------

async function runConformerConvModule(hiddenTensor, layerWeights, audioConfig, seqLen) {
  const hiddenSize = audioConfig.hiddenSize;
  const kernelSize = audioConfig.convKernelSize;

  // Pre layer norm
  const normed = await runRMSNorm(
    hiddenTensor,
    layerWeights.lconvPreLayerNorm,
    audioConfig.rmsNormEps,
    { batchSize: seqLen, hiddenSize }
  );

  // Linear start (pointwise): [seqLen, hiddenSize] -> [seqLen, hiddenSize*2]
  // The GLU split: first half goes through sigmoid gate, second half is the value
  const linearStart = await runClippableLinear(
    normed,
    layerWeights.lconvLinearStartWeight,
    seqLen,
    hiddenSize * 2,
    hiddenSize,
    layerWeights.lconvLinearStartClip,
    'audio_lconv_start'
  );
  releaseBuffer(normed.buffer);

  // GLU: split into gate and value, gate = sigmoid(gate_half), output = gate * value_half
  const startData = await readBuffer(linearStart.buffer, seqLen * hiddenSize * 2 * 4);
  const startF32 = new Float32Array(startData instanceof ArrayBuffer ? startData : startData.buffer);
  const gluOutput = new Float32Array(seqLen * hiddenSize);
  for (let s = 0; s < seqLen; s++) {
    for (let d = 0; d < hiddenSize; d++) {
      const gateVal = startF32[s * hiddenSize * 2 + d];
      const valueVal = startF32[s * hiddenSize * 2 + hiddenSize + d];
      const sigmoid = 1.0 / (1.0 + Math.exp(-gateVal));
      gluOutput[s * hiddenSize + d] = sigmoid * valueVal;
    }
  }
  releaseBuffer(linearStart.buffer);

  // Depthwise Conv1D: [channels=hiddenSize, length=seqLen]
  // Transpose from [seqLen, hiddenSize] to [hiddenSize, seqLen]
  const transposed = new Float32Array(hiddenSize * seqLen);
  for (let s = 0; s < seqLen; s++) {
    for (let d = 0; d < hiddenSize; d++) {
      transposed[d * seqLen + s] = gluOutput[s * hiddenSize + d];
    }
  }
  const transposedBuffer = await uploadData(transposed, 'audio_lconv_transposed');
  const transposedTensor = createTensor(transposedBuffer, 'f32', [hiddenSize, seqLen], 'audio_lconv_transposed');

  const convResult = await runDepthwiseConv1D(
    transposedTensor,
    layerWeights.lconvDepthwiseWeight,
    { channels: hiddenSize, length: seqLen, kernelSize }
  );
  releaseBuffer(transposedTensor.buffer);

  // Transpose back: [hiddenSize, seqLen] -> [seqLen, hiddenSize]
  const convData = await readBuffer(convResult.buffer, hiddenSize * seqLen * 4);
  const convF32 = new Float32Array(convData instanceof ArrayBuffer ? convData : convData.buffer);
  const convTransposed = new Float32Array(seqLen * hiddenSize);
  for (let s = 0; s < seqLen; s++) {
    for (let d = 0; d < hiddenSize; d++) {
      convTransposed[s * hiddenSize + d] = convF32[d * seqLen + s];
    }
  }
  releaseBuffer(convResult.buffer);

  // Conv norm (RMSNorm on the convolution output)
  const convNormBuffer = await uploadData(convTransposed, 'audio_lconv_normed');
  const convNormTensor = createTensor(convNormBuffer, 'f32', [seqLen, hiddenSize], 'audio_lconv_normed');
  const convNormed = await runRMSNorm(
    convNormTensor,
    layerWeights.lconvConvNorm,
    audioConfig.rmsNormEps,
    { batchSize: seqLen, hiddenSize }
  );
  releaseBuffer(convNormTensor.buffer);

  // SiLU activation
  const activated = await runSiLU(convNormed, { size: seqLen * hiddenSize });
  releaseBuffer(convNormed.buffer);

  // Linear end (pointwise): [seqLen, hiddenSize] -> [seqLen, hiddenSize]
  const linearEnd = await runClippableLinear(
    activated,
    layerWeights.lconvLinearEndWeight,
    seqLen,
    hiddenSize,
    hiddenSize,
    layerWeights.lconvLinearEndClip,
    'audio_lconv_end'
  );
  releaseBuffer(activated.buffer);

  // Residual add
  const result = await runResidualAdd(hiddenTensor, linearEnd, seqLen * hiddenSize);
  releaseBuffer(linearEnd.buffer);

  return result;
}

// ---------------------------------------------------------------------------
// Full Conformer Layer
// ---------------------------------------------------------------------------

async function runConformerLayer(hiddenTensor, layerWeights, audioConfig, seqLen, layerIdx) {
  // 1. Half-step FFN1 (Macaron)
  let current = await runConformerFFN(
    hiddenTensor,
    layerWeights.feedForward1,
    audioConfig,
    seqLen,
    `audio_layer${layerIdx}_ff1`
  );

  // 2. Self-attention with relative position encoding
  const afterAttn = await runConformerAttention(
    current,
    layerWeights,
    audioConfig,
    seqLen
  );
  releaseBuffer(current.buffer);
  current = afterAttn;

  // 3. Convolution module
  const afterConv = await runConformerConvModule(
    current,
    layerWeights,
    audioConfig,
    seqLen
  );
  releaseBuffer(current.buffer);
  current = afterConv;

  // 4. Half-step FFN2
  const afterFFN2 = await runConformerFFN(
    current,
    layerWeights.feedForward2,
    audioConfig,
    seqLen,
    `audio_layer${layerIdx}_ff2`
  );
  releaseBuffer(current.buffer);
  current = afterFFN2;

  // 5. Final norm_out
  const hiddenSize = audioConfig.hiddenSize;
  const normed = await runRMSNorm(
    current,
    layerWeights.normOut,
    audioConfig.rmsNormEps,
    { batchSize: seqLen, hiddenSize }
  );
  releaseBuffer(current.buffer);

  return normed;
}

// ---------------------------------------------------------------------------
// Main Encoder Entry Point
// ---------------------------------------------------------------------------

/**
 * Encode audio through the Gemma 4 conformer audio tower.
 *
 * @param {object} params
 * @param {Float32Array} params.melFeatures  Log-mel spectrogram [numFrames * nMels]
 * @param {number}       params.numFrames    Number of mel frames
 * @param {number}       params.nMels        Number of mel bands
 * @param {object}       params.audioConfig  Resolved audio encoder config
 * @param {object}       params.weights      Audio encoder weight buffers
 * @returns {Promise<{ features: GPUBuffer, numTokens: number }>}
 */
export async function encodeGemma4Audio(params) {
  const { melFeatures, numFrames, nMels, audioConfig, weights } = params;

  if (audioConfig.audioArchitecture !== 'gemma4') {
    throw new Error(
      `[Audio] Unsupported audio architecture "${audioConfig.audioArchitecture}". Expected "gemma4".`
    );
  }
  if (!audioConfig.useClippedLinears) {
    throw new Error('[Audio] Gemma 4 audio encoder requires useClippedLinears=true.');
  }

  const hiddenSize = audioConfig.hiddenSize;
  const depth = audioConfig.depth;

  log.debug('Audio', `encodeGemma4Audio: ${numFrames} frames, ${nMels} mels, ${depth} layers`);

  // Step 1: Subsampling — Conv -> Norm -> Linear projection
  const { tensor: subsampledTensor, seqLen } = await runSubsampling(
    melFeatures, numFrames, nMels, weights, audioConfig
  );

  log.debug('Audio', `Subsampling complete: seqLen=${seqLen}, hiddenSize=${hiddenSize}`);

  // Step 2: Conformer layers
  let currentTensor = subsampledTensor;
  for (let i = 0; i < depth; i++) {
    const layerResult = await runConformerLayer(
      currentTensor,
      weights.layers[i],
      audioConfig,
      seqLen,
      i
    );
    if (currentTensor !== subsampledTensor || i > 0) {
      releaseBuffer(currentTensor.buffer);
    }
    currentTensor = layerResult;
  }
  if (depth > 0) {
    // subsampledTensor was released in the loop
  } else {
    releaseBuffer(subsampledTensor.buffer);
  }

  // Step 3: Output projection [seqLen, hiddenSize] -> [seqLen, outputProjDims]
  const outputProjDims = audioConfig.outputProjDims;
  const outputProj = await runMatmul(
    currentTensor,
    weights.outputProjWeight,
    seqLen,
    outputProjDims,
    hiddenSize,
    { outputDtype: 'f32', transposeB: 'auto' }
  );
  releaseBuffer(currentTensor.buffer);

  // Add output projection bias
  if (weights.outputProjBias) {
    const biasData = await readBuffer(weights.outputProjBias, outputProjDims * 4);
    const biasF32 = new Float32Array(biasData instanceof ArrayBuffer ? biasData : biasData.buffer);
    const projData = await readBuffer(outputProj.buffer, seqLen * outputProjDims * 4);
    const projF32 = new Float32Array(projData instanceof ArrayBuffer ? projData : projData.buffer);
    for (let s = 0; s < seqLen; s++) {
      for (let d = 0; d < outputProjDims; d++) {
        projF32[s * outputProjDims + d] += biasF32[d];
      }
    }
    const device = getDevice();
    device.queue.writeBuffer(outputProj.buffer, 0, projF32);
  }

  // Step 4: Audio embedding projection [seqLen, outputProjDims] -> [seqLen, outputProjDims]
  const embedProj = await runMatmul(
    outputProj,
    weights.audioEmbeddingProjWeight,
    seqLen,
    outputProjDims,
    outputProjDims,
    { outputDtype: 'f32', transposeB: 'auto' }
  );
  releaseBuffer(outputProj.buffer);

  log.info('Audio', `Gemma 4 audio encoding complete: ${seqLen} tokens, ${outputProjDims} dims`);

  return {
    features: embedProj.buffer,
    numTokens: seqLen,
  };
}
