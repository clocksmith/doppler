

import { isWeightBuffer, getWeightDtype } from '../../../gpu/weight-buffer.js';
import { acquireBuffer } from '../../../gpu/buffer-pool.js';
import {
  recordMatmul,
  recordRMSNorm,
  recordRoPE,
  recordAttention,
  recordCastF16ToF32,
  recordCastF32ToF16,
  recordSplitQKV,
  recordMatmulResidualFused,
  shouldUseFusedMatmulResidual,
} from '../../../gpu/kernel-selector.js';
import { createTensor } from '../../../gpu/tensor.js';
import { applyLoRA } from '../lora-apply.js';
import { getLoRAModule } from '../lora.js';
import { log, trace } from '../../../debug/index.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';

import { releaseOrTrack, shouldDebugLayer } from './types.js';

const ATTENTION_DTYPE_LOGGED = new Set();


export async function recordLayerAttentionGPU(
  recorder,
  input,
  layerWeights,
  config,
  state,
  debug = false,
  debugFlags = {},
  getWeightBuffer,
  getNormWeightBuffer,
  debugCheckBuffer,
  lora
) {
  const {
    layerIdx,
    numTokens,
    isPrefill,
    numHeads,
    numKVHeads,
    headDim,
    hiddenSize,
    rmsNormEps,
    currentSeqLen,
    slidingWindow,
    layerType,
    residualTensor,
    attnSoftcap,
    queryPreAttnScalar,
    skipInputNorm = false,
  } = config;

  const wantsF16Output = input.dtype === 'f16';
  const kvCacheFallback = selectRuleValue('inference', 'dtype', 'f16OrF32', { useF16: wantsF16Output });
  const kvCacheDtype = state.kvCache?.kvDtype ?? kvCacheFallback;
  const allowF16Attention = wantsF16Output && kvCacheDtype === 'f16';
  let attentionInput = input;
  let attentionInputTemp = false;
  if (wantsF16Output && !allowF16Attention) {
    attentionInput = await recordCastF16ToF32(recorder, input);
    attentionInputTemp = true;
  }

  if (!layerWeights) {
    const bytesPerElement = wantsF16Output ? 2 : 4;
    const outputBuf = acquireBuffer(numTokens * hiddenSize * bytesPerElement, undefined, 'attn_output');
    const outputDtype = selectRuleValue('inference', 'dtype', 'f16OrF32', { useF16: wantsF16Output });
    const output = createTensor(outputBuf, outputDtype, [numTokens, hiddenSize], 'attn_output');
    return { output, residualFused: false };
  }

  const qSize = numTokens * numHeads * headDim;
  const kvSize = numTokens * numKVHeads * headDim;

  // 1. Input norm
  
  let normed = attentionInput;
  if (!skipInputNorm && layerWeights.inputNorm && getNormWeightBuffer) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.inputNorm, 'input_norm');
    normed = await recordRMSNorm(recorder, attentionInput, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      rmsNormWeightOffset: config.rmsNormWeightOffset,
    });
    if (!(layerWeights.inputNorm instanceof GPUBuffer)) releaseOrTrack(recorder, normWeightBuf);
  }

  const debugLayers = debugFlags.debugLayers;
  const shouldLogLayer = debugLayers === null ? layerIdx === 0 : shouldDebugLayer(layerIdx, debugLayers);
  if (shouldLogLayer) {
    const phase = selectRuleValue('kernels', 'attention', 'phase', { isDecode: !isPrefill });
    const logKey = `L${layerIdx}_${phase}_dtypes`;
    if (!ATTENTION_DTYPE_LOGGED.has(logKey)) {
      ATTENTION_DTYPE_LOGGED.add(logKey);
      trace.attn(layerIdx, `dtypes: activation=${config.activationDtype ?? 'unknown'}, input=${input.dtype}, normed=${normed.dtype}`);
    }
  }

  // 2. Q/K/V projections
  // Use F16 activation outputs when KV cache is F16 (reduces memory bandwidth and avoids F32->F16 cast)
  const useF16Activations = attentionInput.dtype === 'f16';
  const matmulOutputDtype = selectRuleValue('shared', 'dtype', 'f16OrFallbackByFlag', {
    useF16: useF16Activations,
    fallback: undefined,
  });
  
  let qTensor;
  
  let kTensor;
  
  let vTensor;

  // Check for fused QKV path (3->1 matmul optimization)
  const hasLoRA = getLoRAModule(lora, layerIdx, 'q_proj') ||
                  getLoRAModule(lora, layerIdx, 'k_proj') ||
                  getLoRAModule(lora, layerIdx, 'v_proj');
  const useFusedQKV = layerWeights.qkvProj && layerWeights.qkvSizes && !hasLoRA;

  if (useFusedQKV && layerWeights.qkvProj && layerWeights.qkvSizes) {
    // FUSED PATH: Single matmul for Q/K/V, then split
    const [qSize_, kSize_, vSize_] = layerWeights.qkvSizes;
    const qkvSizeTotal = qSize_ + kSize_ + vSize_;

    // One fused matmul instead of 3 separate ones
    const qkvTensor = await recordMatmul(recorder, normed, layerWeights.qkvProj, numTokens, qkvSizeTotal, hiddenSize, {
      transposeB: 'auto',
      role: 'qkv_proj',
      layerIdx,
      outputDtype: matmulOutputDtype,
    });

    // Split fused output into Q, K, V (returns Tensors)
    const split = await recordSplitQKV(recorder, qkvTensor, {
      numTokens,
      qSize: qSize_,
      kSize: kSize_,
      vSize: vSize_,
    });
    // Already Tensors from recordSplitQKV
    qTensor = split.Q;
    kTensor = split.K;
    vTensor = split.V;

    // Track fused buffer for cleanup
    recorder.trackTemporaryBuffer(qkvTensor.buffer);
  } else {
    // STANDARD PATH: Separate Q/K/V matmuls
    if (layerWeights.qProj && getWeightBuffer) {
      const qProjBuf = getWeightBuffer(layerWeights.qProj, 'q_proj');
      qTensor = await recordMatmul(recorder, normed, qProjBuf, numTokens, numHeads * headDim, hiddenSize, {
        transposeB: 'auto',
        role: 'q_proj',
        layerIdx,
        outputDtype: matmulOutputDtype,
      });
      if (!(layerWeights.qProj instanceof GPUBuffer) && !isWeightBuffer(layerWeights.qProj)) {
        releaseOrTrack(recorder, isWeightBuffer(qProjBuf) ? qProjBuf.buffer : qProjBuf);
      }
    } else {
      const qBuf = acquireBuffer(qSize * 4, undefined, 'Q');
      qTensor = createTensor(qBuf, normed.dtype, [numTokens, numHeads * headDim], 'Q');
    }

    const loraQ = getLoRAModule(lora, layerIdx, 'q_proj');
    if (loraQ && getWeightBuffer) {
      const combined = await applyLoRA(
        normed,
        qTensor,
        loraQ,
        { M: numTokens, N: numHeads * headDim, K: hiddenSize },
        getWeightBuffer,
        recorder
      );
      if (combined.buffer !== qTensor.buffer) {
        recorder.trackTemporaryBuffer(qTensor.buffer);
        qTensor = combined;
      }
    }

    if (layerWeights.kProj && getWeightBuffer) {
      const kProjBuf = getWeightBuffer(layerWeights.kProj, 'k_proj');
      kTensor = await recordMatmul(recorder, normed, kProjBuf, numTokens, numKVHeads * headDim, hiddenSize, {
        transposeB: 'auto',
        role: 'k_proj',
        layerIdx,
        outputDtype: matmulOutputDtype,
      });
      if (!(layerWeights.kProj instanceof GPUBuffer) && !isWeightBuffer(layerWeights.kProj)) {
        releaseOrTrack(recorder, isWeightBuffer(kProjBuf) ? kProjBuf.buffer : kProjBuf);
      }
    } else {
      const kBuf = acquireBuffer(kvSize * 4, undefined, 'K');
      kTensor = createTensor(kBuf, normed.dtype, [numTokens, numKVHeads * headDim], 'K');
    }

    const loraK = getLoRAModule(lora, layerIdx, 'k_proj');
    if (loraK && getWeightBuffer) {
      const combined = await applyLoRA(
        normed,
        kTensor,
        loraK,
        { M: numTokens, N: numKVHeads * headDim, K: hiddenSize },
        getWeightBuffer,
        recorder
      );
      if (combined.buffer !== kTensor.buffer) {
        recorder.trackTemporaryBuffer(kTensor.buffer);
        kTensor = combined;
      }
    }

    if (layerWeights.vProj && getWeightBuffer) {
      const vProjBuf = getWeightBuffer(layerWeights.vProj, 'v_proj');
      vTensor = await recordMatmul(recorder, normed, vProjBuf, numTokens, numKVHeads * headDim, hiddenSize, {
        transposeB: 'auto',
        role: 'v_proj',
        layerIdx,
        outputDtype: matmulOutputDtype,
      });
      if (!(layerWeights.vProj instanceof GPUBuffer) && !isWeightBuffer(layerWeights.vProj)) {
        releaseOrTrack(recorder, isWeightBuffer(vProjBuf) ? vProjBuf.buffer : vProjBuf);
      }
    } else {
      const vBuf = acquireBuffer(kvSize * 4, undefined, 'V');
      vTensor = createTensor(vBuf, normed.dtype, [numTokens, numKVHeads * headDim], 'V');
    }

    const loraV = getLoRAModule(lora, layerIdx, 'v_proj');
    if (loraV && getWeightBuffer) {
      const combined = await applyLoRA(
        normed,
        vTensor,
        loraV,
        { M: numTokens, N: numKVHeads * headDim, K: hiddenSize },
        getWeightBuffer,
        recorder
      );
      if (combined.buffer !== vTensor.buffer) {
        recorder.trackTemporaryBuffer(vTensor.buffer);
        vTensor = combined;
      }
    }
  }

  // Optional per-head Q/K norm (Gemma-family)
  // Note: Gemma 3 q_norm and k_norm use Gemma3RMSNorm with (1+weight) formula
  const wantsQKNorm = config.queryKeyNorm === true;
  if (wantsQKNorm && layerIdx === 0 && (!layerWeights.qNorm || !layerWeights.kNorm)) {
    log.warn('Attention', `Q/K norm requested but weights missing (hasQ=${!!layerWeights.qNorm}, hasK=${!!layerWeights.kNorm}); skipping QK norm.`);
  }
  if (layerWeights.qNorm && getNormWeightBuffer) {
    const qNormBuf = getNormWeightBuffer(layerWeights.qNorm, 'q_norm');
    // Handle both F16 (2 bytes) and F32 (4 bytes) norm weights
    const qElemsF32 = qNormBuf.size / 4;
    const qElemsF16 = qNormBuf.size / 2;
    const qElems = qElemsF32 === headDim ? qElemsF32 : qElemsF16;
    if (qElems === headDim) {
      const qNormedTensor = await recordRMSNorm(recorder, qTensor, qNormBuf, rmsNormEps, {
        batchSize: numTokens * numHeads,
        hiddenSize: headDim,
        rmsNormWeightOffset: config.rmsNormWeightOffset,
      });
      releaseOrTrack(recorder, qTensor.buffer);
      qTensor = qNormedTensor;
    }
    if (!(layerWeights.qNorm instanceof GPUBuffer)) releaseOrTrack(recorder, qNormBuf);
  }

  if (layerWeights.kNorm && getNormWeightBuffer) {
    const kNormBuf = getNormWeightBuffer(layerWeights.kNorm, 'k_norm');
    // Handle both F16 (2 bytes) and F32 (4 bytes) norm weights
    const kElemsF32 = kNormBuf.size / 4;
    const kElemsF16 = kNormBuf.size / 2;
    const kElems = kElemsF32 === headDim ? kElemsF32 : kElemsF16;
    if (kElems === headDim) {
      const kNormedTensor = await recordRMSNorm(recorder, kTensor, kNormBuf, rmsNormEps, {
        batchSize: numTokens * numKVHeads,
        hiddenSize: headDim,
        rmsNormWeightOffset: config.rmsNormWeightOffset,
      });
      releaseOrTrack(recorder, kTensor.buffer);
      kTensor = kNormedTensor;
    }
    if (!(layerWeights.kNorm instanceof GPUBuffer)) releaseOrTrack(recorder, kNormBuf);
  }

  if (normed !== attentionInput) releaseOrTrack(recorder, normed.buffer);
  if (attentionInputTemp) recorder.trackTemporaryBuffer(attentionInput.buffer);

  // 3. RoPE (modifies tensor in-place)
  if (state.ropeFreqsCos && state.ropeFreqsSin) {
    await recordRoPE(recorder, qTensor, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
      numHeads, headDim, startPos: currentSeqLen,
    });
    await recordRoPE(recorder, kTensor, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
      numHeads: numKVHeads, headDim, startPos: currentSeqLen,
    });
  }

  // 4. Update KV cache (cache stores raw GPUBuffers for memory efficiency)
  
  let cachedK;
  
  let cachedV;
  let kvLenForAttention = currentSeqLen + numTokens;
  let causalForAttention = true;
  let startPosForMask = currentSeqLen;

  const hasCache = state.kvCache?.hasGPUCache?.();

  if (hasCache) {
    // Use recordUpdateFromGPU to record copy operations to the recorder's encoder
    // This ensures K/V buffers are populated before copying (all ops submitted together)
    const enc = recorder.getEncoder();
    if (state.kvCache.kvDtype === 'f16') {
      // Use tensor dtype to determine if cast is needed
      const kCasted = kTensor.dtype === 'f16' ? kTensor : await recordCastF32ToF16(recorder, kTensor);
      const vCasted = vTensor.dtype === 'f16' ? vTensor : await recordCastF32ToF16(recorder, vTensor);

      state.kvCache.recordUpdateFromGPU(enc, layerIdx, kCasted.buffer, vCasted.buffer, currentSeqLen, numTokens);

      // Track for cleanup after submit (not release!) - only if we created new buffers
      if (kTensor.dtype !== 'f16') recorder.trackTemporaryBuffer(kCasted.buffer);
      if (vTensor.dtype !== 'f16') recorder.trackTemporaryBuffer(vCasted.buffer);
    } else {
      state.kvCache.recordUpdateFromGPU(enc, layerIdx, kTensor.buffer, vTensor.buffer, currentSeqLen, numTokens);
    }
    const gpuBuffers = state.kvCache.getGPUBuffers(layerIdx);
    cachedK = gpuBuffers.keysGPU;
    cachedV = gpuBuffers.valuesGPU;
    kvLenForAttention = gpuBuffers.seqLen;
  } else {
    cachedK = kTensor.buffer;
    cachedV = vTensor.buffer;
    kvLenForAttention = numTokens;
    startPosForMask = 0;
  }

  // Sliding window attention for specific layers
  // The kernel now handles both causal AND sliding window masking together.
  // We no longer need to disable causal masking for sliding layers.
  const isLayerSliding = layerType === 'sliding_attention';
  const effectiveSlidingWindow = isLayerSliding ? slidingWindow : null;

  if (kvLenForAttention <= 0) {
    throw new Error(`Invalid kvLen ${kvLenForAttention} at layer ${layerIdx}`);
  }

  // 5. Attention
  // query_pre_attn_scalar is used as: scale = scalar^(-0.5) = 1/sqrt(scalar)
  // For Gemma 2 with query_pre_attn_scalar=256: scale = 1/sqrt(256) = 1/16 (standard head_dim scaling)
  const attnScale = queryPreAttnScalar ? 1.0 / Math.sqrt(queryPreAttnScalar) : 1.0 / Math.sqrt(headDim);

  // Wrap cached K/V in Tensors (dtype from cache or input tensor)
  const cachedKDtype = selectRuleValue('inference', 'dtype', 'f16OrFallback', {
    kvDtype: state.kvCache?.kvDtype,
    fallback: kTensor.dtype,
  });
  const cachedVDtype = selectRuleValue('inference', 'dtype', 'f16OrFallback', {
    kvDtype: state.kvCache?.kvDtype,
    fallback: vTensor.dtype,
  });
  const cachedKTensor = createTensor(cachedK, cachedKDtype, [kvLenForAttention, numKVHeads * headDim], 'cached_K');
  const cachedVTensor = createTensor(cachedV, cachedVDtype, [kvLenForAttention, numKVHeads * headDim], 'cached_V');

  let attnOutput = await recordAttention(recorder, qTensor, cachedKTensor, cachedVTensor, null, numHeads, headDim, {
    seqLen: numTokens,
    kvLen: kvLenForAttention,
    numKVHeads,
    causal: causalForAttention,
    startPos: startPosForMask,
    layerIdx,
    slidingWindow: effectiveSlidingWindow,
    attnSoftcap,
    scale: attnScale,
  });

  // 6. Output projection (with optional fused residual for decode)
  
  let output;
  let residualFused = false;
  if (layerWeights.oProj && getWeightBuffer) {
    const oProjBuf = getWeightBuffer(layerWeights.oProj, 'o_proj');
    const loraO = getLoRAModule(lora, layerIdx, 'o_proj');

    // Use fused o_proj + residual for decode when possible
    // Note: dtype from WeightBuffer metadata (buffer-dtypes WeakMap removed)
    const oProjDtype = getWeightDtype(oProjBuf);
    const canUseFused = shouldUseFusedMatmulResidual(numTokens) &&
                        residualTensor &&
                        residualTensor.dtype === attnOutput.dtype &&
                        attnOutput.dtype === 'f32' &&
                        !loraO &&
                        oProjDtype === 'f16';

    if (canUseFused && residualTensor) {
      // FUSED PATH: o_proj matmul + residual add in one dispatch
      output = await recordMatmulResidualFused(recorder, attnOutput, oProjBuf, residualTensor, {
        N: hiddenSize,
        K: numHeads * headDim,
      });
      residualFused = true;
    } else {
      // STANDARD PATH: o_proj matmul only
      output = await recordMatmul(recorder, attnOutput, oProjBuf, numTokens, hiddenSize, numHeads * headDim, {
        transposeB: 'auto',
        role: 'o_proj',
        layerIdx,
        outputDtype: matmulOutputDtype,
      });
    }
    // Release temporary buffer if we created it (original was not already on GPU)
    if (!(layerWeights.oProj instanceof GPUBuffer) && !isWeightBuffer(layerWeights.oProj)) {
      releaseOrTrack(recorder, isWeightBuffer(oProjBuf) ? oProjBuf.buffer : oProjBuf);
    }
  } else {
    output = attnOutput;
  }

  // Apply LoRA to output projection if present (only if not using fused path)
  if (!residualFused) {
    const loraO = getLoRAModule(lora, layerIdx, 'o_proj');
    if (loraO && getWeightBuffer) {
      const combined = await applyLoRA(
        attnOutput,
        output,
        loraO,
        { M: numTokens, N: hiddenSize, K: numHeads * headDim },
        getWeightBuffer,
        recorder
      );
      if (combined.buffer !== output.buffer) {
        recorder.trackTemporaryBuffer(output.buffer);
        output = combined;
      }
    }
  }

  let finalOutput = output;
  
  const buffersToTrack = [];
  if (output.buffer !== attnOutput.buffer) {
    buffersToTrack.push(attnOutput.buffer);
  }
  if (wantsF16Output && output.dtype !== 'f16') {
    const f16Output = await recordCastF32ToF16(recorder, output);
    buffersToTrack.push(output.buffer);
    finalOutput = f16Output;
  }

  // Track intermediate buffers for cleanup after submit (not release!)
  // These buffers are used by recorded operations that haven't executed yet.
  // Releasing them back to the pool would allow reuse before the encoder is submitted,
  // causing data corruption (especially for small decode buffers).
  recorder.trackTemporaryBuffer(qTensor.buffer);
  recorder.trackTemporaryBuffer(kTensor.buffer);
  recorder.trackTemporaryBuffer(vTensor.buffer);
  for (const buffer of buffersToTrack) {
    recorder.trackTemporaryBuffer(buffer);
  }

  return { output: finalOutput, residualFused };
}
