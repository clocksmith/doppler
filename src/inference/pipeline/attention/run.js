/**
 * Attention Run - Immediate GPU submission path
 *
 * Contains runLayerAttentionGPU which executes attention operations
 * with immediate GPU submission (each kernel submits independently).
 *
 * @module inference/pipeline/attention/run
 */

import { isWeightBuffer, getWeightDtype } from '../../../gpu/weight-buffer.js';
import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../gpu/buffer-pool.js';
import {
  runMatmul,
  runRMSNorm,
  runRoPE,
  runAttention,
  castF16ToF32,
  castF32ToF16,
  runMatmulResidualFused,
  shouldUseFusedMatmulResidual,
} from '../../../gpu/kernel-selector.js';
import { createTensor } from '../../../gpu/tensor.js';
import { isKernelDebugEnabled, dumpTokenVector, dumpKVCache, logKernelStep } from '../debug-utils.js';
import { applyLoRA } from '../lora-apply.js';
import { getLoRAModule } from '../lora.js';
import { kernelTrace, traceStep } from '../kernel-trace.js';
import { log, trace } from '../../../debug/index.js';

import {
  shouldDebugLayer,
  markStageLogged,
} from './types.js';

const ATTENTION_DTYPE_LOGGED = new Set();

/**
 * Run attention for a single layer (GPU path).
 *
 * @param {import('../../../gpu/tensor.js').Tensor} input - Input hidden states tensor
 * @param {import('../types.js').LayerWeights | null} layerWeights - Weights for this layer
 * @param {import('./types.js').AttentionConfig} config - Attention configuration
 * @param {import('./types.js').AttentionState} state - Shared state (RoPE freqs, KV cache)
 * @param {boolean} [debug] - Debug mode flag
 * @param {import('./types.js').AttentionDebugFlags} [debugFlags] - Mutable debug flags to prevent repeated logging
 * @param {(weight: GPUBuffer | import('../../../gpu/weight-buffer.js').WeightBuffer | Float32Array | ArrayBuffer | import('../../../gpu/weight-buffer.js').CpuWeightBuffer, label: string) => GPUBuffer | import('../../../gpu/weight-buffer.js').WeightBuffer} [getWeightBuffer]
 * @param {(weight: GPUBuffer | Float32Array | ArrayBuffer | import('../../../gpu/weight-buffer.js').CpuWeightBuffer, label: string) => GPUBuffer} [getNormWeightBuffer]
 * @param {(buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>} [debugCheckBuffer]
 * @param {import('../lora.js').LoRAAdapter | null} [lora]
 * @returns {Promise<import('./types.js').AttentionResult>}
 */
export async function runLayerAttentionGPU(
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
    attnSoftcap = 0,
    queryPreAttnScalar,
    skipInputNorm = false,
  } = config;

  const device = getDevice();

  const wantsF16Output = input.dtype === 'f16';
  const kvCacheDtype = state.kvCache?.kvDtype ?? (wantsF16Output ? 'f16' : 'f32');
  const allowF16Attention = wantsF16Output && kvCacheDtype === 'f16';
  let attentionInput = input;
  let attentionInputTemp = false;
  if (wantsF16Output && !allowF16Attention) {
    attentionInput = await castF16ToF32(input);
    attentionInputTemp = true;
  }

  // Debug: attention input for configured layers
  if (isPrefill && shouldDebugLayer(layerIdx, debugFlags.debugLayers) && !markStageLogged(layerIdx, 'attn_input', debugFlags) && debugCheckBuffer) {
    await debugCheckBuffer(attentionInput.buffer, `L${layerIdx} attention input (GPU)`, numTokens, hiddenSize);
  }

  // Debug logging moved to debug-utils.ts (enable via setDebugConfig)

  if (!layerWeights) {
    // Return zeros if no weights
    const bytesPerElement = wantsF16Output ? 2 : 4;
    const outputBuf = acquireBuffer(numTokens * hiddenSize * bytesPerElement, undefined, 'attn_output');
    const output = createTensor(outputBuf, wantsF16Output ? 'f16' : 'f32', [numTokens, hiddenSize], 'attn_output');
    return { output, residualFused: false };
  }

  const qSize = numTokens * numHeads * headDim;
  const kvSize = numTokens * numKVHeads * headDim;

  // 1. Input norm
  /** @type {import('../../../gpu/tensor.js').Tensor} */
  let normed = attentionInput;
  if (!skipInputNorm && layerWeights.inputNorm && getNormWeightBuffer) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.inputNorm, 'input_norm');

    // Debug: norm weights for configured layers
    if (isPrefill && shouldDebugLayer(layerIdx, debugFlags.debugLayers) && !markStageLogged(layerIdx, 'norm_weights', debugFlags) && debugCheckBuffer) {
      await debugCheckBuffer(normWeightBuf, `L${layerIdx} input norm weights (GPU)`, 1, hiddenSize);
    }

    normed = await runRMSNorm(attentionInput, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      rmsNormWeightOffset: config.rmsNormWeightOffset,
    });
    if (!(layerWeights.inputNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);

    // Trace input norm output
    if (kernelTrace.enabled) {
      await traceStep('rmsnorm', `L${layerIdx}.input_norm`, layerIdx, normed.buffer, [numTokens, hiddenSize]);
    }

    if (isKernelDebugEnabled(layerIdx)) {
      logKernelStep('rmsnorm', { layerIdx, label: 'input_norm', size: numTokens * hiddenSize });
      await dumpTokenVector(normed.buffer, 'input_norm_out', {
        layerIdx,
        tokenIdx: Math.max(0, numTokens - 1),
        rowSize: hiddenSize,
        dtype: normed.dtype,
      });
    }
  }

  // Debug: Check normed input for L0 prefill
  // Debug: normed input for configured layers
  if (isPrefill && shouldDebugLayer(layerIdx, debugFlags.debugLayers) && !markStageLogged(layerIdx, 'attn_normed', debugFlags) && debugCheckBuffer) {
    await debugCheckBuffer(normed.buffer, `L${layerIdx} normed input (GPU)`, numTokens);
  }

  const debugLayers = debugFlags.debugLayers;
  const shouldLogLayer = debugLayers === null ? layerIdx === 0 : shouldDebugLayer(layerIdx, debugLayers);
  if (shouldLogLayer) {
    const phase = isPrefill ? 'prefill' : 'decode';
    const logKey = `L${layerIdx}_${phase}_dtypes`;
    if (!ATTENTION_DTYPE_LOGGED.has(logKey)) {
      ATTENTION_DTYPE_LOGGED.add(logKey);
      trace.attn(layerIdx, `dtypes: activation=${config.activationDtype ?? 'unknown'}, input=${input.dtype}, normed=${normed.dtype}`);
    }
  }

  if (isKernelDebugEnabled(layerIdx)) {
    await dumpTokenVector(normed.buffer, 'attn_in', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: hiddenSize,
      dtype: normed.dtype,
    });
  }

  // 2. Q/K/V projections
  // Use F16 activation outputs when KV cache is F16 (reduces memory bandwidth and avoids F32->F16 cast)
  const useF16Activations = attentionInput.dtype === 'f16';
  /** @type {import('../../../gpu/tensor.js').Tensor} */
  let qTensor;
  /** @type {import('../../../gpu/tensor.js').Tensor} */
  let kTensor;
  /** @type {import('../../../gpu/tensor.js').Tensor} */
  let vTensor;

  // Check for fused QKV path (3->1 matmul optimization)
  const hasLoRA = getLoRAModule(lora, layerIdx, 'q_proj') ||
    getLoRAModule(lora, layerIdx, 'k_proj') ||
    getLoRAModule(lora, layerIdx, 'v_proj');
  const useFusedQKV = layerWeights.qkvProj && layerWeights.qkvSizes && !hasLoRA;

  if (useFusedQKV && layerWeights.qkvProj && layerWeights.qkvSizes) {
    // FUSED PATH: Single matmul for Q/K/V, then split
    const [qSize_, kSize_, vSize_] = layerWeights.qkvSizes;
    const qkvSize = qSize_ + kSize_ + vSize_;

    // One fused matmul instead of 3 separate ones
    const qkvTensor = await runMatmul(normed, layerWeights.qkvProj, numTokens, qkvSize, hiddenSize, {
      transposeB: 'auto',
      role: 'qkv_proj',
      layerIdx,
      outputDtype: useF16Activations ? 'f16' : undefined,
    });

    // Split fused output into Q, K, V (returns Tensors)
    const { runSplitQKV } = await import('../../../gpu/kernels/split_qkv.js');
    const split = await runSplitQKV(qkvTensor, {
      numTokens,
      qSize: qSize_,
      kSize: kSize_,
      vSize: vSize_,
    });
    // Already Tensors from runSplitQKV
    qTensor = split.Q;
    kTensor = split.K;
    vTensor = split.V;

    // Release fused buffer
    releaseBuffer(qkvTensor.buffer);

    if (layerIdx === 0 && isPrefill) {
      trace.attn(layerIdx, `Using fused QKV path: ${qSize_}+${kSize_}+${vSize_}=${qkvSize}`);
    }
  } else {
    // STANDARD PATH: Separate Q/K/V matmuls
    if (layerWeights.qProj && getWeightBuffer) {
      const qProjBuf = getWeightBuffer(layerWeights.qProj, 'q_proj');
      qTensor = await runMatmul(normed, qProjBuf, numTokens, numHeads * headDim, hiddenSize, {
        transposeB: 'auto',
        role: 'q_proj',
        layerIdx,
        outputDtype: useF16Activations ? 'f16' : undefined,
      });
      if (!(layerWeights.qProj instanceof GPUBuffer) && !isWeightBuffer(layerWeights.qProj)) {
        releaseBuffer(isWeightBuffer(qProjBuf) ? qProjBuf.buffer : qProjBuf);
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
        getWeightBuffer
      );
      if (combined.buffer !== qTensor.buffer) {
        releaseBuffer(qTensor.buffer);
        qTensor = combined;
      }
    }

    if (layerWeights.kProj && getWeightBuffer) {
      const kProjBuf = getWeightBuffer(layerWeights.kProj, 'k_proj');
      kTensor = await runMatmul(normed, kProjBuf, numTokens, numKVHeads * headDim, hiddenSize, {
        transposeB: 'auto',
        role: 'k_proj',
        layerIdx,
        outputDtype: useF16Activations ? 'f16' : undefined,
      });
      if (!(layerWeights.kProj instanceof GPUBuffer) && !isWeightBuffer(layerWeights.kProj)) {
        releaseBuffer(isWeightBuffer(kProjBuf) ? kProjBuf.buffer : kProjBuf);
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
        getWeightBuffer
      );
      if (combined.buffer !== kTensor.buffer) {
        releaseBuffer(kTensor.buffer);
        kTensor = combined;
      }
    }

    if (layerWeights.vProj && getWeightBuffer) {
      const vProjBuf = getWeightBuffer(layerWeights.vProj, 'v_proj');

      vTensor = await runMatmul(normed, vProjBuf, numTokens, numKVHeads * headDim, hiddenSize, {
        transposeB: 'auto',
        role: 'v_proj',
        layerIdx,
        outputDtype: useF16Activations ? 'f16' : undefined,
      });
      if (!(layerWeights.vProj instanceof GPUBuffer) && !isWeightBuffer(layerWeights.vProj)) {
        releaseBuffer(isWeightBuffer(vProjBuf) ? vProjBuf.buffer : vProjBuf);
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
        getWeightBuffer
      );
      if (combined.buffer !== vTensor.buffer) {
        releaseBuffer(vTensor.buffer);
        vTensor = combined;
      }
    }
  }

  // Trace Q/K/V projections
  if (kernelTrace.enabled) {
    await traceStep('matmul', `L${layerIdx}.q_proj`, layerIdx, qTensor.buffer, [numTokens, numHeads * headDim]);
    await traceStep('matmul', `L${layerIdx}.k_proj`, layerIdx, kTensor.buffer, [numTokens, numKVHeads * headDim]);
    await traceStep('matmul', `L${layerIdx}.v_proj`, layerIdx, vTensor.buffer, [numTokens, numKVHeads * headDim]);
  }

  // Kernel step debug: Q/K/V projections
  if (isKernelDebugEnabled(layerIdx)) {
    logKernelStep('matmul', { layerIdx, label: 'Q_proj', M: numTokens, N: numHeads * headDim, K: hiddenSize });
    await dumpTokenVector(qTensor.buffer, 'Q_proj', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: numHeads * headDim,
      dtype: qTensor.dtype,
    });
    logKernelStep('matmul', { layerIdx, label: 'K_proj', M: numTokens, N: numKVHeads * headDim, K: hiddenSize });
    await dumpTokenVector(kTensor.buffer, 'K_proj', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: numKVHeads * headDim,
      dtype: kTensor.dtype,
    });
    logKernelStep('matmul', { layerIdx, label: 'V_proj', M: numTokens, N: numKVHeads * headDim, K: hiddenSize });
    await dumpTokenVector(vTensor.buffer, 'V_proj', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: numKVHeads * headDim,
      dtype: vTensor.dtype,
    });
  }

  // Debug: Check Q/K/V after projections for L0 prefill
  // Debug: Q/K/V projections for configured layers
  if (isPrefill && shouldDebugLayer(layerIdx, debugFlags.debugLayers) && !markStageLogged(layerIdx, 'qkv_proj', debugFlags) && debugCheckBuffer) {
    await debugCheckBuffer(qTensor.buffer, `L${layerIdx} Q after proj (GPU)`, numTokens, numHeads * headDim);
    await debugCheckBuffer(kTensor.buffer, `L${layerIdx} K after proj (GPU)`, numTokens, numKVHeads * headDim);
    await debugCheckBuffer(vTensor.buffer, `L${layerIdx} V after proj (GPU)`, numTokens, numKVHeads * headDim);
  }

  // Optional per-head Q/K norm (Gemma-family)
  const wantsQKNorm = config.queryKeyNorm === true;
  const hasQNorm = !!layerWeights.qNorm;
  const hasKNorm = !!layerWeights.kNorm;
  if (isKernelDebugEnabled(layerIdx)) {
    logKernelStep('qk_norm', { layerIdx, label: `hasQ=${hasQNorm} hasK=${hasKNorm} wants=${wantsQKNorm}` });
  }
  if (wantsQKNorm && layerIdx === 0 && (!hasQNorm || !hasKNorm)) {
    log.warn('Attention', `Q/K norm requested but weights missing (hasQ=${hasQNorm}, hasK=${hasKNorm}); skipping QK norm.`);
  }

  // Note: Gemma 3 q_norm and k_norm use Gemma3RMSNorm with (1+weight) formula
  // Same as layer norms - all Gemma 3 norms use (1+weight)
  // See: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py
  if (hasQNorm && getNormWeightBuffer && layerWeights.qNorm) {
    const qNormBuf = getNormWeightBuffer(layerWeights.qNorm, 'q_norm');
    const qElems = qNormBuf.size / 4;
    if (layerIdx === 0 && isPrefill) {
      trace.attn(layerIdx, `Q_NORM: qElems=${qElems}, headDim=${headDim}, match=${qElems === headDim}, bufSize=${qNormBuf.size}`);
    }
    if (qElems === headDim) {
      const qNormedTensor = await runRMSNorm(qTensor, qNormBuf, rmsNormEps, {
        batchSize: numTokens * numHeads,
        hiddenSize: headDim,
        rmsNormWeightOffset: config.rmsNormWeightOffset,
      });
      releaseBuffer(qTensor.buffer);
      qTensor = qNormedTensor;
      if (isKernelDebugEnabled(layerIdx)) {
        await dumpTokenVector(qTensor.buffer, 'Q_norm', {
          layerIdx,
          tokenIdx: Math.max(0, numTokens - 1),
          rowSize: numHeads * headDim,
          dtype: qTensor.dtype,
        });
      }
    }
    if (!(layerWeights.qNorm instanceof GPUBuffer)) releaseBuffer(qNormBuf);
  }

  if (hasKNorm && getNormWeightBuffer && layerWeights.kNorm) {
    const kNormBuf = getNormWeightBuffer(layerWeights.kNorm, 'k_norm');
    const kElems = kNormBuf.size / 4;
    if (kElems === headDim) {
      const kNormedTensor = await runRMSNorm(kTensor, kNormBuf, rmsNormEps, {
        batchSize: numTokens * numKVHeads,
        hiddenSize: headDim,
        rmsNormWeightOffset: config.rmsNormWeightOffset,
      });
      releaseBuffer(kTensor.buffer);
      kTensor = kNormedTensor;
      if (isKernelDebugEnabled(layerIdx)) {
        await dumpTokenVector(kTensor.buffer, 'K_norm', {
          layerIdx,
          tokenIdx: Math.max(0, numTokens - 1),
          rowSize: numKVHeads * headDim,
          dtype: kTensor.dtype,
        });
      }
    }
    if (!(layerWeights.kNorm instanceof GPUBuffer)) releaseBuffer(kNormBuf);
  }

  if (normed !== attentionInput) releaseBuffer(normed.buffer);
  if (attentionInputTemp) releaseBuffer(attentionInput.buffer);

  // 3. RoPE (modifies tensor in-place)

  if (state.ropeFreqsCos && state.ropeFreqsSin) {
    await runRoPE(qTensor, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
      numHeads, headDim, startPos: currentSeqLen,
    });
    await runRoPE(kTensor, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
      numHeads: numKVHeads, headDim, startPos: currentSeqLen,
    });

    // Trace RoPE outputs
    if (kernelTrace.enabled) {
      await traceStep('rope', `L${layerIdx}.q_rope`, layerIdx, qTensor.buffer, [numTokens, numHeads * headDim]);
      await traceStep('rope', `L${layerIdx}.k_rope`, layerIdx, kTensor.buffer, [numTokens, numKVHeads * headDim]);
    }
  }
  if (isKernelDebugEnabled(layerIdx)) {
    logKernelStep('rope', { layerIdx, label: `startPos=${currentSeqLen}` });
    await dumpTokenVector(qTensor.buffer, 'Q_rope', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: numHeads * headDim,
      dtype: qTensor.dtype,
    });
    await dumpTokenVector(kTensor.buffer, 'K_rope', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: numKVHeads * headDim,
      dtype: kTensor.dtype,
    });
  }

  // Debug: Check Q/K after RoPE for L0 prefill
  // Debug: Q/K after RoPE for configured layers
  if (isPrefill && shouldDebugLayer(layerIdx, debugFlags.debugLayers) && !markStageLogged(layerIdx, 'qk_rope', debugFlags) && debugCheckBuffer) {
    await debugCheckBuffer(qTensor.buffer, `L${layerIdx} Q after RoPE (GPU)`, numTokens, numHeads * headDim);
    await debugCheckBuffer(kTensor.buffer, `L${layerIdx} K after RoPE (GPU)`, numTokens, numKVHeads * headDim);
  }

  // 4. Update KV cache (cache stores raw GPUBuffers for memory efficiency)
  /** @type {GPUBuffer} */
  let cachedK;
  /** @type {GPUBuffer} */
  let cachedV;
  let kvLenForAttention = currentSeqLen + numTokens;
  let causalForAttention = true;
  let startPosForMask = currentSeqLen;

  const hasCache = state.kvCache?.hasGPUCache?.();

  if (hasCache) {
    if (state.kvCache.kvDtype === 'f16') {
      // Use tensor dtype to determine if cast is needed
      const kCasted = kTensor.dtype === 'f16' ? kTensor : await castF32ToF16(kTensor);
      const vCasted = vTensor.dtype === 'f16' ? vTensor : await castF32ToF16(vTensor);

      state.kvCache.updateFromGPU(layerIdx, kCasted.buffer, vCasted.buffer, currentSeqLen, numTokens);

      // Only release if we created new buffers
      if (kTensor.dtype !== 'f16') releaseBuffer(kCasted.buffer);
      if (vTensor.dtype !== 'f16') releaseBuffer(vCasted.buffer);
    } else {
      state.kvCache.updateFromGPU(layerIdx, kTensor.buffer, vTensor.buffer, currentSeqLen, numTokens);
    }
    const gpuBuffers = state.kvCache.getGPUBuffers(layerIdx);
    cachedK = gpuBuffers.keysGPU;
    cachedV = gpuBuffers.valuesGPU;
    kvLenForAttention = gpuBuffers.seqLen;

    // Kernel step debug: KV cache state after update
    if (isKernelDebugEnabled(layerIdx)) {
      trace.kv(layerIdx, `KV cache updated: kvLen=${kvLenForAttention}, startPos=${currentSeqLen}, numTokens=${numTokens}`);
      await dumpKVCache(/** @type {import('../../kv-cache.js').KVCache} */(/** @type {unknown} */ (state.kvCache)), layerIdx);
    }
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

  // 5. Attention (uses raw GPUBuffers)
  // query_pre_attn_scalar is used as: scale = scalar^(-0.5) = 1/sqrt(scalar)
  // For Gemma 2 with query_pre_attn_scalar=256: scale = 1/sqrt(256) = 1/16 (standard head_dim scaling)
  const attnScale = queryPreAttnScalar ? 1.0 / Math.sqrt(queryPreAttnScalar) : 1.0 / Math.sqrt(headDim);
  // Debug: log scale on layer 0
  if (layerIdx === 0 && isPrefill) {
    trace.attn(layerIdx, `Attention scale=${attnScale.toFixed(6)}, queryPreAttnScalar=${queryPreAttnScalar ?? 'undefined'}, headDim=${headDim}`);
  }
  // Wrap cached K/V in Tensors (dtype from cache or input tensor)
  const cachedKDtype = state.kvCache?.kvDtype === 'f16' ? /** @type {const} */ ('f16') : kTensor.dtype;
  const cachedVDtype = state.kvCache?.kvDtype === 'f16' ? /** @type {const} */ ('f16') : vTensor.dtype;
  const cachedKTensor = createTensor(cachedK, cachedKDtype, [kvLenForAttention, numKVHeads * headDim], 'cached_K');
  const cachedVTensor = createTensor(cachedV, cachedVDtype, [kvLenForAttention, numKVHeads * headDim], 'cached_V');

  let attnOutput = await runAttention(qTensor, cachedKTensor, cachedVTensor, null, numHeads, headDim, {
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

  // Trace attention output
  if (kernelTrace.enabled) {
    await traceStep('attention', `L${layerIdx}.attention`, layerIdx, attnOutput.buffer, [numTokens, numHeads * headDim]);
  }

  // Kernel step debug: attention output
  if (isKernelDebugEnabled(layerIdx)) {
    logKernelStep('attention', { layerIdx, label: `seqLen=${numTokens} kvLen=${kvLenForAttention}` });
    await dumpTokenVector(attnOutput.buffer, 'attn_out', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: numHeads * headDim,
      dtype: attnOutput.dtype,
    });
  }

  // Debug: attention output for configured layers
  if (isPrefill && shouldDebugLayer(layerIdx, debugFlags.debugLayers) && !markStageLogged(layerIdx, 'attn_out', debugFlags) && debugCheckBuffer) {
    await debugCheckBuffer(attnOutput.buffer, `L${layerIdx} attention output (before o_proj, GPU)`, numTokens, numHeads * headDim);
  }

  // 6. Output projection (with optional fused residual for decode)
  /** @type {import('../../../gpu/tensor.js').Tensor} */
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
      oProjDtype === 'f16';  // GEMV kernel expects f16 weights

    if (canUseFused && residualTensor) {
      // FUSED PATH: o_proj matmul + residual add in one dispatch
      output = await runMatmulResidualFused(attnOutput, oProjBuf, residualTensor, {
        N: hiddenSize,
        K: numHeads * headDim,
      });
      residualFused = true;

      if (layerIdx === 0 && !isPrefill) {
        trace.attn(layerIdx, `Using fused o_proj+residual path`);
      }
    } else {
      // STANDARD PATH: o_proj matmul only (residual will be added by layer.ts)
      output = await runMatmul(attnOutput, oProjBuf, numTokens, hiddenSize, numHeads * headDim, {
        transposeB: 'auto',
        role: 'o_proj',
        layerIdx,
        outputDtype: useF16Activations ? 'f16' : undefined,
      });
    }
    // Release temporary buffer if we created it (original was not already on GPU)
    if (!(layerWeights.oProj instanceof GPUBuffer) && !isWeightBuffer(layerWeights.oProj)) {
      releaseBuffer(isWeightBuffer(oProjBuf) ? oProjBuf.buffer : oProjBuf);
    }

    // Trace output projection
    if (kernelTrace.enabled) {
      await traceStep('matmul', `L${layerIdx}.o_proj${residualFused ? '+residual' : ''}`, layerIdx, output.buffer, [numTokens, hiddenSize]);
    }

    // Kernel step debug: output projection
    if (isKernelDebugEnabled(layerIdx)) {
      logKernelStep('matmul', { layerIdx, label: residualFused ? 'O_proj+residual' : 'O_proj', M: numTokens, N: hiddenSize, K: numHeads * headDim });
      await dumpTokenVector(output.buffer, 'o_proj_out', {
        layerIdx,
        tokenIdx: Math.max(0, numTokens - 1),
        rowSize: hiddenSize,
        dtype: output.dtype,
      });
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
        getWeightBuffer
      );
      if (combined.buffer !== output.buffer) {
        releaseBuffer(output.buffer);
        output = combined;
      }
    }
  }

  // Debug: o_proj output for configured layers
  if (isPrefill && shouldDebugLayer(layerIdx, debugFlags.debugLayers) && !markStageLogged(layerIdx, 'o_proj', debugFlags) && debugCheckBuffer) {
    await debugCheckBuffer(output.buffer, `L${layerIdx} attention output (after o_proj, GPU)`, numTokens, hiddenSize);
  }

  let finalOutput = output;
  /** @type {GPUBuffer[]} */
  const buffersToRelease = [];
  if (output.buffer !== attnOutput.buffer) {
    buffersToRelease.push(attnOutput.buffer);
  }

  if (wantsF16Output && output.dtype !== 'f16') {
    const f16Output = await castF32ToF16(output);
    buffersToRelease.push(output.buffer);
    finalOutput = f16Output;
  }

  // Cleanup
  releaseBuffer(qTensor.buffer);
  releaseBuffer(kTensor.buffer);
  releaseBuffer(vTensor.buffer);
  for (const buffer of buffersToRelease) {
    releaseBuffer(buffer);
  }

  return { output: finalOutput, residualFused };
}
