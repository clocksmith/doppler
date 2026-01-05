/**
 * Attention operations for transformer layers.
 *
 * This module handles:
 * - Q/K/V projections
 * - RoPE position encoding
 * - KV cache management
 * - Multi-head attention computation
 * - Output projection
 *
 * @module inference/pipeline/attention
 */

import type { ParsedModelConfig } from './config.js';
import type { KVCacheInterface, PipelineContext, LayerWeights, MaybeGPUBuffer } from './types.js';
import { type WeightBuffer, isWeightBuffer, getWeightDtype } from '../../gpu/weight-buffer.js';
import { getDevice } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../gpu/buffer-pool.js';
import {
  runMatmul,
  runRMSNorm,
  runRoPE,
  runAttention,
  castF16ToF32,
  castF32ToF16,
  recordMatmul,
  recordRMSNorm,
  recordRoPE,
  recordAttention,
  recordCastF16ToF32,
  recordCastF32ToF16,
  recordSplitQKV,
  runMatmulResidualFused,
  recordMatmulResidualFused,
  shouldUseFusedMatmulResidual,
  CommandRecorder,
} from '../../gpu/kernel-selector.js';
import { Tensor, createTensor, type TensorDtype } from '../../gpu/tensor.js';
import { isKernelDebugEnabled, dumpTokenVector, dumpKVCache, logKernelStep } from './debug-utils.js';
import { applyLoRA } from './lora-apply.js';
import { getLoRAModule, type LoRAAdapter } from './lora.js';
import { kernelTrace, traceStep } from './kernel-trace.js';
import { log, trace } from '../../debug/index.js';

/**
 * Attention configuration for a layer.
 */
export interface AttentionConfig {
  layerIdx: number;
  numTokens: number;
  isPrefill: boolean;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  hiddenSize: number;
  rmsNormEps: number;
  currentSeqLen: number;
  slidingWindow?: number | null;
  layerType?: string;
  /** Residual tensor for fused o_proj + residual add (decode only) */
  residualTensor?: Tensor | null;
  /** Skip input RMSNorm even if weights are present */
  skipInputNorm?: boolean;
  /** Gemma 2 attention softcapping: score = tanh(score / softcap) * softcap. 0 = disabled. */
  attnSoftcap?: number;
  /** Gemma 2 attention scaling: uses head_dim (256) instead of sqrt(head_dim) (16). */
  queryPreAttnScalar?: number;
  /** Apply query/key RMSNorm even when per-head weights are absent. */
  queryKeyNorm?: boolean;
}

/**
 * Attention state passed between operations.
 */
export interface AttentionState {
  ropeFreqsCos: GPUBuffer | null;
  ropeFreqsSin: GPUBuffer | null;
  kvCache: KVCacheInterface;
}

/**
 * Result from attention layer execution.
 */
export interface AttentionResult {
  /** Output tensor after attention + o_proj */
  output: Tensor;
  /** Whether the attention residual was fused into o_proj (layer.ts should skip residual add) */
  residualFused: boolean;
}

/**
 * Debug flags to prevent repeated logging.
 */
export interface AttentionDebugFlags {
  l0NormedDebugDone?: boolean;
  l0QKVDebugDone?: boolean;
  l0RoPEDebugDone?: boolean;
  l0AttnDebugDone?: boolean;
  l0OProjDebugDone?: boolean;
}

const qkNormOnesCache = new Map<number, GPUBuffer>();

function getQKNormOnesBuffer(headDim: number): GPUBuffer {
  const cached = qkNormOnesCache.get(headDim);
  if (cached) return cached;
  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for Q/K norm buffer');
  }
  const data = new Float32Array(headDim);
  data.fill(1);
  const buffer = device.createBuffer({
    label: `qk_norm_ones_${headDim}`,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  qkNormOnesCache.set(headDim, buffer);
  return buffer;
}

/**
 * Run attention for a single layer (GPU path).
 *
 * @param input - Input hidden states tensor
 * @param layerWeights - Weights for this layer
 * @param config - Attention configuration
 * @param state - Shared state (RoPE freqs, KV cache)
 * @param debug - Debug mode flag
 * @param debugFlags - Mutable debug flags to prevent repeated logging
 * @returns Output tensor after attention
 */
export async function runLayerAttentionGPU(
  input: Tensor,
  layerWeights: LayerWeights | null,
  config: AttentionConfig,
  state: AttentionState,
  debug: boolean = false,
  debugFlags: AttentionDebugFlags = {},
  getWeightBuffer?: (weight: GPUBuffer | WeightBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer | WeightBuffer,
  getNormWeightBuffer?: (weight: GPUBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer,
  debugCheckBuffer?: (buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>,
  lora?: LoRAAdapter | null
): Promise<AttentionResult> {
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
  let attentionInput = input;
  let attentionInputTemp = false;
  if (wantsF16Output) {
    attentionInput = await castF16ToF32(input);
    attentionInputTemp = true;
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
  let normed: Tensor = attentionInput;
  if (!skipInputNorm && layerWeights.inputNorm && getNormWeightBuffer) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.inputNorm, 'input_norm');

    normed = await runRMSNorm(attentionInput, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
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
      });
    }
  }

  // Debug: Check normed input for L0 prefill
  if (layerIdx === 0 && isPrefill && !debugFlags.l0NormedDebugDone && debugCheckBuffer) {
    debugFlags.l0NormedDebugDone = true;
    await debugCheckBuffer(normed.buffer, 'L0 normed input (GPU)', numTokens);
  }

  if (isKernelDebugEnabled(layerIdx)) {
    await dumpTokenVector(normed.buffer, 'attn_in', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: hiddenSize,
    });
  }

  // 2. Q/K/V projections
  // Use F16 activation outputs when KV cache is F16 (reduces memory bandwidth and avoids F32->F16 cast)
  const useF16Activations = attentionInput.dtype === 'f16';
  let qTensor: Tensor, kTensor: Tensor, vTensor: Tensor;

  // Check for fused QKV path (3→1 matmul optimization)
  const hasLoRA = getLoRAModule(lora, layerIdx, 'q_proj') ||
                  getLoRAModule(lora, layerIdx, 'k_proj') ||
                  getLoRAModule(lora, layerIdx, 'v_proj');
  const useFusedQKV = layerWeights.qkvProj && layerWeights.qkvSizes && !hasLoRA && !useF16Activations;

  if (useFusedQKV && layerWeights.qkvProj && layerWeights.qkvSizes) {
    // FUSED PATH: Single matmul for Q/K/V, then split
    const [qSize_, kSize_, vSize_] = layerWeights.qkvSizes;
    const qkvSize = qSize_ + kSize_ + vSize_;

    // One fused matmul instead of 3 separate ones
    const qkvTensor = await runMatmul(normed, layerWeights.qkvProj, numTokens, qkvSize, hiddenSize, {
      transposeB: 'auto',
      role: 'qkv_proj',
    });

    // Split fused output into Q, K, V (returns Tensors)
    const { runSplitQKV } = await import('../../gpu/kernels/split_qkv.js');
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
    await dumpTokenVector(qTensor.buffer, 'Q_proj', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numHeads * headDim });
    logKernelStep('matmul', { layerIdx, label: 'K_proj', M: numTokens, N: numKVHeads * headDim, K: hiddenSize });
    await dumpTokenVector(kTensor.buffer, 'K_proj', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numKVHeads * headDim });
    logKernelStep('matmul', { layerIdx, label: 'V_proj', M: numTokens, N: numKVHeads * headDim, K: hiddenSize });
    await dumpTokenVector(vTensor.buffer, 'V_proj', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numKVHeads * headDim });
  }

  // Debug: Check Q/K/V after projections for L0 prefill
  if (layerIdx === 0 && isPrefill && !debugFlags.l0QKVDebugDone && debugCheckBuffer) {
    debugFlags.l0QKVDebugDone = true;
    await debugCheckBuffer(qTensor.buffer, 'L0 Q after proj (GPU)', numTokens, numHeads * headDim);
    await debugCheckBuffer(kTensor.buffer, 'L0 K after proj (GPU)', numTokens, numKVHeads * headDim);
    await debugCheckBuffer(vTensor.buffer, 'L0 V after proj (GPU)', numTokens, numKVHeads * headDim);
  }

  // Optional per-head Q/K norm (Gemma-family)
  const wantsQKNorm = config.queryKeyNorm === true;
  const hasQNorm = !!layerWeights.qNorm;
  const hasKNorm = !!layerWeights.kNorm;
  if (isKernelDebugEnabled(layerIdx)) {
    logKernelStep('qk_norm', { layerIdx, label: `hasQ=${hasQNorm} hasK=${hasKNorm} wants=${wantsQKNorm}` });
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
      });
      releaseBuffer(qTensor.buffer);
      qTensor = qNormedTensor;
      if (isKernelDebugEnabled(layerIdx)) {
        await dumpTokenVector(qTensor.buffer, 'Q_norm', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numHeads * headDim });
      }
    }
    if (!(layerWeights.qNorm instanceof GPUBuffer)) releaseBuffer(qNormBuf);
  } else if (wantsQKNorm) {
    if (layerIdx === 0 && isPrefill) {
      trace.attn(layerIdx, `Q_NORM: using unit weights (headDim=${headDim})`);
    }
    const qNormBuf = getQKNormOnesBuffer(headDim);
    const qNormedTensor = await runRMSNorm(qTensor, qNormBuf, rmsNormEps, {
      batchSize: numTokens * numHeads,
      hiddenSize: headDim,
    });
    releaseBuffer(qTensor.buffer);
    qTensor = qNormedTensor;
    if (isKernelDebugEnabled(layerIdx)) {
      await dumpTokenVector(qTensor.buffer, 'Q_norm', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numHeads * headDim });
    }
  }

  if (hasKNorm && getNormWeightBuffer && layerWeights.kNorm) {
    const kNormBuf = getNormWeightBuffer(layerWeights.kNorm, 'k_norm');
    const kElems = kNormBuf.size / 4;
    if (kElems === headDim) {
      const kNormedTensor = await runRMSNorm(kTensor, kNormBuf, rmsNormEps, {
        batchSize: numTokens * numKVHeads,
        hiddenSize: headDim,
      });
      releaseBuffer(kTensor.buffer);
      kTensor = kNormedTensor;
      if (isKernelDebugEnabled(layerIdx)) {
        await dumpTokenVector(kTensor.buffer, 'K_norm', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numKVHeads * headDim });
      }
    }
    if (!(layerWeights.kNorm instanceof GPUBuffer)) releaseBuffer(kNormBuf);
  } else if (wantsQKNorm) {
    if (layerIdx === 0 && isPrefill) {
      trace.attn(layerIdx, `K_NORM: using unit weights (headDim=${headDim})`);
    }
    const kNormBuf = getQKNormOnesBuffer(headDim);
    const kNormedTensor = await runRMSNorm(kTensor, kNormBuf, rmsNormEps, {
      batchSize: numTokens * numKVHeads,
      hiddenSize: headDim,
    });
    releaseBuffer(kTensor.buffer);
    kTensor = kNormedTensor;
    if (isKernelDebugEnabled(layerIdx)) {
      await dumpTokenVector(kTensor.buffer, 'K_norm', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numKVHeads * headDim });
    }
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
    await dumpTokenVector(qTensor.buffer, 'Q_rope', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numHeads * headDim });
    await dumpTokenVector(kTensor.buffer, 'K_rope', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numKVHeads * headDim });
  }

  // Debug: Check Q/K after RoPE for L0 prefill
  if (layerIdx === 0 && isPrefill && !debugFlags.l0RoPEDebugDone && debugCheckBuffer) {
    debugFlags.l0RoPEDebugDone = true;
    await debugCheckBuffer(qTensor.buffer, 'L0 Q after RoPE (GPU)', numTokens, numHeads * headDim);
    await debugCheckBuffer(kTensor.buffer, 'L0 K after RoPE (GPU)', numTokens, numKVHeads * headDim);
  }

  // 4. Update KV cache (cache stores raw GPUBuffers for memory efficiency)
  let cachedK: GPUBuffer, cachedV: GPUBuffer;
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
      await dumpKVCache(state.kvCache as unknown as import('../kv-cache.js').KVCache, layerIdx);
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
  const cachedKDtype = state.kvCache?.kvDtype === 'f16' ? 'f16' as const : kTensor.dtype;
  const cachedVDtype = state.kvCache?.kvDtype === 'f16' ? 'f16' as const : vTensor.dtype;
  const cachedKTensor = createTensor(cachedK, cachedKDtype, [kvLenForAttention, numKVHeads * headDim], 'cached_K');
  const cachedVTensor = createTensor(cachedV, cachedVDtype, [kvLenForAttention, numKVHeads * headDim], 'cached_V');

  let attnOutput = await runAttention(qTensor, cachedKTensor, cachedVTensor, null, numHeads, headDim, {
    seqLen: numTokens,
    kvLen: kvLenForAttention,
    numKVHeads,
    causal: causalForAttention,
    startPos: startPosForMask,
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
    await dumpTokenVector(attnOutput.buffer, 'attn_out', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numHeads * headDim });
  }

  // Debug: Check attention output for L0 prefill
  if (layerIdx === 0 && isPrefill && !debugFlags.l0AttnDebugDone && debugCheckBuffer) {
    debugFlags.l0AttnDebugDone = true;
    await debugCheckBuffer(attnOutput.buffer, 'L0 attention output (before o_proj, GPU)', numTokens, numHeads * headDim);
  }

  // 6. Output projection (with optional fused residual for decode)
  let output: Tensor;
  let residualFused = false;
  if (layerWeights.oProj && getWeightBuffer) {
    const oProjBuf = getWeightBuffer(layerWeights.oProj, 'o_proj');
    const loraO = getLoRAModule(lora, layerIdx, 'o_proj');

    // Use fused o_proj + residual for decode when possible
    // Note: dtype from WeightBuffer metadata (buffer-dtypes WeakMap removed)
    const oProjDtype = getWeightDtype(oProjBuf);
    const canUseFused = shouldUseFusedMatmulResidual(numTokens) &&
                        residualTensor &&
                        residualTensor.dtype === 'f32' &&
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
      await dumpTokenVector(output.buffer, 'o_proj_out', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: hiddenSize });
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

  // Debug: Check after o_proj for L0 prefill
  if (layerIdx === 0 && isPrefill && !debugFlags.l0OProjDebugDone && debugCheckBuffer) {
    debugFlags.l0OProjDebugDone = true;
    await debugCheckBuffer(output.buffer, 'L0 attention output (after o_proj, GPU)', numTokens, hiddenSize);
  }

  let finalOutput = output;
  const buffersToRelease: GPUBuffer[] = [];
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

/**
 * Record attention for a single layer (batched, no submit).
 *
 * Uses record* kernel variants to batch all GPU operations into a shared
 * command encoder. No GPU submits happen here - submit once at end of forward pass.
 */
export async function recordLayerAttentionGPU(
  recorder: CommandRecorder,
  input: Tensor,
  layerWeights: LayerWeights | null,
  config: AttentionConfig,
  state: AttentionState,
  debug: boolean = false,
  debugFlags: AttentionDebugFlags = {},
  getWeightBuffer?: (weight: GPUBuffer | WeightBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer | WeightBuffer,
  getNormWeightBuffer?: (weight: GPUBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer,
  debugCheckBuffer?: (buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>,
  lora?: LoRAAdapter | null
): Promise<AttentionResult> {
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

  const wantsF16Output = input.dtype === 'f16';
  let attentionInput = input;
  let attentionInputTemp = false;
  if (wantsF16Output) {
    attentionInput = await recordCastF16ToF32(recorder, input);
    attentionInputTemp = true;
  }

  if (!layerWeights) {
    const bytesPerElement = wantsF16Output ? 2 : 4;
    const outputBuf = acquireBuffer(numTokens * hiddenSize * bytesPerElement, undefined, 'attn_output');
    const output = createTensor(outputBuf, wantsF16Output ? 'f16' : 'f32', [numTokens, hiddenSize], 'attn_output');
    return { output, residualFused: false };
  }

  const qSize = numTokens * numHeads * headDim;
  const kvSize = numTokens * numKVHeads * headDim;

  // 1. Input norm
  let normed: Tensor = attentionInput;
  if (!skipInputNorm && layerWeights.inputNorm && getNormWeightBuffer) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.inputNorm, 'input_norm');
    normed = await recordRMSNorm(recorder, attentionInput, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
    });
    if (!(layerWeights.inputNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);
  }

  // 2. Q/K/V projections
  // Use F16 activation outputs when KV cache is F16 (reduces memory bandwidth and avoids F32->F16 cast)
  const useF16Activations = attentionInput.dtype === 'f16';
  let qTensor: Tensor, kTensor: Tensor, vTensor: Tensor;

  // Check for fused QKV path (3->1 matmul optimization)
  const hasLoRA = getLoRAModule(lora, layerIdx, 'q_proj') ||
                  getLoRAModule(lora, layerIdx, 'k_proj') ||
                  getLoRAModule(lora, layerIdx, 'v_proj');
  const useFusedQKV = layerWeights.qkvProj && layerWeights.qkvSizes && !hasLoRA && !useF16Activations;

  if (useFusedQKV && layerWeights.qkvProj && layerWeights.qkvSizes) {
    // FUSED PATH: Single matmul for Q/K/V, then split
    const [qSize_, kSize_, vSize_] = layerWeights.qkvSizes;
    const qkvSizeTotal = qSize_ + kSize_ + vSize_;

    // One fused matmul instead of 3 separate ones
    const qkvTensor = await recordMatmul(recorder, normed, layerWeights.qkvProj, numTokens, qkvSizeTotal, hiddenSize, {
      transposeB: 'auto',
      role: 'qkv_proj',
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
  if (layerWeights.qNorm && getNormWeightBuffer) {
    const qNormBuf = getNormWeightBuffer(layerWeights.qNorm, 'q_norm');
    const qElems = qNormBuf.size / 4;
    if (qElems === headDim) {
      const qNormedTensor = await recordRMSNorm(recorder, qTensor, qNormBuf, rmsNormEps, {
        batchSize: numTokens * numHeads,
        hiddenSize: headDim,
      });
      releaseBuffer(qTensor.buffer);
      qTensor = qNormedTensor;
    }
    if (!(layerWeights.qNorm instanceof GPUBuffer)) releaseBuffer(qNormBuf);
  } else if (wantsQKNorm) {
    const qNormBuf = getQKNormOnesBuffer(headDim);
    const qNormedTensor = await recordRMSNorm(recorder, qTensor, qNormBuf, rmsNormEps, {
      batchSize: numTokens * numHeads,
      hiddenSize: headDim,
    });
    releaseBuffer(qTensor.buffer);
    qTensor = qNormedTensor;
  }

  if (layerWeights.kNorm && getNormWeightBuffer) {
    const kNormBuf = getNormWeightBuffer(layerWeights.kNorm, 'k_norm');
    const kElems = kNormBuf.size / 4;
    if (kElems === headDim) {
      const kNormedTensor = await recordRMSNorm(recorder, kTensor, kNormBuf, rmsNormEps, {
        batchSize: numTokens * numKVHeads,
        hiddenSize: headDim,
      });
      releaseBuffer(kTensor.buffer);
      kTensor = kNormedTensor;
    }
    if (!(layerWeights.kNorm instanceof GPUBuffer)) releaseBuffer(kNormBuf);
  } else if (wantsQKNorm) {
    const kNormBuf = getQKNormOnesBuffer(headDim);
    const kNormedTensor = await recordRMSNorm(recorder, kTensor, kNormBuf, rmsNormEps, {
      batchSize: numTokens * numKVHeads,
      hiddenSize: headDim,
    });
    releaseBuffer(kTensor.buffer);
    kTensor = kNormedTensor;
  }

  if (normed !== attentionInput) releaseBuffer(normed.buffer);
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
  let cachedK: GPUBuffer, cachedV: GPUBuffer;
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
  const cachedKDtype = state.kvCache?.kvDtype === 'f16' ? 'f16' as const : kTensor.dtype;
  const cachedVDtype = state.kvCache?.kvDtype === 'f16' ? 'f16' as const : vTensor.dtype;
  const cachedKTensor = createTensor(cachedK, cachedKDtype, [kvLenForAttention, numKVHeads * headDim], 'cached_K');
  const cachedVTensor = createTensor(cachedV, cachedVDtype, [kvLenForAttention, numKVHeads * headDim], 'cached_V');

  let attnOutput = await recordAttention(recorder, qTensor, cachedKTensor, cachedVTensor, null, numHeads, headDim, {
    seqLen: numTokens,
    kvLen: kvLenForAttention,
    numKVHeads,
    causal: causalForAttention,
    startPos: startPosForMask,
    slidingWindow: effectiveSlidingWindow,
    attnSoftcap,
    scale: attnScale,
  });

  // 6. Output projection (with optional fused residual for decode)
  let output: Tensor;
  let residualFused = false;
  if (layerWeights.oProj && getWeightBuffer) {
    const oProjBuf = getWeightBuffer(layerWeights.oProj, 'o_proj');
    const loraO = getLoRAModule(lora, layerIdx, 'o_proj');

    // Use fused o_proj + residual for decode when possible
    // Note: dtype from WeightBuffer metadata (buffer-dtypes WeakMap removed)
    const oProjDtype = getWeightDtype(oProjBuf);
    const canUseFused = shouldUseFusedMatmulResidual(numTokens) &&
                        residualTensor &&
                        residualTensor.dtype === 'f32' &&
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
        outputDtype: useF16Activations ? 'f16' : undefined,
      });
    }
    // Release temporary buffer if we created it (original was not already on GPU)
    if (!(layerWeights.oProj instanceof GPUBuffer) && !isWeightBuffer(layerWeights.oProj)) {
      releaseBuffer(isWeightBuffer(oProjBuf) ? oProjBuf.buffer : oProjBuf);
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
  const buffersToTrack: GPUBuffer[] = [];
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
