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
import { getDevice } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../gpu/buffer-pool.js';
import {
  runMatmul,
  runRMSNorm,
  runRoPE,
  runAttention,
  castF32ToF16,
  recordMatmul,
  recordRMSNorm,
  recordRoPE,
  recordAttention,
  recordCastF32ToF16,
  recordSplitQKV,
  runMatmulResidualFused,
  recordMatmulResidualFused,
  shouldUseFusedMatmulResidual,
  CommandRecorder,
} from '../../gpu/kernel-selector.js';
import { isKernelDebugEnabled, dumpTokenVector, dumpKVCache, logKernelStep } from './debug-utils.js';
import { applyLoRA } from './lora-apply.js';
import { getLoRAModule, type LoRAAdapter } from './lora.js';
import { getBufferDtype } from '../../gpu/buffer-dtypes.js';
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
  attentionKernelOverride?: string | null;
  /** Residual buffer for fused o_proj + residual add (decode only) */
  residualBuffer?: GPUBuffer | null;
  /** Gemma 2 attention softcapping: score = tanh(score / softcap) * softcap. 0 = disabled. */
  attnSoftcap?: number;
  /** Gemma 2 attention scaling: uses head_dim (256) instead of sqrt(head_dim) (16). */
  queryPreAttnScalar?: number;
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
  /** Output buffer after attention + o_proj */
  output: GPUBuffer;
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

/**
 * Run attention for a single layer (GPU path).
 *
 * @param inputBuffer - Input hidden states (GPUBuffer)
 * @param layerWeights - Weights for this layer
 * @param config - Attention configuration
 * @param state - Shared state (RoPE freqs, KV cache)
 * @param debug - Debug mode flag
 * @param debugFlags - Mutable debug flags to prevent repeated logging
 * @returns Output buffer after attention
 */
export async function runLayerAttentionGPU(
  inputBuffer: GPUBuffer,
  layerWeights: LayerWeights | null,
  config: AttentionConfig,
  state: AttentionState,
  debug: boolean = false,
  debugFlags: AttentionDebugFlags = {},
  getWeightBuffer?: (weight: GPUBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer,
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
    attentionKernelOverride,
    residualBuffer,
    attnSoftcap = 0,
    queryPreAttnScalar,
  } = config;

  const device = getDevice();

  // Debug logging moved to debug-utils.ts (enable via setDebugConfig)

  if (!layerWeights) {
    // Return zeros if no weights
    const output = acquireBuffer(numTokens * hiddenSize * 4, undefined, 'attn_output');
    return { output, residualFused: false };
  }

  const qSize = numTokens * numHeads * headDim;
  const kvSize = numTokens * numKVHeads * headDim;

  // 1. Input norm
  let normedBuffer = inputBuffer;
  if (layerWeights.inputNorm && getNormWeightBuffer) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.inputNorm, 'input_norm');

    normedBuffer = await runRMSNorm(inputBuffer, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
    });
    if (!(layerWeights.inputNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);

    // Trace input norm output
    if (kernelTrace.enabled) {
      await traceStep('rmsnorm', `L${layerIdx}.input_norm`, layerIdx, normedBuffer, [numTokens, hiddenSize]);
    }

    if (isKernelDebugEnabled(layerIdx)) {
      logKernelStep('rmsnorm', { layerIdx, label: 'input_norm', size: numTokens * hiddenSize });
      await dumpTokenVector(normedBuffer, 'input_norm_out', {
        layerIdx,
        tokenIdx: Math.max(0, numTokens - 1),
        rowSize: hiddenSize,
      });
    }
  }

  // Debug: Check normed input for L0 prefill
  if (layerIdx === 0 && isPrefill && !debugFlags.l0NormedDebugDone && debugCheckBuffer) {
    debugFlags.l0NormedDebugDone = true;
    await debugCheckBuffer(normedBuffer, 'L0 normed input (GPU)', numTokens);
  }

  if (isKernelDebugEnabled(layerIdx)) {
    await dumpTokenVector(normedBuffer, 'attn_in', {
      layerIdx,
      tokenIdx: Math.max(0, numTokens - 1),
      rowSize: hiddenSize,
    });
  }

  // 2. Q/K/V projections
  // Use F16 activation outputs when KV cache is F16 (reduces memory bandwidth and avoids F32->F16 cast)
  const useF16Activations = state.kvCache?.kvDtype === 'f16';
  let Q: GPUBuffer, K: GPUBuffer, V: GPUBuffer;

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
    const qkvOutput = await runMatmul(normedBuffer, layerWeights.qkvProj, numTokens, qkvSize, hiddenSize, {
      transposeB: 'auto',
    });

    // Split fused output into Q, K, V
    const { runSplitQKV } = await import('../../gpu/kernels/split_qkv.js');
    const split = await runSplitQKV(qkvOutput, {
      numTokens,
      qSize: qSize_,
      kSize: kSize_,
      vSize: vSize_,
    });
    Q = split.Q;
    K = split.K;
    V = split.V;

    // Release fused buffer
    releaseBuffer(qkvOutput);

    if (layerIdx === 0 && isPrefill) {
      trace.attn(layerIdx, `Using fused QKV path: ${qSize_}+${kSize_}+${vSize_}=${qkvSize}`);
    }
  } else {
    // STANDARD PATH: Separate Q/K/V matmuls
    if (layerWeights.qProj && getWeightBuffer) {
      const qProjBuf = getWeightBuffer(layerWeights.qProj, 'q_proj');
      Q = await runMatmul(normedBuffer, qProjBuf, numTokens, numHeads * headDim, hiddenSize, {
        transposeB: 'auto',
        outputDtype: useF16Activations ? 'f16' : undefined,
      });
      if (!(layerWeights.qProj instanceof GPUBuffer)) releaseBuffer(qProjBuf);
    } else {
      Q = acquireBuffer(qSize * 4, undefined, 'Q');
    }

    const loraQ = getLoRAModule(lora, layerIdx, 'q_proj');
    if (loraQ && getWeightBuffer) {
      const combined = await applyLoRA(
        normedBuffer,
        Q,
        loraQ,
        { M: numTokens, N: numHeads * headDim, K: hiddenSize },
        getWeightBuffer
      );
      if (combined !== Q) {
        releaseBuffer(Q);
        Q = combined;
      }
    }

    if (layerWeights.kProj && getWeightBuffer) {
      const kProjBuf = getWeightBuffer(layerWeights.kProj, 'k_proj');
      K = await runMatmul(normedBuffer, kProjBuf, numTokens, numKVHeads * headDim, hiddenSize, {
        transposeB: 'auto',
        outputDtype: useF16Activations ? 'f16' : undefined,
      });
      if (!(layerWeights.kProj instanceof GPUBuffer)) releaseBuffer(kProjBuf);
    } else {
      K = acquireBuffer(kvSize * 4, undefined, 'K');
    }

    const loraK = getLoRAModule(lora, layerIdx, 'k_proj');
    if (loraK && getWeightBuffer) {
      const combined = await applyLoRA(
        normedBuffer,
        K,
        loraK,
        { M: numTokens, N: numKVHeads * headDim, K: hiddenSize },
        getWeightBuffer
      );
      if (combined !== K) {
        releaseBuffer(K);
        K = combined;
      }
    }

    if (layerWeights.vProj && getWeightBuffer) {
      const vProjBuf = getWeightBuffer(layerWeights.vProj, 'v_proj');

      V = await runMatmul(normedBuffer, vProjBuf, numTokens, numKVHeads * headDim, hiddenSize, {
        transposeB: 'auto',
        outputDtype: useF16Activations ? 'f16' : undefined,
      });
      if (!(layerWeights.vProj instanceof GPUBuffer)) releaseBuffer(vProjBuf);
    } else {
      V = acquireBuffer(kvSize * 4, undefined, 'V');
    }

    const loraV = getLoRAModule(lora, layerIdx, 'v_proj');
    if (loraV && getWeightBuffer) {
      const combined = await applyLoRA(
        normedBuffer,
        V,
        loraV,
        { M: numTokens, N: numKVHeads * headDim, K: hiddenSize },
        getWeightBuffer
      );
      if (combined !== V) {
        releaseBuffer(V);
        V = combined;
      }
    }
  }

  // Trace Q/K/V projections
  if (kernelTrace.enabled) {
    await traceStep('matmul', `L${layerIdx}.q_proj`, layerIdx, Q, [numTokens, numHeads * headDim]);
    await traceStep('matmul', `L${layerIdx}.k_proj`, layerIdx, K, [numTokens, numKVHeads * headDim]);
    await traceStep('matmul', `L${layerIdx}.v_proj`, layerIdx, V, [numTokens, numKVHeads * headDim]);
  }

  // Kernel step debug: Q/K/V projections
  if (isKernelDebugEnabled(layerIdx)) {
    logKernelStep('matmul', { layerIdx, label: 'Q_proj', M: numTokens, N: numHeads * headDim, K: hiddenSize });
    await dumpTokenVector(Q, 'Q_proj', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numHeads * headDim });
    logKernelStep('matmul', { layerIdx, label: 'K_proj', M: numTokens, N: numKVHeads * headDim, K: hiddenSize });
    await dumpTokenVector(K, 'K_proj', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numKVHeads * headDim });
    logKernelStep('matmul', { layerIdx, label: 'V_proj', M: numTokens, N: numKVHeads * headDim, K: hiddenSize });
    await dumpTokenVector(V, 'V_proj', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numKVHeads * headDim });
  }

  // Debug: Check Q/K/V after projections for L0 prefill
  if (layerIdx === 0 && isPrefill && !debugFlags.l0QKVDebugDone && debugCheckBuffer) {
    debugFlags.l0QKVDebugDone = true;
    await debugCheckBuffer(Q, 'L0 Q after proj (GPU)', numTokens, numHeads * headDim);
    await debugCheckBuffer(K, 'L0 K after proj (GPU)', numTokens, numKVHeads * headDim);
    await debugCheckBuffer(V, 'L0 V after proj (GPU)', numTokens, numKVHeads * headDim);
  }

  // Optional per-head Q/K norm (Gemma-family)
  const hasQNorm = !!layerWeights.qNorm;
  const hasKNorm = !!layerWeights.kNorm;
  if (isKernelDebugEnabled(layerIdx)) {
    logKernelStep('qk_norm', { layerIdx, label: `hasQ=${hasQNorm} hasK=${hasKNorm}` });
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
      const qNormed = await runRMSNorm(Q, qNormBuf, rmsNormEps, {
        batchSize: numTokens * numHeads,
        hiddenSize: headDim,
      });
      releaseBuffer(Q);
      Q = qNormed;
      if (isKernelDebugEnabled(layerIdx)) {
        await dumpTokenVector(Q, 'Q_norm', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numHeads * headDim });
      }
    }
    if (!(layerWeights.qNorm instanceof GPUBuffer)) releaseBuffer(qNormBuf);
  }

  if (hasKNorm && getNormWeightBuffer && layerWeights.kNorm) {
    const kNormBuf = getNormWeightBuffer(layerWeights.kNorm, 'k_norm');
    const kElems = kNormBuf.size / 4;
    if (kElems === headDim) {
      const kNormed = await runRMSNorm(K, kNormBuf, rmsNormEps, {
        batchSize: numTokens * numKVHeads,
        hiddenSize: headDim,
      });
      releaseBuffer(K);
      K = kNormed;
      if (isKernelDebugEnabled(layerIdx)) {
        await dumpTokenVector(K, 'K_norm', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numKVHeads * headDim });
      }
    }
    if (!(layerWeights.kNorm instanceof GPUBuffer)) releaseBuffer(kNormBuf);
  }

  if (normedBuffer !== inputBuffer) releaseBuffer(normedBuffer);

  // 3. RoPE

  if (state.ropeFreqsCos && state.ropeFreqsSin) {
    await runRoPE(Q, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
      numHeads, headDim, startPos: currentSeqLen,
    });
    await runRoPE(K, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
      numHeads: numKVHeads, headDim, startPos: currentSeqLen,
    });

    // Trace RoPE outputs
    if (kernelTrace.enabled) {
      await traceStep('rope', `L${layerIdx}.q_rope`, layerIdx, Q, [numTokens, numHeads * headDim]);
      await traceStep('rope', `L${layerIdx}.k_rope`, layerIdx, K, [numTokens, numKVHeads * headDim]);
    }
  }
  if (isKernelDebugEnabled(layerIdx)) {
    logKernelStep('rope', { layerIdx, label: `startPos=${currentSeqLen}` });
    await dumpTokenVector(Q, 'Q_rope', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numHeads * headDim });
    await dumpTokenVector(K, 'K_rope', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numKVHeads * headDim });
  }

  // Debug: Check Q/K after RoPE for L0 prefill
  if (layerIdx === 0 && isPrefill && !debugFlags.l0RoPEDebugDone && debugCheckBuffer) {
    debugFlags.l0RoPEDebugDone = true;
    await debugCheckBuffer(Q, 'L0 Q after RoPE (GPU)', numTokens, numHeads * headDim);
    await debugCheckBuffer(K, 'L0 K after RoPE (GPU)', numTokens, numKVHeads * headDim);
  }

  // 4. Update KV cache
  let cachedK: GPUBuffer, cachedV: GPUBuffer;
  let kvLenForAttention = currentSeqLen + numTokens;
  let causalForAttention = true;
  let startPosForMask = currentSeqLen;

  const hasCache = state.kvCache?.hasGPUCache?.();

  if (hasCache) {
    if (state.kvCache.kvDtype === 'f16') {
      const kElems = kvSize;
      const kDtype = getBufferDtype(K);
      const vDtype = getBufferDtype(V);

      const kForCache = kDtype === 'f16' ? K : await castF32ToF16(K, kElems);
      const vForCache = vDtype === 'f16' ? V : await castF32ToF16(V, kElems);

      state.kvCache.updateFromGPU(layerIdx, kForCache, vForCache, currentSeqLen, numTokens);

      // Only release if we created new buffers
      if (kDtype !== 'f16') releaseBuffer(kForCache);
      if (vDtype !== 'f16') releaseBuffer(vForCache);
    } else {
      state.kvCache.updateFromGPU(layerIdx, K, V, currentSeqLen, numTokens);
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
    cachedK = K;
    cachedV = V;
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
  // Debug: log scale on layer 0
  if (layerIdx === 0 && isPrefill) {
    trace.attn(layerIdx, `Attention scale=${attnScale.toFixed(6)}, queryPreAttnScalar=${queryPreAttnScalar ?? 'undefined'}, headDim=${headDim}`);
  }
  const attnOutput = await runAttention(Q, cachedK, cachedV, null, numHeads, headDim, {
    seqLen: numTokens,
    kvLen: kvLenForAttention,
    numKVHeads,
    causal: causalForAttention,
    startPos: startPosForMask,
    attentionKernel: attentionKernelOverride || undefined,
    slidingWindow: effectiveSlidingWindow,
    attnSoftcap,
    scale: attnScale,
  });

  // Trace attention output
  if (kernelTrace.enabled) {
    await traceStep('attention', `L${layerIdx}.attention`, layerIdx, attnOutput, [numTokens, numHeads * headDim]);
  }

  // Kernel step debug: attention output
  if (isKernelDebugEnabled(layerIdx)) {
    logKernelStep('attention', { layerIdx, label: `seqLen=${numTokens} kvLen=${kvLenForAttention}` });
    await dumpTokenVector(attnOutput, 'attn_out', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: numHeads * headDim });
  }

  // Debug: Check attention output for L0 prefill
  if (layerIdx === 0 && isPrefill && !debugFlags.l0AttnDebugDone && debugCheckBuffer) {
    debugFlags.l0AttnDebugDone = true;
    await debugCheckBuffer(attnOutput, 'L0 attention output (before o_proj, GPU)', numTokens, numHeads * headDim);
  }

  // 6. Output projection (with optional fused residual for decode)
  let output: GPUBuffer;
  let residualFused = false;
  if (layerWeights.oProj && getWeightBuffer) {
    const oProjBuf = getWeightBuffer(layerWeights.oProj, 'o_proj');
    const loraO = getLoRAModule(lora, layerIdx, 'o_proj');

    // Use fused o_proj + residual for decode when possible
    const canUseFused = shouldUseFusedMatmulResidual(numTokens) &&
                        residualBuffer &&
                        !loraO &&
                        getBufferDtype(oProjBuf) === 'f16';  // GEMV kernel expects f16 weights

    if (canUseFused && residualBuffer) {
      // FUSED PATH: o_proj matmul + residual add in one dispatch
      output = await runMatmulResidualFused(attnOutput, oProjBuf, residualBuffer, {
        N: hiddenSize,
        K: numHeads * headDim,
        residual: residualBuffer,
      });
      residualFused = true;

      if (layerIdx === 0 && !isPrefill) {
        trace.attn(layerIdx, `Using fused o_proj+residual path`);
      }
    } else {
      // STANDARD PATH: o_proj matmul only (residual will be added by layer.ts)
      output = await runMatmul(attnOutput, oProjBuf, numTokens, hiddenSize, numHeads * headDim, { transposeB: 'auto' });
    }
    if (!(layerWeights.oProj instanceof GPUBuffer)) releaseBuffer(oProjBuf);

    // Trace output projection
    if (kernelTrace.enabled) {
      await traceStep('matmul', `L${layerIdx}.o_proj${residualFused ? '+residual' : ''}`, layerIdx, output, [numTokens, hiddenSize]);
    }

    // Kernel step debug: output projection
    if (isKernelDebugEnabled(layerIdx)) {
      logKernelStep('matmul', { layerIdx, label: residualFused ? 'O_proj+residual' : 'O_proj', M: numTokens, N: hiddenSize, K: numHeads * headDim });
      await dumpTokenVector(output, 'o_proj_out', { layerIdx, tokenIdx: Math.max(0, numTokens - 1), rowSize: hiddenSize });
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
      if (combined !== output) {
        releaseBuffer(output);
        output = combined;
      }
    }
  }

  // Debug: Check after o_proj for L0 prefill
  if (layerIdx === 0 && isPrefill && !debugFlags.l0OProjDebugDone && debugCheckBuffer) {
    debugFlags.l0OProjDebugDone = true;
    await debugCheckBuffer(output, 'L0 attention output (after o_proj, GPU)', numTokens, hiddenSize);
  }

  // Cleanup
  releaseBuffer(Q);
  releaseBuffer(K);
  releaseBuffer(V);
  if (output !== attnOutput) releaseBuffer(attnOutput);

  return { output, residualFused };
}

/**
 * Record attention for a single layer (batched, no submit).
 *
 * Uses record* kernel variants to batch all GPU operations into a shared
 * command encoder. No GPU submits happen here - submit once at end of forward pass.
 */
export async function recordLayerAttentionGPU(
  recorder: CommandRecorder,
  inputBuffer: GPUBuffer,
  layerWeights: LayerWeights | null,
  config: AttentionConfig,
  state: AttentionState,
  debug: boolean = false,
  debugFlags: AttentionDebugFlags = {},
  getWeightBuffer?: (weight: GPUBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer,
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
    attentionKernelOverride,
    residualBuffer,
    attnSoftcap = 0,
    queryPreAttnScalar,
  } = config;

  if (!layerWeights) {
    const output = acquireBuffer(numTokens * hiddenSize * 4, undefined, 'attn_output');
    return { output, residualFused: false };
  }

  const qSize = numTokens * numHeads * headDim;
  const kvSize = numTokens * numKVHeads * headDim;

  // 1. Input norm
  let normedBuffer = inputBuffer;
  if (layerWeights.inputNorm && getNormWeightBuffer) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.inputNorm, 'input_norm');
    normedBuffer = await recordRMSNorm(recorder, inputBuffer, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
    });
    if (!(layerWeights.inputNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);
  }

  // 2. Q/K/V projections
  // Use F16 activation outputs when KV cache is F16 (reduces memory bandwidth and avoids F32->F16 cast)
  const useF16Activations = state.kvCache?.kvDtype === 'f16';
  let Q: GPUBuffer, K: GPUBuffer, V: GPUBuffer;

  // Check for fused QKV path (3→1 matmul optimization)
  const hasLoRA = getLoRAModule(lora, layerIdx, 'q_proj') ||
                  getLoRAModule(lora, layerIdx, 'k_proj') ||
                  getLoRAModule(lora, layerIdx, 'v_proj');
  const useFusedQKV = layerWeights.qkvProj && layerWeights.qkvSizes && !hasLoRA && !useF16Activations;

  if (useFusedQKV && layerWeights.qkvProj && layerWeights.qkvSizes) {
    // FUSED PATH: Single matmul for Q/K/V, then split
    const [qSize_, kSize_, vSize_] = layerWeights.qkvSizes;
    const qkvSizeTotal = qSize_ + kSize_ + vSize_;

    // One fused matmul instead of 3 separate ones
    const qkvOutput = await recordMatmul(recorder, normedBuffer, layerWeights.qkvProj, numTokens, qkvSizeTotal, hiddenSize, {
      transposeB: 'auto',
    });

    // Split fused output into Q, K, V
    const split = await recordSplitQKV(recorder, qkvOutput, {
      numTokens,
      qSize: qSize_,
      kSize: kSize_,
      vSize: vSize_,
    });
    Q = split.Q;
    K = split.K;
    V = split.V;

    // Track fused buffer for cleanup
    recorder.trackTemporaryBuffer(qkvOutput);
  } else {
    // STANDARD PATH: Separate Q/K/V matmuls
    if (layerWeights.qProj && getWeightBuffer) {
      const qProjBuf = getWeightBuffer(layerWeights.qProj, 'q_proj');
      Q = await recordMatmul(recorder, normedBuffer, qProjBuf, numTokens, numHeads * headDim, hiddenSize, {
        transposeB: 'auto',
        outputDtype: useF16Activations ? 'f16' : undefined,
      });
      if (!(layerWeights.qProj instanceof GPUBuffer)) releaseBuffer(qProjBuf);
    } else {
      Q = acquireBuffer(qSize * 4, undefined, 'Q');
    }

    const loraQ = getLoRAModule(lora, layerIdx, 'q_proj');
    if (loraQ && getWeightBuffer) {
      const combined = await applyLoRA(
        normedBuffer,
        Q,
        loraQ,
        { M: numTokens, N: numHeads * headDim, K: hiddenSize },
        getWeightBuffer,
        recorder
      );
      if (combined !== Q) {
        recorder.trackTemporaryBuffer(Q);
        Q = combined;
      }
    }

    if (layerWeights.kProj && getWeightBuffer) {
      const kProjBuf = getWeightBuffer(layerWeights.kProj, 'k_proj');
      K = await recordMatmul(recorder, normedBuffer, kProjBuf, numTokens, numKVHeads * headDim, hiddenSize, {
        transposeB: 'auto',
        outputDtype: useF16Activations ? 'f16' : undefined,
      });
      if (!(layerWeights.kProj instanceof GPUBuffer)) releaseBuffer(kProjBuf);
    } else {
      K = acquireBuffer(kvSize * 4, undefined, 'K');
    }

    const loraK = getLoRAModule(lora, layerIdx, 'k_proj');
    if (loraK && getWeightBuffer) {
      const combined = await applyLoRA(
        normedBuffer,
        K,
        loraK,
        { M: numTokens, N: numKVHeads * headDim, K: hiddenSize },
        getWeightBuffer,
        recorder
      );
      if (combined !== K) {
        recorder.trackTemporaryBuffer(K);
        K = combined;
      }
    }

    if (layerWeights.vProj && getWeightBuffer) {
      const vProjBuf = getWeightBuffer(layerWeights.vProj, 'v_proj');
      V = await recordMatmul(recorder, normedBuffer, vProjBuf, numTokens, numKVHeads * headDim, hiddenSize, {
        transposeB: 'auto',
        outputDtype: useF16Activations ? 'f16' : undefined,
      });
      if (!(layerWeights.vProj instanceof GPUBuffer)) releaseBuffer(vProjBuf);
    } else {
      V = acquireBuffer(kvSize * 4, undefined, 'V');
    }

    const loraV = getLoRAModule(lora, layerIdx, 'v_proj');
    if (loraV && getWeightBuffer) {
      const combined = await applyLoRA(
        normedBuffer,
        V,
        loraV,
        { M: numTokens, N: numKVHeads * headDim, K: hiddenSize },
        getWeightBuffer,
        recorder
      );
      if (combined !== V) {
        recorder.trackTemporaryBuffer(V);
        V = combined;
      }
    }
  }

  // Optional per-head Q/K norm (Gemma-family)
  // Note: Gemma 3 q_norm and k_norm use Gemma3RMSNorm with (1+weight) formula
  if (layerWeights.qNorm && getNormWeightBuffer) {
    const qNormBuf = getNormWeightBuffer(layerWeights.qNorm, 'q_norm');
    const qElems = qNormBuf.size / 4;
    if (qElems === headDim) {
      const qNormed = await recordRMSNorm(recorder, Q, qNormBuf, rmsNormEps, {
        batchSize: numTokens * numHeads,
        hiddenSize: headDim,
      });
      releaseBuffer(Q);
      Q = qNormed;
    }
    if (!(layerWeights.qNorm instanceof GPUBuffer)) releaseBuffer(qNormBuf);
  }

  if (layerWeights.kNorm && getNormWeightBuffer) {
    const kNormBuf = getNormWeightBuffer(layerWeights.kNorm, 'k_norm');
    const kElems = kNormBuf.size / 4;
    if (kElems === headDim) {
      const kNormed = await recordRMSNorm(recorder, K, kNormBuf, rmsNormEps, {
        batchSize: numTokens * numKVHeads,
        hiddenSize: headDim,
      });
      releaseBuffer(K);
      K = kNormed;
    }
    if (!(layerWeights.kNorm instanceof GPUBuffer)) releaseBuffer(kNormBuf);
  }

  if (normedBuffer !== inputBuffer) releaseBuffer(normedBuffer);

  // 3. RoPE
  if (state.ropeFreqsCos && state.ropeFreqsSin) {
    await recordRoPE(recorder, Q, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
      numHeads, headDim, startPos: currentSeqLen,
    });
    await recordRoPE(recorder, K, state.ropeFreqsCos, state.ropeFreqsSin, numTokens, {
      numHeads: numKVHeads, headDim, startPos: currentSeqLen,
    });
  }

  // 4. Update KV cache
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
      const kElems = kvSize;
      const kDtype = getBufferDtype(K);
      const vDtype = getBufferDtype(V);

      const kForCache = kDtype === 'f16' ? K : await recordCastF32ToF16(recorder, K, kElems);
      const vForCache = vDtype === 'f16' ? V : await recordCastF32ToF16(recorder, V, kElems);

      state.kvCache.recordUpdateFromGPU(enc, layerIdx, kForCache, vForCache, currentSeqLen, numTokens);

      // Track for cleanup after submit (not release!) - only if we created new buffers
      if (kDtype !== 'f16') recorder.trackTemporaryBuffer(kForCache);
      if (vDtype !== 'f16') recorder.trackTemporaryBuffer(vForCache);
    } else {
      state.kvCache.recordUpdateFromGPU(enc, layerIdx, K, V, currentSeqLen, numTokens);
    }
    const gpuBuffers = state.kvCache.getGPUBuffers(layerIdx);
    cachedK = gpuBuffers.keysGPU;
    cachedV = gpuBuffers.valuesGPU;
    kvLenForAttention = gpuBuffers.seqLen;
  } else {
    cachedK = K;
    cachedV = V;
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
  const attnOutput = await recordAttention(recorder, Q, cachedK, cachedV, null, numHeads, headDim, {
    seqLen: numTokens,
    kvLen: kvLenForAttention,
    numKVHeads,
    causal: causalForAttention,
    startPos: startPosForMask,
    attentionKernel: attentionKernelOverride || undefined,
    slidingWindow: effectiveSlidingWindow,
    attnSoftcap,
    scale: attnScale,
  });

  // 6. Output projection (with optional fused residual for decode)
  let output: GPUBuffer;
  let residualFused = false;
  if (layerWeights.oProj && getWeightBuffer) {
    const oProjBuf = getWeightBuffer(layerWeights.oProj, 'o_proj');
    const loraO = getLoRAModule(lora, layerIdx, 'o_proj');

    // Use fused o_proj + residual for decode when possible
    const canUseFused = shouldUseFusedMatmulResidual(numTokens) &&
                        residualBuffer &&
                        !loraO &&
                        getBufferDtype(oProjBuf) === 'f16';

    if (canUseFused && residualBuffer) {
      // FUSED PATH: o_proj matmul + residual add in one dispatch
      output = await recordMatmulResidualFused(recorder, attnOutput, oProjBuf, residualBuffer, {
        N: hiddenSize,
        K: numHeads * headDim,
        residual: residualBuffer,
      });
      residualFused = true;
    } else {
      // STANDARD PATH: o_proj matmul only
      output = await recordMatmul(recorder, attnOutput, oProjBuf, numTokens, hiddenSize, numHeads * headDim, { transposeB: 'auto' });
    }
    if (!(layerWeights.oProj instanceof GPUBuffer)) releaseBuffer(oProjBuf);
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
      if (combined !== output) {
        recorder.trackTemporaryBuffer(output);
        output = combined;
      }
    }
  }

  // Track intermediate buffers for cleanup after submit (not release!)
  // These buffers are used by recorded operations that haven't executed yet.
  // Releasing them back to the pool would allow reuse before the encoder is submitted,
  // causing data corruption (especially for small decode buffers).
  recorder.trackTemporaryBuffer(Q);
  recorder.trackTemporaryBuffer(K);
  recorder.trackTemporaryBuffer(V);
  if (output !== attnOutput) recorder.trackTemporaryBuffer(attnOutput);

  return { output, residualFused };
}
