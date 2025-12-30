/**
 * Transformer layer processing (attention + FFN).
 *
 * This module orchestrates single-layer computation:
 * - Input normalization
 * - Self-attention
 * - Residual connections
 * - Feed-forward network (dense or MoE)
 *
 * Supports both standard (LLaMA-style) and sandwich norm (Gemma 3) architectures.
 *
 * @module inference/pipeline/layer
 */

import type { ParsedModelConfig } from './config.js';
import type { ExpertWeights, LayerWeights, MaybeGPUBuffer, RouterWeights } from './types.js';
import type { KVCache, SlidingWindowKVCache } from '../kv-cache.js';
import type { ExpertLoader } from './moe-impl.js';
import type { MoERouter } from '../moe-router.js';
import { getDevice } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../gpu/buffer-pool.js';
import { allowReadback } from '../../gpu/perf-guards.js';
import {
  runRMSNorm, runResidualAdd, runMatmul, runSiLU, runGeLU,
  recordRMSNorm, recordResidualAdd, recordMatmul, recordSiLU, recordGeLU,
  runSiLURowSplit, recordSiLURowSplit,
  runMatmulRMSNormFused, recordMatmulRMSNormFused, shouldUseFusedMatmulRMSNorm,
  CommandRecorder,
  type SiLURowSplitOptions,
} from '../../gpu/kernel-selector.js';
import { runLayerAttentionGPU, recordLayerAttentionGPU, type AttentionConfig, type AttentionState, type AttentionDebugFlags } from './attention.js';
import { getWeightBuffer, getNormWeightBuffer, type WeightBufferConfig, type WeightDebugFlags } from './weights.js';
import { logLayer, logAttn, logFFN, getBufferStats, isKernelDebugEnabled, dumpTokenVector, logKernelStep } from './debug-utils.js';
import { applyLoRA } from './lora-apply.js';
import { getLoRAModule, type LoRAAdapter } from './lora.js';
import { kernelTrace, traceStep } from './kernel-trace.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Layer context contains all state needed for layer processing.
 */
export interface LayerContext {
  /** Model configuration */
  config: ParsedModelConfig;
  /** Layer weights map */
  weights: Map<string, LayerWeights | Float32Array | GPUBuffer>;
  /** KV cache instance */
  kvCache: KVCache | SlidingWindowKVCache;
  /** Current sequence length */
  currentSeqLen: number;
  /** Whether to use GPU */
  useGPU: boolean;
  /** Debug mode */
  debug: boolean;
  /** RoPE frequency buffers (global for full_attention layers) */
  ropeFreqsCos: GPUBuffer | Float32Array | null;
  ropeFreqsSin: GPUBuffer | Float32Array | null;
  /** Local RoPE frequency buffers for sliding_attention layers (Gemma 3: 10K theta) */
  ropeLocalCos?: GPUBuffer | Float32Array | null;
  ropeLocalSin?: GPUBuffer | Float32Array | null;
  /** Attention kernel override */
  attentionKernelOverride: string | null;
  /** Weight buffer config */
  weightConfig: WeightBufferConfig;
  /** Debug flags (mutable) */
  debugFlags?: WeightDebugFlags;
  /** Expert weights map (for MoE) */
  expertWeights?: Map<string, ExpertWeights>;
  /** Expert loader (for MoE) */
  expertLoader?: ExpertLoader | null;
  /** MoE router (for MoE) */
  moeRouter?: MoERouter | null;
  /** Layer router weights (for models with per-layer routers) */
  layerRouterWeights?: Map<number, RouterWeights>;
  /** Command recorder for batched GPU operations (optional) */
  recorder?: CommandRecorder;
  /** Optional LoRA adapter */
  lora?: LoRAAdapter | null;
}

/**
 * Layer processing result.
 */
export interface LayerResult {
  /** Output hidden states (GPUBuffer or Float32Array) */
  output: GPUBuffer | Float32Array;
  /** Whether output is on GPU */
  isGPU: boolean;
}

/**
 * Sandwich norm detection result.
 */
export interface SandwichNormInfo {
  /** Whether sandwich norms are used */
  useSandwichNorm: boolean;
  /** Has pre-feedforward norm */
  hasPreFeedforwardNorm: boolean;
  /** Has post-feedforward norm */
  hasPostFeedforwardNorm: boolean;
  /** Has post-attention norm */
  hasPostAttentionNorm: boolean;
}

// ============================================================================
// Kernel Wrappers (run or record based on context)
// ============================================================================

/**
 * RMSNorm that uses record variant when recorder is provided.
 */
async function doRMSNorm(
  input: GPUBuffer,
  weight: GPUBuffer,
  eps: number,
  options: { batchSize: number; hiddenSize: number; residual?: GPUBuffer | null; label?: string; layerIdx?: number },
  recorder?: CommandRecorder
): Promise<GPUBuffer> {
  const result = recorder
    ? await recordRMSNorm(recorder, input, weight, eps, options)
    : await runRMSNorm(input, weight, eps, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder) {
    const layer = options.layerIdx ?? -1;
    const label = options.label ?? 'rmsnorm';
    await traceStep('rmsnorm', label, layer, result, [options.batchSize, options.hiddenSize]);
  }

  return result;
}

/**
 * ResidualAdd that uses record variant when recorder is provided.
 */
async function doResidualAdd(
  a: GPUBuffer,
  b: GPUBuffer,
  size: number,
  recorder?: CommandRecorder,
  traceOptions?: { label?: string; layerIdx?: number }
): Promise<GPUBuffer> {
  const result = recorder
    ? await recordResidualAdd(recorder, a, b, size)
    : await runResidualAdd(a, b, size);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder && traceOptions) {
    await traceStep('residual_add', traceOptions.label ?? 'residual', traceOptions.layerIdx ?? -1, result, [size]);
  }

  return result;
}

/**
 * Matmul that uses record variant when recorder is provided.
 */
async function doMatmul(
  A: GPUBuffer,
  B: GPUBuffer,
  M: number,
  N: number,
  K: number,
  options: { transposeB?: boolean | 'auto'; label?: string; layerIdx?: number } = {},
  recorder?: CommandRecorder
): Promise<GPUBuffer> {
  const result = recorder
    ? await recordMatmul(recorder, A, B, M, N, K, options)
    : await runMatmul(A, B, M, N, K, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder) {
    const layer = options.layerIdx ?? -1;
    const label = options.label ?? 'matmul';
    await traceStep('matmul', label, layer, result, [M, N]);
  }

  return result;
}

/**
 * SiLU that uses record variant when recorder is provided.
 * Supports gated variant (SiLU with gate multiplication).
 */
async function doSiLU(
  input: GPUBuffer,
  options: { size?: number; gate?: GPUBuffer; label?: string; layerIdx?: number } = {},
  recorder?: CommandRecorder
): Promise<GPUBuffer> {
  const result = recorder
    ? await recordSiLU(recorder, input, options)
    : await runSiLU(input, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder && options.size) {
    await traceStep('silu', options.label ?? 'silu', options.layerIdx ?? -1, result, [options.size]);
  }

  return result;
}

/**
 * GeLU that uses record variant when recorder is provided.
 * Supports gated variant (GeGLU).
 */
async function doGeLU(
  input: GPUBuffer,
  options: { size?: number; gate?: GPUBuffer; label?: string; layerIdx?: number } = {},
  recorder?: CommandRecorder
): Promise<GPUBuffer> {
  const result = recorder
    ? await recordGeLU(recorder, input, options)
    : await runGeLU(input, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder && options.size) {
    await traceStep('gelu', options.label ?? 'gelu', options.layerIdx ?? -1, result, [options.size]);
  }

  return result;
}

/**
 * SiLURowSplit that uses record variant when recorder is provided.
 * Used for fused gate+up FFN path: splits combined output and applies activation.
 */
async function doSiLURowSplit(
  input: GPUBuffer,
  options: SiLURowSplitOptions & { label?: string; layerIdx?: number },
  recorder?: CommandRecorder
): Promise<GPUBuffer> {
  const result = recorder
    ? await recordSiLURowSplit(recorder, input, options)
    : await runSiLURowSplit(input, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder) {
    await traceStep('silu_row_split', options.label ?? 'ffn_activation', options.layerIdx ?? -1, result, [options.numTokens, options.dim]);
  }

  return result;
}

/**
 * Fused Matmul + RMSNorm that uses record variant when recorder is provided.
 * Used for down projection + post-FFN norm fusion during decode (M=1).
 */
async function doMatmulRMSNormFused(
  input: GPUBuffer,
  weight: GPUBuffer,
  normWeight: GPUBuffer,
  options: { N: number; K: number; eps: number; residual?: GPUBuffer | null; label?: string; layerIdx?: number },
  recorder?: CommandRecorder
): Promise<GPUBuffer> {
  const result = recorder
    ? await recordMatmulRMSNormFused(recorder, input, weight, normWeight, options)
    : await runMatmulRMSNormFused(input, weight, normWeight, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder) {
    await traceStep('matmul_rmsnorm_fused', options.label ?? 'ffn_fused', options.layerIdx ?? -1, result, [1, options.N]);
  }

  return result;
}

/**
 * Attention that uses record variant when recorder is provided.
 */
async function doAttention(
  inputBuffer: GPUBuffer,
  layerWeights: LayerWeights | null,
  config: AttentionConfig,
  state: AttentionState,
  debug: boolean,
  debugFlags: AttentionDebugFlags,
  getWeightBufferFn: (weight: GPUBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer,
  getNormWeightBufferFn: (weight: GPUBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer,
  recorder?: CommandRecorder,
  lora?: LoRAAdapter | null
): Promise<GPUBuffer> {
  if (recorder) {
    return recordLayerAttentionGPU(
      recorder,
      inputBuffer,
      layerWeights,
      config,
      state,
      debug,
      debugFlags,
      getWeightBufferFn,
      getNormWeightBufferFn,
      undefined,
      lora
    );
  }
  return runLayerAttentionGPU(
    inputBuffer,
    layerWeights,
    config,
    state,
    debug,
    debugFlags,
    getWeightBufferFn,
    getNormWeightBufferFn,
    undefined,
    lora
  );
}

// ============================================================================
// Architecture Detection
// ============================================================================

/**
 * Detect sandwich norm architecture (Gemma 3).
 */
export function detectSandwichNorm(layerWeights: LayerWeights | null): SandwichNormInfo {
  const hasPreFeedforwardNorm = Boolean(layerWeights?.preFeedforwardNorm);
  const hasPostFeedforwardNorm = Boolean(layerWeights?.postFeedforwardNorm);
  const hasPostAttentionNorm = Boolean(layerWeights?.postAttentionNorm);

  return {
    useSandwichNorm: hasPreFeedforwardNorm || hasPostFeedforwardNorm,
    hasPreFeedforwardNorm,
    hasPostFeedforwardNorm,
    hasPostAttentionNorm,
  };
}

/**
 * Check if a layer is a MoE layer.
 */
export function isMoELayer(
  layerIdx: number,
  config: ParsedModelConfig,
  layerWeights?: LayerWeights | null
): boolean {
  if (!config.useMoE) return false;

  // Check if layer has router weights
  if (layerWeights?.routerWeight) return true;

  // Fall back to layer_types array if available
  const layerTypes = config.layerTypes;
  if (Array.isArray(layerTypes) && layerIdx < layerTypes.length) {
    return layerTypes[layerIdx] === 'moe';
  }

  // Default: assume all layers are MoE if model uses MoE
  return true;
}

// ============================================================================
// Main Layer Processing
// ============================================================================

/**
 * Process a single transformer layer.
 *
 * This is the main orchestration function that delegates to:
 * - processLayerGPU() for GPU execution
 * - processLayerCPU() for CPU fallback
 *
 * The layer processing follows either:
 *
 * Standard (LLaMA-style) architecture:
 *   1. x_norm = input_layernorm(x)
 *   2. attn_out = attention(x_norm)
 *   3. x = x + attn_out  // residual
 *   4. x_norm = post_attn_norm(x)
 *   5. ffn_out = ffn(x_norm)
 *   6. x = x + ffn_out  // residual
 *
 * Sandwich norm (Gemma 3) architecture:
 *   1. x_norm = input_layernorm(x)
 *   2. attn_out = attention(x_norm)
 *   3. attn_out = post_attention_layernorm(attn_out)  // BEFORE residual
 *   4. x = x + attn_out  // residual AFTER norm
 *   5. ffn_in = pre_feedforward_layernorm(x)
 *   6. ffn_out = mlp(ffn_in)
 *   7. ffn_out = post_feedforward_layernorm(ffn_out)  // BEFORE residual
 *   8. x = x + ffn_out  // residual AFTER norm
 *
 * @param layerIdx - Layer index
 * @param hiddenStates - Input hidden states (GPUBuffer or Float32Array)
 * @param numTokens - Number of tokens in the batch
 * @param isPrefill - Whether this is prefill (true) or decode (false)
 * @param context - Layer processing context
 * @returns Layer output
 */
export async function processLayer(
  layerIdx: number,
  hiddenStates: GPUBuffer | Float32Array,
  numTokens: number,
  isPrefill: boolean,
  context: LayerContext
): Promise<GPUBuffer | Float32Array> {
  const { config, useGPU } = context;
  const { hiddenSize } = config;
  const size = numTokens * hiddenSize;

  // Debug routing (uses debug-utils)
  logLayer(layerIdx, 'enter', isPrefill, { numTokens });

  // Debug: check path being taken for layer 0
  if (context.debug && layerIdx === 0) {
    console.log(`[processLayer] L0 routing: useGPU=${useGPU}, isGPUBuffer=${hiddenStates instanceof GPUBuffer}, constructor=${hiddenStates?.constructor?.name}`);
  }

  // GPU-native path
  if (useGPU && hiddenStates instanceof GPUBuffer) {
    return processLayerGPU(layerIdx, hiddenStates, numTokens, isPrefill, size, context);
  }

  // CPU fallback path
  return processLayerCPU(layerIdx, hiddenStates as Float32Array, numTokens, isPrefill, context);
}

// ============================================================================
// GPU Layer Processing
// ============================================================================

/**
 * GPU-native layer processing (no CPU readbacks).
 */
export async function processLayerGPU(
  layerIdx: number,
  inputBuffer: GPUBuffer,
  numTokens: number,
  isPrefill: boolean,
  size: number,
  context: LayerContext
): Promise<GPUBuffer> {
  // Debug entry (uses debug-utils)
  logLayer(layerIdx, 'enter', isPrefill, { numTokens });

  const device = getDevice();
  if (!device) throw new Error('No GPU device available');

  const { config, weights, weightConfig, debugFlags, kvCache, ropeFreqsCos, ropeFreqsSin, attentionKernelOverride, recorder } = context;
  const { hiddenSize, numHeads, numKVHeads, headDim, rmsNormEps } = config;

  const layerWeights = weights.get(`layer_${layerIdx}`) as LayerWeights | undefined;
  const sandwichNorm = detectSandwichNorm(layerWeights);
  const lastTokenIdx = Math.max(0, numTokens - 1);

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    logKernelStep('layer', { layerIdx, label: `seqLen=${numTokens} prefill=${isPrefill}` });
    await dumpTokenVector(inputBuffer, 'layer_in', { layerIdx, tokenIdx: lastTokenIdx, rowSize: hiddenSize });
  }

  // 1. Self-attention (returns GPU buffer)
  // Determine layer type for RoPE frequency selection (Gemma 3: sliding vs full attention)
  const layerType = config.layerTypes?.[layerIdx];
  const isLocalLayer = layerType === 'sliding_attention';

  // Debug: log RoPE selection for first few layers
  if (context.debug && layerIdx < 3) {
    console.log(`[Layer${layerIdx}] RoPE selection: layerType=${layerType}, isLocal=${isLocalLayer}, hasLocalCos=${!!context.ropeLocalCos}, hasLocalSin=${!!context.ropeLocalSin}`);
  }

  const attnConfig: AttentionConfig = {
    layerIdx,
    numTokens,
    isPrefill,
    numHeads,
    numKVHeads,
    headDim,
    hiddenSize,
    rmsNormEps,
    currentSeqLen: context.currentSeqLen,
    slidingWindow: config.slidingWindow,
    layerType,
    attentionKernelOverride,
  };

  // Select RoPE frequencies based on layer type:
  // - Local (sliding_attention) layers use ropeLocalTheta (10K for Gemma 3)
  // - Global (full_attention) layers use ropeTheta (1M for Gemma 3)
  const attnState: AttentionState = {
    ropeFreqsCos: (isLocalLayer && context.ropeLocalCos)
      ? context.ropeLocalCos as GPUBuffer | null
      : ropeFreqsCos as GPUBuffer | null,
    ropeFreqsSin: (isLocalLayer && context.ropeLocalSin)
      ? context.ropeLocalSin as GPUBuffer | null
      : ropeFreqsSin as GPUBuffer | null,
    kvCache: kvCache as unknown as import('./types.js').KVCacheInterface,
  };

  const attnOutput = await doAttention(
    inputBuffer,
    layerWeights ?? null,
    attnConfig,
    attnState,
    context.debug,
    {},
    (weight, label) => getWeightBuffer(weight, label),
    (weight, label) => getNormWeightBuffer(weight, label, weightConfig, debugFlags),
    recorder,
    context.lora
  );

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(attnOutput, 'attn_out', { layerIdx, tokenIdx: lastTokenIdx, rowSize: hiddenSize });
  }

  // Debug: trace attn output (uses debug-utils)
  if (context.debug) {
    const stats = await getBufferStats(attnOutput);
    if (stats) logAttn(layerIdx, isPrefill, { numTokens, kvLen: context.currentSeqLen + (isPrefill ? numTokens : 1), maxAbsOut: stats.maxAbs });

    // Debug layer 0, 2, and 17 attention output specifically (2 is where decode explosion happens)
    // NOTE: Skip detailed readback when using recorder (batched mode) - buffers aren't populated yet!
    console.log(`[Layer${layerIdx}] attnOutput type check: isGPU=${attnOutput instanceof GPUBuffer}, type=${typeof attnOutput}, constructor=${attnOutput?.constructor?.name}, isPrefill=${isPrefill}`);
    if ((layerIdx === 0 || layerIdx === 2 || layerIdx === 17) && attnOutput instanceof GPUBuffer && !recorder) {
      if (allowReadback(`layer.attn-out.${layerIdx}`)) {
        try {
          const sampleSize = Math.min(128, attnOutput.size);
          const staging = device.createBuffer({ size: sampleSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          const enc = device.createCommandEncoder();
          enc.copyBufferToBuffer(attnOutput, 0, staging, 0, sampleSize);
          device.queue.submit([enc.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          const data = new Float32Array(staging.getMappedRange().slice(0));
          staging.unmap();
          staging.destroy();
          const maxAbs = Math.max(...Array.from(data).map(x => Math.abs(x)));
          const nonZero = Array.from(data).filter(x => x !== 0).length;
          console.log(`[Layer${layerIdx}] ATTN_OUT: maxAbs=${maxAbs.toFixed(4)}, nonZero=${nonZero}/${data.length}, sample=[${Array.from(data).slice(0, 5).map(x => x.toFixed(4)).join(', ')}]`);
        } catch (e) {
          console.log(`[Layer${layerIdx}] ATTN_OUT error: ${e}`);
        }
      }
    } else if ((layerIdx === 0 || layerIdx === 2 || layerIdx === 17) && attnOutput instanceof GPUBuffer && recorder) {
      console.log(`[Layer${layerIdx}] ATTN_OUT: (skipped - using batched recorder, values not available until submit)`);
    }
  }

  // 2. Handle residual connection based on architecture
  // Debug: log architecture path for layer 0, 2, and 17
  if (context.debug && (layerIdx === 0 || layerIdx === 2 || layerIdx === 17)) {
    console.log(`[Layer${layerIdx}] ARCH: sandwich=${sandwichNorm.useSandwichNorm}, hasPostAttnNorm=${sandwichNorm.hasPostAttentionNorm}, hasWeights=${!!layerWeights?.postAttentionNorm}`);
  }

  let postAttn: GPUBuffer;
  if (sandwichNorm.useSandwichNorm && sandwichNorm.hasPostAttentionNorm && layerWeights?.postAttentionNorm) {
    // Gemma 3 path: FUSED norm attention output + residual add (1 kernel instead of 2)
    const normWeightBuf = getNormWeightBuffer(layerWeights.postAttentionNorm, 'post_attention_norm', weightConfig, debugFlags);
    if (context.debug && layerIdx === 0) {
      console.log(`[Layer0] RESIDUAL_DEBUG: inputBuffer.size=${inputBuffer.size}, attnOutput.size=${attnOutput.size}, hasRecorder=${!!recorder}`);
      // Debug: verify inputBuffer (residual) has the expected embedding values
      if (inputBuffer instanceof GPUBuffer && allowReadback(`layer.residual-check.${layerIdx}`)) {
        try {
          const device = getDevice();
          const sampleSize = Math.min(128, inputBuffer.size);
          const staging = device.createBuffer({ size: sampleSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          const enc = device.createCommandEncoder();
          enc.copyBufferToBuffer(inputBuffer, 0, staging, 0, sampleSize);
          device.queue.submit([enc.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          const data = new Float32Array(staging.getMappedRange().slice(0));
          staging.unmap();
          staging.destroy();
          const maxAbs = Math.max(...Array.from(data).map(x => Math.abs(x)));
          console.log(`[Layer0] INPUT_RESIDUAL: maxAbs=${maxAbs.toFixed(4)}, first5=[${Array.from(data).slice(0, 5).map(x => x.toFixed(4)).join(', ')}]`);
        } catch (e) {
          console.log(`[Layer0] INPUT_RESIDUAL error: ${e}`);
        }
      }
    }
    postAttn = await doRMSNorm(attnOutput, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      residual: inputBuffer,  // FUSION: Add residual in same kernel
      label: `L${layerIdx}.post_attn_norm`,
      layerIdx,
    }, recorder);

    if (!(layerWeights.postAttentionNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);
    // Track for cleanup after submit if using recorder, otherwise release immediately
    if (recorder) {
      recorder.trackTemporaryBuffer(attnOutput);
    } else {
      releaseBuffer(attnOutput);
    }
  } else {
    // Standard path: residual add first
    postAttn = await doResidualAdd(attnOutput, inputBuffer, size, recorder, { label: `L${layerIdx}.post_attn_residual`, layerIdx });
    // Track for cleanup after submit if using recorder, otherwise release immediately
    if (recorder) {
      recorder.trackTemporaryBuffer(attnOutput);
    } else {
      releaseBuffer(attnOutput);
    }
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(postAttn, 'x_after_attn', { layerIdx, tokenIdx: lastTokenIdx, rowSize: hiddenSize });
  }

  // Debug: log postAttn for layer 0
  // NOTE: Skip when using recorder (batched mode) - buffers not populated yet
  if (context.debug && layerIdx === 0 && postAttn instanceof GPUBuffer && !recorder) {
    if (allowReadback('layer.post-attn.0')) {
      try {
        await device.queue.onSubmittedWorkDone();
        const sampleSize = Math.min(128, postAttn.size);
        const staging = device.createBuffer({ size: sampleSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(postAttn, 0, staging, 0, sampleSize);
        device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        const maxAbs = Math.max(...Array.from(data).map(x => Math.abs(x)));
        console.log(`[Layer0] POST_ATTN: maxAbs=${maxAbs.toFixed(4)}, sample=[${Array.from(data).slice(0, 5).map(x => x.toFixed(4)).join(', ')}]`);
      } catch (e) {
        console.log(`[Layer0] POST_ATTN error: ${e}`);
      }
    }
  }

  // 3. Feed-forward network
  if (sandwichNorm.useSandwichNorm) {
    return processFFNWithSandwichNorm(layerIdx, postAttn, numTokens, size, context, layerWeights, sandwichNorm);
  } else {
    return processFFNStandard(layerIdx, postAttn, numTokens, size, context, layerWeights);
  }
}

/**
 * Process FFN with sandwich norm architecture (Gemma 3).
 */
async function processFFNWithSandwichNorm(
  layerIdx: number,
  postAttn: GPUBuffer,
  numTokens: number,
  size: number,
  context: LayerContext,
  layerWeights: LayerWeights | undefined,
  sandwichNorm: SandwichNormInfo
): Promise<GPUBuffer> {
  const device = getDevice();
  const { config, weightConfig, debugFlags, recorder } = context;
  const { hiddenSize, rmsNormEps } = config;
  const lastTokenIdx = Math.max(0, numTokens - 1);

  // Debug helper for layers 0, 2, and 17 (only runs when context.debug is true)
  // NOTE: Skip when using recorder (batched mode) - buffers not populated yet
  const debugLayer = async (buf: GPUBuffer, label: string) => {
    if (!context.debug || (layerIdx !== 0 && layerIdx !== 2 && layerIdx !== 17) || !device || recorder) return;
    if (!allowReadback(`layer.ffn.${label}.${layerIdx}`)) return;
    try {
      // Force GPU sync before reading
      await device.queue.onSubmittedWorkDone();

      const sampleSize = Math.min(128, buf.size);
      const staging = device.createBuffer({ size: sampleSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(buf, 0, staging, 0, sampleSize);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      const maxAbs = Math.max(...Array.from(data).map(x => Math.abs(x)));
      const nonZero = Array.from(data).filter(x => x !== 0).length;
      console.log(`[Layer${layerIdx}] FFN_${label}: maxAbs=${maxAbs.toFixed(4)}, nonZero=${nonZero}/${data.length}`);
    } catch (e) {
      console.log(`[Layer${layerIdx}] FFN_${label} error: ${e}`);
    }
  };

  // 1. Pre-FFN norm (applied to residual stream before FFN)
  let ffnInput = postAttn;
  if (sandwichNorm.hasPreFeedforwardNorm && layerWeights?.preFeedforwardNorm) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.preFeedforwardNorm, 'pre_feedforward_norm', weightConfig, debugFlags);

    // Debug: check norm weight values for layer 0, 2, and 17
    if (context.debug && (layerIdx === 0 || layerIdx === 2 || layerIdx === 17) && device) {
      if (allowReadback(`layer.pre-ffn-norm.${layerIdx}`)) {
        try {
          const ws = Math.min(128, normWeightBuf.size);
          const wstaging = device.createBuffer({ size: ws, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          const wenc = device.createCommandEncoder();
          wenc.copyBufferToBuffer(normWeightBuf, 0, wstaging, 0, ws);
          device.queue.submit([wenc.finish()]);
          await wstaging.mapAsync(GPUMapMode.READ);
          const wdata = new Float32Array(wstaging.getMappedRange().slice(0));
          wstaging.unmap();
          wstaging.destroy();
          const wmaxAbs = Math.max(...Array.from(wdata).map(x => Math.abs(x)));
          const wnonZero = Array.from(wdata).filter(x => x !== 0).length;
          console.log(`[Layer${layerIdx}] PRE_FFN_NORM_WEIGHTS: maxAbs=${wmaxAbs.toFixed(4)}, nonZero=${wnonZero}/${wdata.length}, sample=[${Array.from(wdata).slice(0, 5).map(x => x.toFixed(4)).join(', ')}]`);
        } catch (e) { console.log(`[Layer${layerIdx}] PRE_FFN_NORM_WEIGHTS error: ${e}`); }
      }
    }

    // Debug: check input values for layers 0 and 17
    await debugLayer(postAttn, 'PRE_NORM_INPUT');

    ffnInput = await doRMSNorm(postAttn, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      label: `L${layerIdx}.pre_ffn_norm`,
      layerIdx,
    }, recorder);
    if (!(layerWeights.preFeedforwardNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);
    await debugLayer(ffnInput, 'PRE_NORM_OUTPUT');
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(ffnInput, 'pre_ffn_norm_out', { layerIdx, tokenIdx: lastTokenIdx, rowSize: hiddenSize });
  }

  // 2. FFN (or MoE FFN)
  // Check if we can use fused down+norm kernel (decode only, dense FFN with post-FFN norm)
  const canUseFusedDownNorm = numTokens === 1
    && !config.useMoE
    && !isMoELayer(layerIdx, config, layerWeights)
    && sandwichNorm.hasPostFeedforwardNorm
    && layerWeights?.postFeedforwardNorm
    && layerWeights?.down
    && shouldUseFusedMatmulRMSNorm(numTokens, hiddenSize);

  let ffnOutput: GPUBuffer;
  let usedFusedDownNorm = false;

  if (config.useMoE && isMoELayer(layerIdx, config, layerWeights)) {
    ffnOutput = await runMoEFFNGPU(layerIdx, ffnInput, numTokens, context);
  } else if (canUseFusedDownNorm && layerWeights?.down && layerWeights?.postFeedforwardNorm &&
             (layerWeights?.gateUp || (layerWeights?.gate && layerWeights?.up))) {
    // FUSED PATH: gate+up (or separate gate/up) -> activation -> down+norm+residual (single kernel for last step)
    if (layerIdx === 0) console.warn('[FUSED] Using fused down+norm kernel for layer 0');
    ffnOutput = await runDenseFFNWithFusedPostNormGPU(
      layerIdx, ffnInput, numTokens, context, layerWeights,
      postAttn,  // residual for post-FFN norm
      rmsNormEps
    );
    usedFusedDownNorm = true;
  } else {
    ffnOutput = await runDenseFFNGPU(layerIdx, ffnInput, numTokens, context, layerWeights);
  }
  await debugLayer(ffnOutput, 'FFN_OUT');

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(ffnOutput, 'ffn_out', { layerIdx, tokenIdx: lastTokenIdx, rowSize: hiddenSize });
  }

  // Track for cleanup after submit if using recorder, otherwise release immediately
  if (ffnInput !== postAttn) {
    if (recorder) {
      recorder.trackTemporaryBuffer(ffnInput);
    } else {
      releaseBuffer(ffnInput);
    }
  }

  // Debug: trace FFN output (uses debug-utils)
  const ffnStats = await getBufferStats(ffnOutput);
  if (ffnStats) logFFN(layerIdx, { maxAbsOut: ffnStats.maxAbs });

  // 3. Post-FFN norm - applied to FFN output BEFORE residual add
  // Skip if we already used fused down+norm kernel
  let output: GPUBuffer;
  if (usedFusedDownNorm) {
    // Fused kernel already applied norm + residual
    output = ffnOutput;
  } else if (sandwichNorm.hasPostFeedforwardNorm && layerWeights?.postFeedforwardNorm) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.postFeedforwardNorm, 'post_feedforward_norm', weightConfig, debugFlags);

    // FUSED: norm FFN output + residual add (1 kernel instead of 2)
    output = await doRMSNorm(ffnOutput, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      residual: postAttn,  // FUSION: Add residual in same kernel
      label: `L${layerIdx}.post_ffn_norm`,
      layerIdx,
    }, recorder);

    if (!(layerWeights.postFeedforwardNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);
    // Track for cleanup after submit if using recorder, otherwise release immediately
    if (recorder) {
      recorder.trackTemporaryBuffer(ffnOutput);
    } else {
      releaseBuffer(ffnOutput);
    }
  } else {
    // Standard path: residual add without norm
    output = await doResidualAdd(ffnOutput, postAttn, size, recorder, { label: `L${layerIdx}.post_ffn_residual`, layerIdx });
    if (recorder) {
      recorder.trackTemporaryBuffer(ffnOutput);
    } else {
      releaseBuffer(ffnOutput);
    }
  }

  await debugLayer(output, 'FINAL');

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(output, 'layer_out', { layerIdx, tokenIdx: lastTokenIdx, rowSize: hiddenSize });
  }

  // Debug: Log output magnitude for EVERY layer to track growth
  if (context.debug && device && !recorder) {
    if (allowReadback(`layer.output.${layerIdx}`)) {
      try {
        const fullSize = output.size;
        const staging = device.createBuffer({ size: fullSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(output, 0, staging, 0, fullSize);
        device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        // Find max value and its position
        let maxAbs = 0;
        let maxIdx = -1;
        for (let i = 0; i < data.length; i++) {
          const abs = Math.abs(data[i]);
          if (abs > maxAbs) { maxAbs = abs; maxIdx = i; }
        }
        // Calculate which token this is (hiddenSize = 1152 for Gemma 3 1B)
        const tokenIdx = Math.floor(maxIdx / hiddenSize);
        const dimIdx = maxIdx % hiddenSize;
        console.log(`[LAYER_OUT] L${layerIdx}: maxAbs=${maxAbs.toFixed(4)} at idx=${maxIdx} (token=${tokenIdx}, dim=${dimIdx}), bufSize=${data.length}`);
      } catch (e) { /* ignore */ }
    }
  }
  // Track postAttn for cleanup after submit if using recorder, otherwise release immediately
  if (recorder) {
    recorder.trackTemporaryBuffer(postAttn);
  } else {
    releaseBuffer(postAttn);
  }

  return output;
}

/**
 * Process FFN with standard architecture (LLaMA-style).
 */
async function processFFNStandard(
  layerIdx: number,
  postAttn: GPUBuffer,
  numTokens: number,
  size: number,
  context: LayerContext,
  layerWeights: LayerWeights | undefined
): Promise<GPUBuffer> {
  const { config, weightConfig, debugFlags, recorder } = context;
  const { hiddenSize, rmsNormEps } = config;

  // 1. Post-attention norm (LLaMA-style pre-FFN norm)
  let normedBuffer = postAttn;
  if (layerWeights?.postAttnNorm) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.postAttnNorm, 'post_attn_norm', weightConfig, debugFlags);
    normedBuffer = await doRMSNorm(postAttn, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      label: `L${layerIdx}.post_attn_norm`,
      layerIdx,
    }, recorder);
    if (!(layerWeights.postAttnNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);
  }

  // 2. FFN (or MoE FFN)
  let ffnOutput: GPUBuffer;
  if (config.useMoE && isMoELayer(layerIdx, config, layerWeights)) {
    ffnOutput = await runMoEFFNGPU(layerIdx, normedBuffer, numTokens, context);
  } else {
    ffnOutput = await runDenseFFNGPU(layerIdx, normedBuffer, numTokens, context, layerWeights);
  }

  // 3. Residual add: ffnOutput + postAttn
  const output = await doResidualAdd(ffnOutput, postAttn, size, recorder, { label: `L${layerIdx}.ffn_residual`, layerIdx });

  // Track for cleanup after submit if using recorder, otherwise release immediately
  if (normedBuffer !== postAttn) {
    if (recorder) {
      recorder.trackTemporaryBuffer(normedBuffer);
    } else {
      releaseBuffer(normedBuffer);
    }
  }
  if (recorder) {
    recorder.trackTemporaryBuffer(postAttn);
    recorder.trackTemporaryBuffer(ffnOutput);
  } else {
    releaseBuffer(postAttn);
    releaseBuffer(ffnOutput);
  }

  return output;
}

// ============================================================================
// FFN Operations
// ============================================================================

/**
 * Run dense FFN on GPU.
 *
 * Supports two paths:
 * - Fused (2 matmuls): gateUp -> split+activate -> down
 * - Separate (3 matmuls): gate, up -> activate -> down
 */
async function runDenseFFNGPU(
  layerIdx: number,
  inputBuffer: GPUBuffer,
  numTokens: number,
  context: LayerContext,
  layerWeights: LayerWeights | undefined
): Promise<GPUBuffer> {
  const device = getDevice();
  if (!device) throw new Error('No GPU device');

  const { config, recorder } = context;
  const { hiddenSize, intermediateSize, hiddenActivation } = config;
  const lastTokenIdx = Math.max(0, numTokens - 1);
  const lora = context.lora || null;

  // Check for fused gate+up path (2 matmuls instead of 3)
  if (layerWeights?.gateUp && layerWeights?.down) {
    const gateUpWeight = getWeightBuffer(layerWeights.gateUp, 'ffn_gate_up');
    const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');

    // 1. Fused gate+up projection: [numTokens, hiddenSize] @ [intermediateSize*2, hiddenSize]^T -> [numTokens, intermediateSize*2]
    let gateUpOutput = await doMatmul(
      inputBuffer, gateUpWeight,
      numTokens, intermediateSize * 2, hiddenSize,
      { transposeB: 'auto', label: `L${layerIdx}.ffn_gate_up`, layerIdx },
      recorder
    );

    const loraGateUp = getLoRAModule(lora, layerIdx, 'gate_up_proj');
    if (loraGateUp) {
      const combined = await applyLoRA(
        inputBuffer,
        gateUpOutput,
        loraGateUp,
        { M: numTokens, N: intermediateSize * 2, K: hiddenSize },
        getWeightBuffer,
        recorder
      );
      if (combined !== gateUpOutput) {
        if (recorder) {
          recorder.trackTemporaryBuffer(gateUpOutput);
        } else {
          releaseBuffer(gateUpOutput);
        }
        gateUpOutput = combined;
      }
    }

    if (isKernelDebugEnabled(layerIdx) && !recorder) {
      await dumpTokenVector(gateUpOutput, 'ffn_gate_up', {
        layerIdx,
        tokenIdx: lastTokenIdx,
        rowSize: intermediateSize * 2,
      });
    }

    if (!(layerWeights.gateUp instanceof GPUBuffer)) releaseBuffer(gateUpWeight);

    // 2. Split + Activation: output[i] = activation(gate[i]) * up[i]
    const activation = hiddenActivation === 'gelu' ? 'gelu' : 'silu';
    const activatedOutput = await doSiLURowSplit(gateUpOutput, {
      numTokens,
      dim: intermediateSize,
      activation,
      label: `L${layerIdx}.ffn_activation`,
      layerIdx,
    }, recorder);

    if (isKernelDebugEnabled(layerIdx) && !recorder) {
      await dumpTokenVector(activatedOutput, 'ffn_activated', {
        layerIdx,
        tokenIdx: lastTokenIdx,
        rowSize: intermediateSize,
      });
    }

    // Track for cleanup after submit if using recorder, otherwise release immediately
    if (recorder) {
      recorder.trackTemporaryBuffer(gateUpOutput);
    } else {
      releaseBuffer(gateUpOutput);
    }

    // 3. Down projection: [numTokens, intermediateSize] @ [hiddenSize, intermediateSize]^T -> [numTokens, hiddenSize]
    let output = await doMatmul(
      activatedOutput, downWeight,
      numTokens, hiddenSize, intermediateSize,
      { transposeB: 'auto', label: `L${layerIdx}.ffn_down`, layerIdx },
      recorder
    );

    const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
    if (loraDown) {
      const combined = await applyLoRA(
        activatedOutput,
        output,
        loraDown,
        { M: numTokens, N: hiddenSize, K: intermediateSize },
        getWeightBuffer,
        recorder
      );
      if (combined !== output) {
        if (recorder) {
          recorder.trackTemporaryBuffer(output);
        } else {
          releaseBuffer(output);
        }
        output = combined;
      }
    }

    if (isKernelDebugEnabled(layerIdx) && !recorder) {
      await dumpTokenVector(output, 'ffn_down_out', {
        layerIdx,
        tokenIdx: lastTokenIdx,
        rowSize: hiddenSize,
      });
    }

    if (!(layerWeights.down instanceof GPUBuffer)) releaseBuffer(downWeight);
    // Track for cleanup after submit if using recorder, otherwise release immediately
    if (recorder) {
      recorder.trackTemporaryBuffer(activatedOutput);
    } else {
      releaseBuffer(activatedOutput);
    }

    return output;
  }

  // Fallback: separate gate/up path (3 matmuls)
  if (!layerWeights?.gate || !layerWeights?.up || !layerWeights?.down) {
    // Return copy of input (no FFN weights)
    console.warn(`[Layer ${layerIdx}] FFN: no weights found (gateUp=${!!layerWeights?.gateUp}, gate=${!!layerWeights?.gate}, up=${!!layerWeights?.up}, down=${!!layerWeights?.down})`);
    const output = acquireBuffer(numTokens * hiddenSize * 4, undefined, 'ffn_output');
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(inputBuffer, 0, output, 0, numTokens * hiddenSize * 4);
    device.queue.submit([encoder.finish()]);
    return output;
  }

  // 1. Gate projection
  const gateWeight = getWeightBuffer(layerWeights.gate, 'ffn_gate');
  let gateOutput = await doMatmul(inputBuffer, gateWeight, numTokens, intermediateSize, hiddenSize, { transposeB: 'auto', label: `L${layerIdx}.ffn_gate`, layerIdx }, recorder);
  if (!(layerWeights.gate instanceof GPUBuffer)) releaseBuffer(gateWeight);

  const loraGate = getLoRAModule(lora, layerIdx, 'gate_proj');
  if (loraGate) {
    const combined = await applyLoRA(
      inputBuffer,
      gateOutput,
      loraGate,
      { M: numTokens, N: intermediateSize, K: hiddenSize },
      getWeightBuffer,
      recorder
    );
    if (combined !== gateOutput) {
      if (recorder) {
        recorder.trackTemporaryBuffer(gateOutput);
      } else {
        releaseBuffer(gateOutput);
      }
      gateOutput = combined;
    }
  }

  // 2. Up projection
  const upWeight = getWeightBuffer(layerWeights.up, 'ffn_up');
  let upOutput = await doMatmul(inputBuffer, upWeight, numTokens, intermediateSize, hiddenSize, { transposeB: 'auto', label: `L${layerIdx}.ffn_up`, layerIdx }, recorder);
  if (!(layerWeights.up instanceof GPUBuffer)) releaseBuffer(upWeight);

  const loraUp = getLoRAModule(lora, layerIdx, 'up_proj');
  if (loraUp) {
    const combined = await applyLoRA(
      inputBuffer,
      upOutput,
      loraUp,
      { M: numTokens, N: intermediateSize, K: hiddenSize },
      getWeightBuffer,
      recorder
    );
    if (combined !== upOutput) {
      if (recorder) {
        recorder.trackTemporaryBuffer(upOutput);
      } else {
        releaseBuffer(upOutput);
      }
      upOutput = combined;
    }
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(gateOutput, 'ffn_gate', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: intermediateSize,
    });
    await dumpTokenVector(upOutput, 'ffn_up', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: intermediateSize,
    });
  }

  // 3. Activation: activation(gate) * up
  const activationFn = hiddenActivation === 'gelu' ? doGeLU : doSiLU;
  const activatedOutput = await activationFn(upOutput, {
    size: numTokens * intermediateSize,
    gate: gateOutput,
    label: `L${layerIdx}.ffn_activation`,
    layerIdx,
  }, recorder);

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(activatedOutput, 'ffn_activated', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: intermediateSize,
    });
  }

  // Track for cleanup after submit if using recorder, otherwise release immediately
  if (recorder) {
    recorder.trackTemporaryBuffer(gateOutput);
    recorder.trackTemporaryBuffer(upOutput);
  } else {
    releaseBuffer(gateOutput);
    releaseBuffer(upOutput);
  }

  // 4. Down projection
  const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');
  let output = await doMatmul(activatedOutput, downWeight, numTokens, hiddenSize, intermediateSize, { transposeB: 'auto', label: `L${layerIdx}.ffn_down`, layerIdx }, recorder);

  const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
  if (loraDown) {
    const combined = await applyLoRA(
      activatedOutput,
      output,
      loraDown,
      { M: numTokens, N: hiddenSize, K: intermediateSize },
      getWeightBuffer,
      recorder
    );
    if (combined !== output) {
      if (recorder) {
        recorder.trackTemporaryBuffer(output);
      } else {
        releaseBuffer(output);
      }
      output = combined;
    }
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(output, 'ffn_down_out', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: hiddenSize,
    });
  }

  if (!(layerWeights.down instanceof GPUBuffer)) releaseBuffer(downWeight);
  // Track for cleanup after submit if using recorder, otherwise release immediately
  if (recorder) {
    recorder.trackTemporaryBuffer(activatedOutput);
  } else {
    releaseBuffer(activatedOutput);
  }

  return output;
}

/**
 * Run dense FFN with fused down projection + post-FFN norm on GPU.
 *
 * This variant fuses the down projection matmul with the post-FFN RMSNorm
 * into a single kernel, reducing GPU dispatch overhead during decode.
 *
 * Only used when:
 * - numTokens == 1 (decode mode)
 * - hiddenSize <= 4096 (fits in single workgroup)
 * - Has post-FFN norm weights (sandwich norm architecture)
 */
async function runDenseFFNWithFusedPostNormGPU(
  layerIdx: number,
  inputBuffer: GPUBuffer,
  numTokens: number,
  context: LayerContext,
  layerWeights: LayerWeights,
  residual: GPUBuffer,
  eps: number
): Promise<GPUBuffer> {
  const device = getDevice();
  if (!device) throw new Error('No GPU device');

  const { config, weightConfig, debugFlags, recorder } = context;
  const { hiddenSize, intermediateSize, hiddenActivation } = config;
  const lastTokenIdx = Math.max(0, numTokens - 1);
  const lora = context.lora || null;

  // Must have down weights and post-FFN norm, plus either gateUp OR separate gate/up
  if (!layerWeights.down || !layerWeights.postFeedforwardNorm) {
    throw new Error('Missing down or norm weights for fused FFN path');
  }
  const hasFusedGateUp = !!layerWeights.gateUp;
  const hasSeparateGateUp = !!layerWeights.gate && !!layerWeights.up;
  if (!hasFusedGateUp && !hasSeparateGateUp) {
    throw new Error('Missing gate/up weights for fused FFN path');
  }

  const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');
  const normWeightBuf = getNormWeightBuffer(layerWeights.postFeedforwardNorm, 'post_feedforward_norm', weightConfig, debugFlags);

  let activatedOutput: GPUBuffer;

  if (hasFusedGateUp) {
    // Fused gate+up path
    const gateUpWeight = getWeightBuffer(layerWeights.gateUp!, 'ffn_gate_up');

    // 1. Fused gate+up projection
    let gateUpOutput = await doMatmul(
      inputBuffer, gateUpWeight,
      numTokens, intermediateSize * 2, hiddenSize,
      { transposeB: 'auto' },
      recorder
    );

    const loraGateUp = getLoRAModule(lora, layerIdx, 'gate_up_proj');
    if (loraGateUp) {
      const combined = await applyLoRA(
        inputBuffer,
        gateUpOutput,
        loraGateUp,
        { M: numTokens, N: intermediateSize * 2, K: hiddenSize },
        getWeightBuffer,
        recorder
      );
      if (combined !== gateUpOutput) {
        if (recorder) {
          recorder.trackTemporaryBuffer(gateUpOutput);
        } else {
          releaseBuffer(gateUpOutput);
        }
        gateUpOutput = combined;
      }
    }

    if (!(layerWeights.gateUp instanceof GPUBuffer)) releaseBuffer(gateUpWeight);

    // 2. Split + Activation
    const activation = hiddenActivation === 'gelu' ? 'gelu' : 'silu';
    activatedOutput = await doSiLURowSplit(gateUpOutput, {
      numTokens,
      dim: intermediateSize,
      activation,
    }, recorder);

    if (recorder) {
      recorder.trackTemporaryBuffer(gateUpOutput);
    } else {
      releaseBuffer(gateUpOutput);
    }
  } else {
    // Separate gate/up path
    const gateWeight = getWeightBuffer(layerWeights.gate!, 'ffn_gate');
    const upWeight = getWeightBuffer(layerWeights.up!, 'ffn_up');

    // 1a. Gate projection
    const gateOutput = await doMatmul(
      inputBuffer, gateWeight,
      numTokens, intermediateSize, hiddenSize,
      { transposeB: 'auto' },
      recorder
    );
    if (!(layerWeights.gate instanceof GPUBuffer)) releaseBuffer(gateWeight);

    // 1b. Up projection
    const upOutput = await doMatmul(
      inputBuffer, upWeight,
      numTokens, intermediateSize, hiddenSize,
      { transposeB: 'auto' },
      recorder
    );
    if (!(layerWeights.up instanceof GPUBuffer)) releaseBuffer(upWeight);

    // 2. Activation: activation(gate) * up
    // The activation function handles both the activation and element-wise multiply
    const activationFn = hiddenActivation === 'gelu' ? doGeLU : doSiLU;
    activatedOutput = await activationFn(upOutput, {
      size: numTokens * intermediateSize,
      gate: gateOutput,
    }, recorder);

    if (recorder) {
      recorder.trackTemporaryBuffer(gateOutput);
      recorder.trackTemporaryBuffer(upOutput);
    } else {
      releaseBuffer(gateOutput);
      releaseBuffer(upOutput);
    }
  }

  // 3. FUSED: Down projection + RMSNorm + Residual (single kernel!)
  // This is the key optimization: combines matmul + norm + residual into one dispatch
  const output = await doMatmulRMSNormFused(
    activatedOutput,
    downWeight,
    normWeightBuf,
    {
      N: hiddenSize,
      K: intermediateSize,
      eps,
      residual,
    },
    recorder
  );

  // Apply LoRA to output if needed (rare case)
  const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
  if (loraDown) {
    // LoRA needs separate application - for now, fall back to non-fused for LoRA
    // This is a rare case during fine-tuning only
    console.warn(`[Layer ${layerIdx}] LoRA down_proj with fused kernel not yet optimized`);
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(output, 'ffn_fused_out', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: hiddenSize,
    });
  }

  if (!(layerWeights.down instanceof GPUBuffer)) releaseBuffer(downWeight);
  if (!(layerWeights.postFeedforwardNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);
  if (recorder) {
    recorder.trackTemporaryBuffer(activatedOutput);
  } else {
    releaseBuffer(activatedOutput);
  }

  return output;
}

/**
 * Run MoE FFN on GPU.
 */
async function runMoEFFNGPU(
  layerIdx: number,
  inputBuffer: GPUBuffer,
  numTokens: number,
  context: LayerContext
): Promise<GPUBuffer> {
  const { config, moeRouter, expertWeights, expertLoader, layerRouterWeights } = context;

  if (!moeRouter || !expertWeights || !expertLoader) {
    throw new Error('MoE components not initialized');
  }

  // Import dynamically to avoid circular dependency
  const { moeFeedForwardGPU } = await import('./moe-impl.js');

  return moeFeedForwardGPU(
    inputBuffer,
    numTokens,
    {
      hiddenSize: config.hiddenSize,
      intermediateSize: config.intermediateSize,
      numExperts: config.numExperts || 8,
      moeTopK: config.moeTopK || 2,
      hiddenActivation: config.hiddenActivation,
    },
    moeRouter,
    expertWeights,
    expertLoader,
    layerIdx,
    layerRouterWeights as Map<number, import('./moe-impl.js').LayerRouterWeights> | undefined
  );
}

// ============================================================================
// CPU Fallback
// ============================================================================

/**
 * CPU fallback layer processing.
 *
 * This is a simplified version that returns zeros since full CPU inference
 * is not the primary use case. The main purpose is to allow the pipeline
 * to continue even when GPU is unavailable.
 */
export async function processLayerCPU(
  layerIdx: number,
  hiddenStates: Float32Array,
  numTokens: number,
  isPrefill: boolean,
  context: LayerContext
): Promise<Float32Array> {
  const { config } = context;
  const { hiddenSize } = config;

  // CPU fallback - return copy of input (simplified)
  // Full CPU inference would require implementing all operations on CPU
  console.warn(`[Layer ${layerIdx}] CPU fallback - returning input unchanged`);
  return new Float32Array(hiddenStates);
}
