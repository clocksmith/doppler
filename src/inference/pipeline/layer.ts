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
import type { ProbeConfigSchema } from '../../config/schema/index.js';
import type { ExpertWeights, LayerWeights, MaybeGPUBuffer, RouterWeights } from './types.js';
import type { KVCache, SlidingWindowKVCache } from '../kv-cache.js';
import type { ExpertLoader } from './moe-impl.js';
import type { MoERouter } from '../moe-router.js';
import { log, trace } from '../../debug/index.js';
import { getDevice } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../gpu/buffer-pool.js';
import { allowReadback } from '../../gpu/perf-guards.js';
import {
  runRMSNorm, runResidualAdd, runMatmul, runSiLU, runGeLU,
  recordRMSNorm, recordResidualAdd, recordMatmul, recordSiLU, recordGeLU,
  runSiLURowSplit, recordSiLURowSplit,
  runFusedFFN, recordFusedFFN,
  runMatmulRMSNormFused, recordMatmulRMSNormFused, shouldUseFusedMatmulRMSNorm,
  runMatmulResidualFused, recordMatmulResidualFused, shouldUseFusedMatmulResidual,
  CommandRecorder,
  type SiLURowSplitOptions,
} from '../../gpu/kernel-selector.js';
import { Tensor, createTensor, type TensorDtype } from '../../gpu/tensor.js';
import { type WeightBuffer, type CpuWeightBuffer, isWeightBuffer, getBuffer, getLayout, getWeightDtype } from '../../gpu/weight-buffer.js';
import { runLayerAttentionGPU, recordLayerAttentionGPU, type AttentionConfig, type AttentionState, type AttentionDebugFlags, type AttentionResult } from './attention.js';
import { getWeightBuffer, getNormWeightBuffer, type WeightBufferConfig, type WeightDebugFlags } from './weights.js';
import { logLayer, logAttn, logFFN, getBufferStats, isKernelDebugEnabled, dumpTokenVector, logKernelStep } from './debug-utils.js';
import { runProbes } from './probes.js';
import type { CompiledLayerPipeline } from './layer-plan.js';
import { getLayerPlanSteps } from './layer-plan.js';
import { applyLoRA } from './lora-apply.js';
import { getLoRAModule, type LoRAAdapter } from './lora.js';
import { kernelTrace, traceStep } from './kernel-trace.js';
import type { DecodeBufferManager } from '../decode-buffers.js';

// Track if we've logged one-time messages (avoid spam)
let loggedFusedDownNorm = false;

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
  weights: Map<string, LayerWeights | Float32Array | GPUBuffer | WeightBuffer | CpuWeightBuffer>;
  /** KV cache instance */
  kvCache: KVCache | SlidingWindowKVCache;
  /** Current sequence length */
  currentSeqLen: number;
  /** Whether to use GPU */
  useGPU: boolean;
  /** Debug mode */
  debug: boolean;
  /** Config-driven probes */
  debugProbes?: ProbeConfigSchema[];
  /** Optional layer pipeline plan (JSON-configured) */
  pipelinePlan?: CompiledLayerPipeline | null;
  /** RoPE frequency buffers (global for full_attention layers) */
  ropeFreqsCos: GPUBuffer | Float32Array | null;
  ropeFreqsSin: GPUBuffer | Float32Array | null;
  /** Local RoPE frequency buffers for sliding_attention layers (Gemma 3: 10K theta) */
  ropeLocalCos?: GPUBuffer | Float32Array | null;
  ropeLocalSin?: GPUBuffer | Float32Array | null;
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
  /** Pre-allocated decode buffers (for M=1 decode optimization) */
  decodeBuffers?: DecodeBufferManager | null;
  /** Activation dtype for hidden states (default: 'f32', experimental: 'f16') */
  activationDtype?: 'f16' | 'f32';
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
 * Input and residual are Tensor, returns Tensor.
 */
async function doRMSNorm(
  input: Tensor,
  weight: GPUBuffer,
  eps: number,
  options: { batchSize: number; hiddenSize: number; residual?: Tensor | null; outputBuffer?: GPUBuffer | null; label?: string; layerIdx?: number },
  recorder?: CommandRecorder
): Promise<Tensor> {
  const result = recorder
    ? await recordRMSNorm(recorder, input, weight, eps, options)
    : await runRMSNorm(input, weight, eps, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder) {
    const layer = options.layerIdx ?? -1;
    const label = options.label ?? 'rmsnorm';
    await traceStep('rmsnorm', label, layer, result.buffer, [options.batchSize, options.hiddenSize]);
  }

  return result;
}

/**
 * ResidualAdd that uses record variant when recorder is provided.
 * Accepts Tensor for inputs, returns Tensor.
 * The residual kernel still operates on raw GPUBuffers internally.
 */
async function doResidualAdd(
  a: Tensor,
  b: Tensor,
  size: number,
  recorder?: CommandRecorder,
  traceOptions?: { label?: string; layerIdx?: number; outputBuffer?: GPUBuffer | null }
): Promise<Tensor> {
  const options = traceOptions?.outputBuffer ? { outputBuffer: traceOptions.outputBuffer } : {};
  const result = recorder
    ? await recordResidualAdd(recorder, a, b, size, options)
    : await runResidualAdd(a, b, size, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder && traceOptions) {
    await traceStep('residual_add', traceOptions.label ?? 'residual', traceOptions.layerIdx ?? -1, result.buffer, [size]);
  }

  return result;
}

/**
 * Matmul that uses record variant when recorder is provided.
 * A is activation Tensor, B is weight (GPUBuffer or WeightBuffer), returns Tensor.
 */
async function doMatmul(
  A: Tensor,
  B: GPUBuffer | WeightBuffer,
  M: number,
  N: number,
  K: number,
  options: { transposeB?: boolean | 'auto'; label?: string; layerIdx?: number; outputDtype?: 'f16' | 'f32'; role?: string } = {},
  recorder?: CommandRecorder
): Promise<Tensor> {
  const result = recorder
    ? await recordMatmul(recorder, A, B, M, N, K, options)
    : await runMatmul(A, B, M, N, K, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder) {
    const layer = options.layerIdx ?? -1;
    const label = options.label ?? 'matmul';
    await traceStep('matmul', label, layer, result.buffer, [M, N]);
  }

  return result;
}

/**
 * SiLU that uses record variant when recorder is provided.
 * Supports gated variant (SiLU with gate multiplication).
 * Input and gate are Tensor, returns Tensor.
 */
async function doSiLU(
  input: Tensor,
  options: { size?: number; gate?: Tensor | null; label?: string; layerIdx?: number } = {},
  recorder?: CommandRecorder
): Promise<Tensor> {
  const result = recorder
    ? await recordSiLU(recorder, input, options)
    : await runSiLU(input, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder && options.size) {
    await traceStep('silu', options.label ?? 'silu', options.layerIdx ?? -1, result.buffer, [options.size]);
  }

  return result;
}

/**
 * GeLU that uses record variant when recorder is provided.
 * Supports gated variant (GeGLU).
 * Input and gate are Tensor, returns Tensor.
 */
async function doGeLU(
  input: Tensor,
  options: { size?: number; gate?: Tensor | null; label?: string; layerIdx?: number } = {},
  recorder?: CommandRecorder
): Promise<Tensor> {
  const result = recorder
    ? await recordGeLU(recorder, input, options)
    : await runGeLU(input, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder && options.size) {
    await traceStep('gelu', options.label ?? 'gelu', options.layerIdx ?? -1, result.buffer, [options.size]);
  }

  return result;
}

/**
 * SiLURowSplit that uses record variant when recorder is provided.
 * Used for fused gate+up FFN path: splits combined output and applies activation.
 * Input is Tensor, returns Tensor.
 */
async function doSiLURowSplit(
  input: Tensor,
  options: Omit<SiLURowSplitOptions, 'activationDtype'> & { label?: string; layerIdx?: number },
  recorder?: CommandRecorder
): Promise<Tensor> {
  const result = recorder
    ? await recordSiLURowSplit(recorder, input, options)
    : await runSiLURowSplit(input, options);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder) {
    await traceStep('silu_row_split', options.label ?? 'ffn_activation', options.layerIdx ?? -1, result.buffer, [options.numTokens, options.dim]);
  }

  return result;
}

/**
 * Fused Matmul + RMSNorm that uses record variant when recorder is provided.
 * Used for down projection + post-FFN norm fusion during decode (M=1).
 * Input is Tensor, residual is Tensor, returns Tensor.
 */
async function doMatmulRMSNormFused(
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  normWeight: GPUBuffer,
  options: { N: number; K: number; eps: number; residual?: Tensor | null; outputBuffer?: GPUBuffer | null; transposeB?: boolean; label?: string; layerIdx?: number },
  recorder?: CommandRecorder
): Promise<Tensor> {
  // The fused kernel takes Tensor input but residual is still GPUBuffer
  const fusedOptions = {
    N: options.N,
    K: options.K,
    eps: options.eps,
    residual: options.residual?.buffer ?? null,
    outputBuffer: options.outputBuffer,
    transposeB: options.transposeB,
  };
  const resultTensor = recorder
    ? await recordMatmulRMSNormFused(recorder, input, weight, normWeight, fusedOptions)
    : await runMatmulRMSNormFused(input, weight, normWeight, fusedOptions);

  // Trace the kernel output
  if (kernelTrace.enabled && !recorder) {
    await traceStep('fused_matmul_rmsnorm', options.label ?? 'fused_matmul_rmsnorm', options.layerIdx ?? -1, resultTensor.buffer, [1, options.N]);
  }

  return resultTensor;
}

/**
 * Attention that uses record variant when recorder is provided.
 * Input is Tensor for dtype-aware processing.
 */
async function doAttention(
  inputTensor: Tensor,
  layerWeights: LayerWeights | null,
  config: AttentionConfig,
  state: AttentionState,
  debug: boolean,
  debugFlags: AttentionDebugFlags,
  getWeightBufferFn: (weight: GPUBuffer | WeightBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer | WeightBuffer,
  getNormWeightBufferFn: (weight: GPUBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer,
  recorder?: CommandRecorder,
  lora?: LoRAAdapter | null
): Promise<AttentionResult> {
  if (recorder) {
    return recordLayerAttentionGPU(
      recorder,
      inputTensor,
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
    inputTensor,
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
    trace.ffn(0, `routing: useGPU=${useGPU}, isGPUBuffer=${hiddenStates instanceof GPUBuffer}, constructor=${hiddenStates?.constructor?.name}`);
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
 * Internally uses Tensor for dtype-aware pipeline, returns GPUBuffer for compatibility.
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

  const { config, weights, weightConfig, debugFlags, kvCache, ropeFreqsCos, ropeFreqsSin, recorder } = context;
  const { hiddenSize, numHeads, numKVHeads, headDim, rmsNormEps } = config;

  // Determine activation dtype from context (defaults to f32)
  const activationDtype: TensorDtype = context.activationDtype === 'f16' ? 'f16' : 'f32';

  // Wrap input buffer as Tensor for dtype-aware processing
  const inputTensor = createTensor(inputBuffer, activationDtype, [numTokens, hiddenSize], 'layer_input');

  const layerWeights = weights.get(`layer_${layerIdx}`) as LayerWeights | undefined;
  const sandwichNorm = detectSandwichNorm(layerWeights);
  const lastTokenIdx = Math.max(0, numTokens - 1);

  if (context.pipelinePlan) {
    return processLayerPlanGPU(layerIdx, inputBuffer, numTokens, isPrefill, size, context, layerWeights, sandwichNorm);
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    logKernelStep('layer', { layerIdx, label: `seqLen=${numTokens} prefill=${isPrefill}` });
    await dumpTokenVector(inputBuffer, 'layer_in', { layerIdx, tokenIdx: lastTokenIdx, rowSize: hiddenSize });
  }

  // 1. Self-attention (returns GPU buffer - attention module not yet migrated to Tensor)
  // Determine layer type for RoPE frequency selection (Gemma 3: sliding vs full attention)
  const layerType = config.layerTypes?.[layerIdx];
  const isLocalLayer = layerType === 'sliding_attention';

  // Debug: log RoPE selection for first few layers
  if (context.debug && layerIdx < 3) {
    trace.attn(layerIdx, `RoPE selection: layerType=${layerType}, isLocal=${isLocalLayer}, hasLocalCos=${!!context.ropeLocalCos}, hasLocalSin=${!!context.ropeLocalSin}`);
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
    // Pass residual tensor for decode mode to enable fused o_proj + residual.
    // For sandwich-norm models with post-attention norm (Gemma 2/3),
    // we must NOT fuse residual into attention output because HF expects:
    // residual + norm(attn_output), not norm(attn_output + residual).
    residualTensor: (numTokens === 1 && !(sandwichNorm.useSandwichNorm && sandwichNorm.hasPostAttentionNorm))
      ? inputTensor
      : null,
    // Gemma 2 attention softcapping (50.0)
    attnSoftcap: config.attnLogitSoftcapping ?? 0,
    // Gemma 2 attention scaling: uses head_dim (256) instead of sqrt(head_dim) (16)
    queryPreAttnScalar: config.queryPreAttnScalar,
    queryKeyNorm: config.queryKeyNorm,
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

  const attnResult = await doAttention(
    inputTensor,
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
  // Attention now returns Tensor directly
  const attnOutput = attnResult.output;
  const residualFused = attnResult.residualFused;

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(attnOutput.buffer, 'attn_out', { layerIdx, tokenIdx: lastTokenIdx, rowSize: hiddenSize });
  }

  // Debug: trace attn output (uses debug-utils)
  if (context.debug) {
    const stats = await getBufferStats(attnOutput.buffer);
    if (stats) logAttn(layerIdx, isPrefill, { numTokens, kvLen: context.currentSeqLen + (isPrefill ? numTokens : 1), maxAbsOut: stats.maxAbs });

    // Debug layer 0, 2, and 17 attention output specifically (2 is where decode explosion happens)
    // NOTE: Skip detailed readback when using recorder (batched mode) - buffers aren't populated yet!
    trace.attn(layerIdx, `attnOutput type check: isGPU=${attnOutput.buffer instanceof GPUBuffer}, type=${typeof attnOutput.buffer}, constructor=${attnOutput.buffer?.constructor?.name}, isPrefill=${isPrefill}`);
    if ((layerIdx === 0 || layerIdx === 2 || layerIdx === 17) && attnOutput.buffer instanceof GPUBuffer && !recorder) {
      if (allowReadback(`layer.attn-out.${layerIdx}`)) {
        try {
          const sampleSize = Math.min(128, attnOutput.buffer.size);
          const staging = device.createBuffer({ size: sampleSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          const enc = device.createCommandEncoder();
          enc.copyBufferToBuffer(attnOutput.buffer, 0, staging, 0, sampleSize);
          device.queue.submit([enc.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          const data = new Float32Array(staging.getMappedRange().slice(0));
          staging.unmap();
          staging.destroy();
          const maxAbs = Math.max(...Array.from(data).map(x => Math.abs(x)));
          const nonZero = Array.from(data).filter(x => x !== 0).length;
          trace.attn(layerIdx, `ATTN_OUT: maxAbs=${maxAbs.toFixed(4)}, nonZero=${nonZero}/${data.length}, sample=[${Array.from(data).slice(0, 5).map(x => x.toFixed(4)).join(', ')}]`);
        } catch (e) {
          trace.attn(layerIdx, `ATTN_OUT error: ${e}`);
        }
      }
    } else if ((layerIdx === 0 || layerIdx === 2 || layerIdx === 17) && attnOutput.buffer instanceof GPUBuffer && recorder) {
      trace.attn(layerIdx, `ATTN_OUT: (skipped - using batched recorder, values not available until submit)`);
    }
  }
  await runProbes('attn_out', attnOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  // 2. Handle residual connection based on architecture
  let postAttn: Tensor;
  if (residualFused) {
    // Fused path: residual was already added in attention o_proj
    // attnOutput already contains the residual, no need to add it again
    postAttn = attnOutput;
    // Note: For sandwich norm models, we may still need post-attention norm
    // but without adding residual again (residual is already in attnOutput)
    if (sandwichNorm.useSandwichNorm && sandwichNorm.hasPostAttentionNorm && layerWeights?.postAttentionNorm) {
      const normWeightBuf = getNormWeightBuffer(layerWeights.postAttentionNorm, 'post_attention_norm', weightConfig, debugFlags);
      postAttn = await doRMSNorm(attnOutput, normWeightBuf, rmsNormEps, {
        batchSize: numTokens,
        hiddenSize,
        // No residual - it's already fused into attnOutput
        label: `L${layerIdx}.post_attn_norm`,
        layerIdx,
      }, recorder);
      if (!(layerWeights.postAttentionNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);
      if (recorder) {
        recorder.trackTemporaryBuffer(attnOutput.buffer);
      } else {
        releaseBuffer(attnOutput.buffer);
      }
    }
  } else if (sandwichNorm.useSandwichNorm && sandwichNorm.hasPostAttentionNorm && layerWeights?.postAttentionNorm) {
    // Gemma 3 path: FUSED norm attention output + residual add (1 kernel instead of 2)
    const normWeightBuf = getNormWeightBuffer(layerWeights.postAttentionNorm, 'post_attention_norm', weightConfig, debugFlags);
    postAttn = await doRMSNorm(attnOutput, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      residual: inputTensor,  // FUSION: Add residual in same kernel
      label: `L${layerIdx}.post_attn_norm`,
      layerIdx,
    }, recorder);

    if (!(layerWeights.postAttentionNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);
    // Track for cleanup after submit if using recorder, otherwise release immediately
    if (recorder) {
      recorder.trackTemporaryBuffer(attnOutput.buffer);
    } else {
      releaseBuffer(attnOutput.buffer);
    }
  } else {
    // Standard path: residual add first
    postAttn = await doResidualAdd(attnOutput, inputTensor, size, recorder, { label: `L${layerIdx}.post_attn_residual`, layerIdx });
    // Track for cleanup after submit if using recorder, otherwise release immediately
    if (recorder) {
      recorder.trackTemporaryBuffer(attnOutput.buffer);
    } else {
      releaseBuffer(attnOutput.buffer);
    }
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(postAttn.buffer, 'x_after_attn', { layerIdx, tokenIdx: lastTokenIdx, rowSize: hiddenSize });
  }

  await runProbes('post_attn', postAttn.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  // 3. Feed-forward network - returns Tensor, extract buffer for return
  let outputTensor: Tensor;
  if (sandwichNorm.useSandwichNorm) {
    outputTensor = await processFFNWithSandwichNorm(layerIdx, postAttn, numTokens, size, context, layerWeights, sandwichNorm);
  } else {
    outputTensor = await processFFNStandard(layerIdx, postAttn, numTokens, size, context, layerWeights);
  }

  return outputTensor.buffer;
}

// ============================================================================
// Configurable Layer Pipeline (JSON-Driven)
// ============================================================================

function resolveNormWeightForPlan(
  weight: 'input' | 'post_attention' | 'post_attn' | 'pre_ffn' | 'post_ffn',
  layerWeights: LayerWeights | undefined
): GPUBuffer | Float32Array | null {
  if (!layerWeights) return null;
  switch (weight) {
    case 'input':
      return layerWeights.inputNorm;
    case 'post_attention':
      return layerWeights.postAttentionNorm ?? layerWeights.postAttnNorm ?? null;
    case 'post_attn':
      return layerWeights.postAttnNorm ?? layerWeights.postAttentionNorm ?? null;
    case 'pre_ffn':
      return layerWeights.preFeedforwardNorm ?? null;
    case 'post_ffn':
      return layerWeights.postFeedforwardNorm ?? null;
    default:
      return null;
  }
}

async function processLayerPlanGPU(
  layerIdx: number,
  inputBuffer: GPUBuffer,
  numTokens: number,
  isPrefill: boolean,
  size: number,
  context: LayerContext,
  layerWeights: LayerWeights | undefined,
  sandwichNorm: SandwichNormInfo
): Promise<GPUBuffer> {
  const { config, weightConfig, debugFlags, kvCache, ropeFreqsCos, ropeFreqsSin, recorder } = context;
  const { hiddenSize, numHeads, numKVHeads, headDim, rmsNormEps } = config;

  if (!context.pipelinePlan) {
    throw new Error('Layer pipeline plan missing from context');
  }

  const steps = getLayerPlanSteps(context.pipelinePlan, layerIdx);
  const device = getDevice();
  if (!device) throw new Error('No GPU device available');

  const layerType = config.layerTypes?.[layerIdx];
  const isLocalLayer = layerType === 'sliding_attention';
  const attnState: AttentionState = {
    ropeFreqsCos: (isLocalLayer && context.ropeLocalCos)
      ? context.ropeLocalCos as GPUBuffer | null
      : ropeFreqsCos as GPUBuffer | null,
    ropeFreqsSin: (isLocalLayer && context.ropeLocalSin)
      ? context.ropeLocalSin as GPUBuffer | null
      : ropeFreqsSin as GPUBuffer | null,
    kvCache: kvCache as unknown as import('./types.js').KVCacheInterface,
  };

  const allowResidualFuse = numTokens === 1 && !(sandwichNorm.useSandwichNorm && sandwichNorm.hasPostAttentionNorm);

  const slots = new Map<string, GPUBuffer>();
  const refCounts = new Map<GPUBuffer, number>();
  const protectedBuffers = new Set<GPUBuffer>([inputBuffer]);

  const addRef = (buf: GPUBuffer): void => {
    refCounts.set(buf, (refCounts.get(buf) ?? 0) + 1);
  };
  const releaseRef = (buf: GPUBuffer): void => {
    const next = (refCounts.get(buf) ?? 0) - 1;
    if (next > 0) {
      refCounts.set(buf, next);
      return;
    }
    refCounts.delete(buf);
    if (protectedBuffers.has(buf)) return;
    if (recorder) {
      recorder.trackTemporaryBuffer(buf);
    } else {
      releaseBuffer(buf);
    }
  };
  const getSlot = (name: string): GPUBuffer => {
    const key = name.trim() || 'state';
    const buf = slots.get(key);
    if (!buf) {
      throw new Error(`Layer pipeline missing slot "${key}" at L${layerIdx}`);
    }
    return buf;
  };
  const setSlot = (name: string, buf: GPUBuffer): void => {
    const key = name.trim() || 'state';
    const prev = slots.get(key);
    if (prev && prev !== buf) {
      releaseRef(prev);
    }
    slots.set(key, buf);
    addRef(buf);
  };
  const clearSlot = (name: string): void => {
    const key = name.trim() || 'state';
    const prev = slots.get(key);
    if (!prev) return;
    slots.delete(key);
    releaseRef(prev);
  };

  setSlot('state', inputBuffer);

  // Helper to clean up all slots except 'state' (used in finally for exception safety)
  const cleanupSlots = (): void => {
    for (const [name, buf] of slots) {
      if (name === 'state' || protectedBuffers.has(buf)) continue;
      const refs = refCounts.get(buf) ?? 0;
      if (refs > 0) {
        refCounts.delete(buf);
        if (recorder) {
          recorder.trackTemporaryBuffer(buf);
        } else {
          releaseBuffer(buf);
        }
      }
    }
  };

  try {
    for (const step of steps) {
      switch (step.op) {
        case 'save': {
          const src = getSlot(step.src);
          setSlot(step.name!, src);
          break;
        }
        case 'load': {
          const src = getSlot(step.name!);
          setSlot(step.dst, src);
          break;
        }
        case 'attention': {
          const srcBuf = getSlot(step.src);
          const residualBuf = step.residual ? getSlot(step.residual) : null;

          // Wrap GPUBuffer in Tensor for doAttention
          const activationDtype: TensorDtype = context.activationDtype === 'f16' ? 'f16' : 'f32';
          const srcTensor = createTensor(srcBuf, activationDtype, [numTokens, hiddenSize], 'plan_attn_src');
          const residualTensor = (residualBuf && allowResidualFuse)
            ? createTensor(residualBuf, activationDtype, [numTokens, hiddenSize], 'plan_attn_residual')
            : null;

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
            residualTensor,
            attnSoftcap: config.attnLogitSoftcapping ?? 0,
            queryPreAttnScalar: config.queryPreAttnScalar,
            queryKeyNorm: config.queryKeyNorm,
            skipInputNorm: step.skipInputNorm === true,
          };

          const result = await doAttention(
            srcTensor,
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

          // Attention result.output is now a Tensor - extract buffer for slot storage
          setSlot(step.dst, result.output.buffer);
          if (step.probeStage) {
            await runProbes(step.probeStage, result.output.buffer, {
              layerIdx,
              numTokens,
              hiddenSize,
              probes: context.debugProbes,
              recorder,
            });
          }
          break;
        }
        case 'rmsnorm': {
          const srcBuf = getSlot(step.src);
          const weight = resolveNormWeightForPlan(step.weight!, layerWeights);
          if (!weight) {
            throw new Error(`Layer pipeline rmsnorm missing weights for "${step.weight}" at L${layerIdx}`);
          }
          const normWeightBuf = getNormWeightBuffer(weight, `rmsnorm_${step.weight}`, weightConfig, debugFlags);
          const residualBuf = step.residual ? getSlot(step.residual) : null;
          // Wrap GPUBuffers in Tensor for doRMSNorm
          const activationDtype: TensorDtype = context.activationDtype === 'f16' ? 'f16' : 'f32';
          const srcTensor = createTensor(srcBuf, activationDtype, [numTokens, hiddenSize], 'plan_rmsnorm_src');
          const residualTensor = residualBuf ? createTensor(residualBuf, activationDtype, [numTokens, hiddenSize], 'plan_rmsnorm_residual') : null;
          const outputTensor = await doRMSNorm(srcTensor, normWeightBuf, rmsNormEps, {
            batchSize: numTokens,
            hiddenSize,
            residual: residualTensor,
            label: `L${layerIdx}.rmsnorm_${step.weight}`,
            layerIdx,
          }, recorder);
          if (!(weight instanceof GPUBuffer)) releaseBuffer(normWeightBuf);
          setSlot(step.dst, outputTensor.buffer);
          if (step.probeStage) {
            await runProbes(step.probeStage, outputTensor.buffer, {
              layerIdx,
              numTokens,
              hiddenSize,
              probes: context.debugProbes,
              recorder,
            });
          }
          break;
        }
        case 'ffn': {
          const srcBuf = getSlot(step.src);
          // Wrap GPUBuffer in Tensor for FFN functions
          const activationDtype: TensorDtype = context.activationDtype === 'f16' ? 'f16' : 'f32';
          const srcTensor = createTensor(srcBuf, activationDtype, [numTokens, hiddenSize], 'plan_ffn_src');
          let outputTensor: Tensor;
          const useMoe = step.variant === 'moe'
            || (step.variant === 'auto' && config.useMoE && isMoELayer(layerIdx, config, layerWeights));
          if (useMoe) {
            outputTensor = await runMoEFFNGPU(layerIdx, srcTensor, numTokens, context);
          } else {
            outputTensor = await runDenseFFNGPU(layerIdx, srcTensor, numTokens, context, layerWeights);
          }
          setSlot(step.dst, outputTensor.buffer);
          if (step.probeStage) {
            await runProbes(step.probeStage, outputTensor.buffer, {
              layerIdx,
              numTokens,
              hiddenSize,
              probes: context.debugProbes,
              recorder,
            });
          }
          break;
        }
        case 'residual_add': {
          const aBuf = getSlot(step.a ?? 'state');
          const bBuf = getSlot(step.b ?? 'residual');
          // Wrap GPUBuffers in Tensor for doResidualAdd
          const activationDtype: TensorDtype = context.activationDtype === 'f16' ? 'f16' : 'f32';
          const aTensor = createTensor(aBuf, activationDtype, [numTokens, hiddenSize], 'plan_residual_a');
          const bTensor = createTensor(bBuf, activationDtype, [numTokens, hiddenSize], 'plan_residual_b');
          const outputTensor = await doResidualAdd(aTensor, bTensor, size, recorder, {
            label: `L${layerIdx}.residual_add`,
            layerIdx,
          });
          setSlot(step.dst, outputTensor.buffer);
          if (step.probeStage) {
            await runProbes(step.probeStage, outputTensor.buffer, {
              layerIdx,
              numTokens,
              hiddenSize,
              probes: context.debugProbes,
              recorder,
            });
          }
          break;
        }
        case 'noop':
          break;
        default:
          throw new Error(`Unknown layer pipeline op "${step.op}" at L${layerIdx}`);
      }
    }

    // Normal cleanup: release all slots except 'state'
    for (const name of Array.from(slots.keys())) {
      if (name !== 'state') {
        clearSlot(name);
      }
    }
  } catch (err) {
    // On error, clean up all allocated buffers to prevent leaks
    cleanupSlots();
    throw err;
  }

  const output = getSlot('state');
  await runProbes('layer_out', output, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  return output;
}

/**
 * Process FFN with sandwich norm architecture (Gemma 3).
 * Input and output are Tensor for dtype-aware processing.
 */
async function processFFNWithSandwichNorm(
  layerIdx: number,
  postAttn: Tensor,
  numTokens: number,
  size: number,
  context: LayerContext,
  layerWeights: LayerWeights | undefined,
  sandwichNorm: SandwichNormInfo
): Promise<Tensor> {
  const { config, weightConfig, debugFlags, recorder, decodeBuffers } = context;
  const { hiddenSize, rmsNormEps } = config;

  // For decode (M=1), get pre-allocated output buffer to avoid allocation
  const decodeOutputBuffer = numTokens === 1 && decodeBuffers
    ? decodeBuffers.getOutputHiddenBuffer()
    : null;
  const lastTokenIdx = Math.max(0, numTokens - 1);

  // 1. Pre-FFN norm (applied to residual stream before FFN)
  let ffnInput = postAttn;
  if (sandwichNorm.hasPreFeedforwardNorm && layerWeights?.preFeedforwardNorm) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.preFeedforwardNorm, 'pre_feedforward_norm', weightConfig, debugFlags);

    ffnInput = await doRMSNorm(postAttn, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      label: `L${layerIdx}.pre_ffn_norm`,
      layerIdx,
    }, recorder);
    if (!(layerWeights.preFeedforwardNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);
  }

  await runProbes('ffn_in', ffnInput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(ffnInput.buffer, 'pre_ffn_norm_out', { layerIdx, tokenIdx: lastTokenIdx, rowSize: hiddenSize });
  }

  // 2. FFN (or MoE FFN)
  // Check if we can use fused down+norm kernel (decode only, dense FFN with post-FFN norm)
  // The fused kernel now supports both column-major and row-major (transposed) weights.
  // BUT: the fused kernel only supports F32 weights - not Q4K or F16 quantized weights.
  const downWeight = layerWeights?.down;
  // Note: getLayout/getWeightDtype handle both WeightBuffer and raw GPUBuffer
  // Float32Array means CPU fallback (row-major, f32)
  const downWeightIsColumnMajor = downWeight && !(downWeight instanceof Float32Array)
    ? getLayout(downWeight) === 'column'
    : false;

  // Check weight dtype - fused kernel requires F32 weights
  const downWeightDtype = downWeight && !(downWeight instanceof Float32Array)
    ? getWeightDtype(downWeight)
    : 'f32';
  const downWeightIsF32 = downWeightDtype === 'f32' || downWeightDtype === null;

  const canUseFusedDownNorm = numTokens === 1
    && !config.useMoE
    && !isMoELayer(layerIdx, config, layerWeights)
    && sandwichNorm.hasPostFeedforwardNorm
    && layerWeights?.postFeedforwardNorm
    && layerWeights?.down
    && ffnInput.dtype === 'f32'
    && downWeightIsF32  // Fused kernel only supports F32, not Q4K/F16
    && shouldUseFusedMatmulRMSNorm(numTokens, hiddenSize);
  // Note: transposeB=true for row-major GGUF weights, transposeB=false for column-major

  let ffnOutput: Tensor;
  let usedFusedDownNorm = false;

  if (config.useMoE && isMoELayer(layerIdx, config, layerWeights)) {
    ffnOutput = await runMoEFFNGPU(layerIdx, ffnInput, numTokens, context);
  } else if (canUseFusedDownNorm && layerWeights?.down && layerWeights?.postFeedforwardNorm &&
             (layerWeights?.gateUp || (layerWeights?.gate && layerWeights?.up))) {
    // FUSED PATH: gate+up (or separate gate/up) -> activation -> down+norm+residual (single kernel for last step)
    if (layerIdx === 0 && !loggedFusedDownNorm) {
      trace.ffn(0, `Using fused down+norm kernel (transposeB=${!downWeightIsColumnMajor})`);
      loggedFusedDownNorm = true;
    }
    // Pass pre-allocated decode buffer for output when available
    // transposeB: true if weights are row-major (GGUF default), false if column-major
    ffnOutput = await runDenseFFNWithFusedPostNormGPU(
      layerIdx, ffnInput, numTokens, context, layerWeights,
      postAttn,  // residual for post-FFN norm
      rmsNormEps,
      !downWeightIsColumnMajor,  // transposeB
      decodeOutputBuffer
    );
    usedFusedDownNorm = true;
  } else {
    ffnOutput = await runDenseFFNGPU(layerIdx, ffnInput, numTokens, context, layerWeights);
  }
  await runProbes('ffn_out', ffnOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(ffnOutput.buffer, 'ffn_out', { layerIdx, tokenIdx: lastTokenIdx, rowSize: hiddenSize });
  }

  // Track for cleanup after submit if using recorder, otherwise release immediately
  if (ffnInput !== postAttn) {
    if (recorder) {
      recorder.trackTemporaryBuffer(ffnInput.buffer);
    } else {
      releaseBuffer(ffnInput.buffer);
    }
  }

  // Debug: trace FFN output (uses debug-utils)
  const ffnStats = await getBufferStats(ffnOutput.buffer);
  if (ffnStats) logFFN(layerIdx, { maxAbsOut: ffnStats.maxAbs });

  // 3. Post-FFN norm - applied to FFN output BEFORE residual add
  // Skip if we already used fused down+norm kernel
  let output: Tensor;
  if (usedFusedDownNorm) {
    // Fused kernel already applied norm + residual
    output = ffnOutput;
  } else if (sandwichNorm.hasPostFeedforwardNorm && layerWeights?.postFeedforwardNorm) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.postFeedforwardNorm, 'post_feedforward_norm', weightConfig, debugFlags);

    // FUSED: norm FFN output + residual add (1 kernel instead of 2)
    // Use pre-allocated decode buffer for output when available
    output = await doRMSNorm(ffnOutput, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      residual: postAttn,  // FUSION: Add residual in same kernel
      outputBuffer: decodeOutputBuffer,  // Use pre-allocated buffer for decode
      label: `L${layerIdx}.post_ffn_norm`,
      layerIdx,
    }, recorder);

    if (!(layerWeights.postFeedforwardNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);
    // Track for cleanup after submit if using recorder, otherwise release immediately
    if (recorder) {
      recorder.trackTemporaryBuffer(ffnOutput.buffer);
    } else {
      releaseBuffer(ffnOutput.buffer);
    }
  } else {
    // Standard path: residual add without norm
    // Use pre-allocated decode buffer for output when available
    output = await doResidualAdd(ffnOutput, postAttn, size, recorder, {
      label: `L${layerIdx}.post_ffn_residual`,
      layerIdx,
      outputBuffer: decodeOutputBuffer,
    });
    if (recorder) {
      recorder.trackTemporaryBuffer(ffnOutput.buffer);
    } else {
      releaseBuffer(ffnOutput.buffer);
    }
  }

  await runProbes('layer_out', output.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(output.buffer, 'layer_out', { layerIdx, tokenIdx: lastTokenIdx, rowSize: hiddenSize });
  }

  // Track postAttn for cleanup after submit if using recorder, otherwise release immediately
  if (recorder) {
    recorder.trackTemporaryBuffer(postAttn.buffer);
  } else {
    releaseBuffer(postAttn.buffer);
  }

  return output;
}

/**
 * Process FFN with standard architecture (LLaMA-style).
 * Input and output are Tensor for dtype-aware processing.
 */
async function processFFNStandard(
  layerIdx: number,
  postAttn: Tensor,
  numTokens: number,
  size: number,
  context: LayerContext,
  layerWeights: LayerWeights | undefined
): Promise<Tensor> {
  const { config, weightConfig, debugFlags, recorder, decodeBuffers } = context;
  const { hiddenSize, rmsNormEps } = config;

  // For decode (M=1), get pre-allocated output buffer to avoid allocation
  const decodeOutputBuffer = numTokens === 1 && decodeBuffers
    ? decodeBuffers.getOutputHiddenBuffer()
    : null;

  // 1. Post-attention norm (LLaMA-style pre-FFN norm)
  let normedTensor = postAttn;
  if (layerWeights?.postAttnNorm) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.postAttnNorm, 'post_attn_norm', weightConfig, debugFlags);
    normedTensor = await doRMSNorm(postAttn, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      label: `L${layerIdx}.post_attn_norm`,
      layerIdx,
    }, recorder);
    if (!(layerWeights.postAttnNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);
  }
  await runProbes('ffn_in', normedTensor.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  // 2. FFN (or MoE FFN)
  let ffnOutput: Tensor;
  if (config.useMoE && isMoELayer(layerIdx, config, layerWeights)) {
    ffnOutput = await runMoEFFNGPU(layerIdx, normedTensor, numTokens, context);
  } else {
    ffnOutput = await runDenseFFNGPU(layerIdx, normedTensor, numTokens, context, layerWeights);
  }
  await runProbes('ffn_out', ffnOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  // 3. Residual add: ffnOutput + postAttn
  // Use pre-allocated decode buffer for output when available
  const output = await doResidualAdd(ffnOutput, postAttn, size, recorder, {
    label: `L${layerIdx}.ffn_residual`,
    layerIdx,
    outputBuffer: decodeOutputBuffer,
  });
  await runProbes('layer_out', output.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  // Track for cleanup after submit if using recorder, otherwise release immediately
  if (normedTensor !== postAttn) {
    if (recorder) {
      recorder.trackTemporaryBuffer(normedTensor.buffer);
    } else {
      releaseBuffer(normedTensor.buffer);
    }
  }
  if (recorder) {
    recorder.trackTemporaryBuffer(postAttn.buffer);
    recorder.trackTemporaryBuffer(ffnOutput.buffer);
  } else {
    releaseBuffer(postAttn.buffer);
    releaseBuffer(ffnOutput.buffer);
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
 *
 * Input and output are Tensor for dtype-aware processing.
 */
async function runDenseFFNGPU(
  layerIdx: number,
  inputTensor: Tensor,
  numTokens: number,
  context: LayerContext,
  layerWeights: LayerWeights | undefined
): Promise<Tensor> {
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
    const useF16 = inputTensor.dtype === 'f16';
    let gateUpOutput = await doMatmul(
      inputTensor, gateUpWeight,
      numTokens, intermediateSize * 2, hiddenSize,
      { transposeB: 'auto', label: `L${layerIdx}.ffn_gate_up`, layerIdx, outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_gate_up' },
      recorder
    );

    const loraGateUp = getLoRAModule(lora, layerIdx, 'gate_up_proj');
    if (loraGateUp) {
      const combined = await applyLoRA(
        inputTensor,
        gateUpOutput,
        loraGateUp,
        { M: numTokens, N: intermediateSize * 2, K: hiddenSize },
        getWeightBuffer,
        recorder
      );
      if (combined.buffer !== gateUpOutput.buffer) {
        if (recorder) {
          recorder.trackTemporaryBuffer(gateUpOutput.buffer);
        } else {
          releaseBuffer(gateUpOutput.buffer);
        }
        gateUpOutput = combined;
      }
    }

    if (isKernelDebugEnabled(layerIdx) && !recorder) {
      await dumpTokenVector(gateUpOutput.buffer, 'ffn_gate_up', {
        layerIdx,
        tokenIdx: lastTokenIdx,
        rowSize: intermediateSize * 2,
      });
    }

    if (!(layerWeights.gateUp instanceof GPUBuffer) && !isWeightBuffer(layerWeights.gateUp)) {
      releaseBuffer(isWeightBuffer(gateUpWeight) ? gateUpWeight.buffer : gateUpWeight);
    }

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
      await dumpTokenVector(activatedOutput.buffer, 'ffn_activated', {
        layerIdx,
        tokenIdx: lastTokenIdx,
        rowSize: intermediateSize,
      });
    }

    // Track for cleanup after submit if using recorder, otherwise release immediately
    if (recorder) {
      recorder.trackTemporaryBuffer(gateUpOutput.buffer);
    } else {
      releaseBuffer(gateUpOutput.buffer);
    }

    // 3. Down projection: [numTokens, intermediateSize] @ [hiddenSize, intermediateSize]^T -> [numTokens, hiddenSize]
    let output = await doMatmul(
      activatedOutput, downWeight,
      numTokens, hiddenSize, intermediateSize,
      { transposeB: 'auto', label: `L${layerIdx}.ffn_down`, layerIdx, outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_down' },
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
      if (combined.buffer !== output.buffer) {
        if (recorder) {
          recorder.trackTemporaryBuffer(output.buffer);
        } else {
          releaseBuffer(output.buffer);
        }
        output = combined;
      }
    }

    if (isKernelDebugEnabled(layerIdx) && !recorder) {
      await dumpTokenVector(output.buffer, 'ffn_down_out', {
        layerIdx,
        tokenIdx: lastTokenIdx,
        rowSize: hiddenSize,
      });
    }

    if (!(layerWeights.down instanceof GPUBuffer) && !isWeightBuffer(layerWeights.down)) {
      releaseBuffer(isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
    }
    // Track for cleanup after submit if using recorder, otherwise release immediately
    if (recorder) {
      recorder.trackTemporaryBuffer(activatedOutput.buffer);
    } else {
      releaseBuffer(activatedOutput.buffer);
    }

    return output;
  }

  // Fused FFN kernel path (single dispatch for gate+up+activation)
  if (layerWeights?.gate && layerWeights?.up && layerWeights?.down && !layerWeights.gateUp && inputTensor.dtype === 'f32') {
    const loraGate = getLoRAModule(lora, layerIdx, 'gate_proj');
    const loraUp = getLoRAModule(lora, layerIdx, 'up_proj');
    if (!loraGate && !loraUp) {
      const gateDtype = isWeightBuffer(layerWeights.gate) ? layerWeights.gate.dtype : 'f32';
      const upDtype = isWeightBuffer(layerWeights.up) ? layerWeights.up.dtype : 'f32';
      const dtypeMatches = gateDtype === upDtype;
      const dtypeSupported = gateDtype === 'f16' || gateDtype === 'f32';
      const f16BatchOk = gateDtype !== 'f16' || numTokens === 1;

      if (dtypeMatches && dtypeSupported && f16BatchOk) {
        const gateWeight = getWeightBuffer(layerWeights.gate, 'ffn_gate');
        const upWeight = getWeightBuffer(layerWeights.up, 'ffn_up');
        const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');

        const activation = hiddenActivation === 'gelu' ? 'gelu' : 'silu';
        const fusedOutput = recorder
          ? await recordFusedFFN(
            recorder,
            inputTensor,
            gateWeight,
            upWeight,
            hiddenSize,
            intermediateSize,
            { batchSize: numTokens, activation }
          )
          : await runFusedFFN(
            inputTensor,
            gateWeight,
            upWeight,
            hiddenSize,
            intermediateSize,
            { batchSize: numTokens, activation }
          );

        if (!(layerWeights.gate instanceof GPUBuffer) && !isWeightBuffer(layerWeights.gate)) {
          releaseBuffer(isWeightBuffer(gateWeight) ? gateWeight.buffer : gateWeight);
        }
        if (!(layerWeights.up instanceof GPUBuffer) && !isWeightBuffer(layerWeights.up)) {
          releaseBuffer(isWeightBuffer(upWeight) ? upWeight.buffer : upWeight);
        }

        let output = await doMatmul(
          fusedOutput,
          downWeight,
          numTokens,
          hiddenSize,
          intermediateSize,
          { transposeB: 'auto', label: `L${layerIdx}.ffn_down`, layerIdx, role: 'ffn_down' },
          recorder
        );

        const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
        if (loraDown) {
          const combined = await applyLoRA(
            fusedOutput,
            output,
            loraDown,
            { M: numTokens, N: hiddenSize, K: intermediateSize },
            getWeightBuffer,
            recorder
          );
          if (combined.buffer !== output.buffer) {
            if (recorder) {
              recorder.trackTemporaryBuffer(output.buffer);
            } else {
              releaseBuffer(output.buffer);
            }
            output = combined;
          }
        }

        if (!(layerWeights.down instanceof GPUBuffer) && !isWeightBuffer(layerWeights.down)) {
          releaseBuffer(isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
        }

        if (recorder) {
          recorder.trackTemporaryBuffer(fusedOutput.buffer);
        } else {
          releaseBuffer(fusedOutput.buffer);
        }

        return output;
      }
    }
  }

  // Fallback: separate gate/up path (3 matmuls)
  if (!layerWeights?.gate || !layerWeights?.up || !layerWeights?.down) {
    // Return copy of input (no FFN weights)
    log.warn('Layer', `L${layerIdx} FFN: no weights found (gateUp=${!!layerWeights?.gateUp}, gate=${!!layerWeights?.gate}, up=${!!layerWeights?.up}, down=${!!layerWeights?.down})`);
    const bytesPerElement = inputTensor.dtype === 'f16' ? 2 : 4;
    const byteSize = numTokens * hiddenSize * bytesPerElement;
    const outputBuffer = acquireBuffer(byteSize, undefined, 'ffn_output');
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(inputTensor.buffer, 0, outputBuffer, 0, byteSize);
    device.queue.submit([encoder.finish()]);
    return createTensor(outputBuffer, inputTensor.dtype, [...inputTensor.shape], 'ffn_output_copy');
  }

  // 1. Gate projection
  const useF16 = inputTensor.dtype === 'f16';
  const gateWeight = getWeightBuffer(layerWeights.gate, 'ffn_gate');
  let gateOutput = await doMatmul(inputTensor, gateWeight, numTokens, intermediateSize, hiddenSize, { transposeB: 'auto', label: `L${layerIdx}.ffn_gate`, layerIdx, outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_gate' }, recorder);
  if (!(layerWeights.gate instanceof GPUBuffer) && !isWeightBuffer(layerWeights.gate)) {
    releaseBuffer(isWeightBuffer(gateWeight) ? gateWeight.buffer : gateWeight);
  }

  const loraGate = getLoRAModule(lora, layerIdx, 'gate_proj');
  if (loraGate) {
    const combined = await applyLoRA(
      inputTensor,
      gateOutput,
      loraGate,
      { M: numTokens, N: intermediateSize, K: hiddenSize },
      getWeightBuffer,
      recorder
    );
    if (combined.buffer !== gateOutput.buffer) {
      if (recorder) {
        recorder.trackTemporaryBuffer(gateOutput.buffer);
      } else {
        releaseBuffer(gateOutput.buffer);
      }
      gateOutput = combined;
    }
  }

  // 2. Up projection
  const upWeight = getWeightBuffer(layerWeights.up, 'ffn_up');
  let upOutput = await doMatmul(inputTensor, upWeight, numTokens, intermediateSize, hiddenSize, { transposeB: 'auto', label: `L${layerIdx}.ffn_up`, layerIdx, outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_up' }, recorder);
  if (!(layerWeights.up instanceof GPUBuffer) && !isWeightBuffer(layerWeights.up)) {
    releaseBuffer(isWeightBuffer(upWeight) ? upWeight.buffer : upWeight);
  }

  const loraUp = getLoRAModule(lora, layerIdx, 'up_proj');
  if (loraUp) {
    const combined = await applyLoRA(
      inputTensor,
      upOutput,
      loraUp,
      { M: numTokens, N: intermediateSize, K: hiddenSize },
      getWeightBuffer,
      recorder
    );
    if (combined.buffer !== upOutput.buffer) {
      if (recorder) {
        recorder.trackTemporaryBuffer(upOutput.buffer);
      } else {
        releaseBuffer(upOutput.buffer);
      }
      upOutput = combined;
    }
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(gateOutput.buffer, 'ffn_gate', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: intermediateSize,
    });
    await dumpTokenVector(upOutput.buffer, 'ffn_up', {
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
    await dumpTokenVector(activatedOutput.buffer, 'ffn_activated', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: intermediateSize,
    });
  }

  // Track for cleanup after submit if using recorder, otherwise release immediately
  if (recorder) {
    recorder.trackTemporaryBuffer(gateOutput.buffer);
    recorder.trackTemporaryBuffer(upOutput.buffer);
  } else {
    releaseBuffer(gateOutput.buffer);
    releaseBuffer(upOutput.buffer);
  }

  // 4. Down projection
  const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');
  let output = await doMatmul(activatedOutput, downWeight, numTokens, hiddenSize, intermediateSize, { transposeB: 'auto', label: `L${layerIdx}.ffn_down`, layerIdx, outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_down' }, recorder);

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
    if (combined.buffer !== output.buffer) {
      if (recorder) {
        recorder.trackTemporaryBuffer(output.buffer);
      } else {
        releaseBuffer(output.buffer);
      }
      output = combined;
    }
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(output.buffer, 'ffn_down_out', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: hiddenSize,
    });
  }

  if (!(layerWeights.down instanceof GPUBuffer) && !isWeightBuffer(layerWeights.down)) {
    releaseBuffer(isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
  }
  // Track for cleanup after submit if using recorder, otherwise release immediately
  if (recorder) {
    recorder.trackTemporaryBuffer(activatedOutput.buffer);
  } else {
    releaseBuffer(activatedOutput.buffer);
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
  inputTensor: Tensor,
  numTokens: number,
  context: LayerContext,
  layerWeights: LayerWeights,
  residualTensor: Tensor,
  eps: number,
  transposeB: boolean,
  outputBuffer?: GPUBuffer | null
): Promise<Tensor> {
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

  let activatedOutput: Tensor;
  const useF16 = inputTensor.dtype === 'f16';

  if (hasFusedGateUp) {
    // Fused gate+up path
    const gateUpWeight = getWeightBuffer(layerWeights.gateUp!, 'ffn_gate_up');

    // 1. Fused gate+up projection
    let gateUpOutput = await doMatmul(
      inputTensor, gateUpWeight,
      numTokens, intermediateSize * 2, hiddenSize,
      { transposeB: 'auto', outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_gate_up' },
      recorder
    );

    const loraGateUp = getLoRAModule(lora, layerIdx, 'gate_up_proj');
    if (loraGateUp) {
      const combined = await applyLoRA(
        inputTensor,
        gateUpOutput,
        loraGateUp,
        { M: numTokens, N: intermediateSize * 2, K: hiddenSize },
        getWeightBuffer,
        recorder
      );
      if (combined.buffer !== gateUpOutput.buffer) {
        if (recorder) {
          recorder.trackTemporaryBuffer(gateUpOutput.buffer);
        } else {
          releaseBuffer(gateUpOutput.buffer);
        }
        gateUpOutput = combined;
      }
    }

    if (!(layerWeights.gateUp instanceof GPUBuffer) && !isWeightBuffer(layerWeights.gateUp)) {
      releaseBuffer(isWeightBuffer(gateUpWeight) ? gateUpWeight.buffer : gateUpWeight);
    }

    // 2. Split + Activation
    const activation = hiddenActivation === 'gelu' ? 'gelu' : 'silu';
    activatedOutput = await doSiLURowSplit(gateUpOutput, {
      numTokens,
      dim: intermediateSize,
      activation,
    }, recorder);

    if (recorder) {
      recorder.trackTemporaryBuffer(gateUpOutput.buffer);
    } else {
      releaseBuffer(gateUpOutput.buffer);
    }
  } else {
    // Separate gate/up path
    const gateWeight = getWeightBuffer(layerWeights.gate!, 'ffn_gate');
    const upWeight = getWeightBuffer(layerWeights.up!, 'ffn_up');

    // 1a. Gate projection
    const gateOutput = await doMatmul(
      inputTensor, gateWeight,
      numTokens, intermediateSize, hiddenSize,
      { transposeB: 'auto', outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_gate' },
      recorder
    );
    if (!(layerWeights.gate instanceof GPUBuffer) && !isWeightBuffer(layerWeights.gate)) {
      releaseBuffer(isWeightBuffer(gateWeight) ? gateWeight.buffer : gateWeight);
    }

    // 1b. Up projection
    const upOutput = await doMatmul(
      inputTensor, upWeight,
      numTokens, intermediateSize, hiddenSize,
      { transposeB: 'auto', outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_up' },
      recorder
    );
    if (!(layerWeights.up instanceof GPUBuffer) && !isWeightBuffer(layerWeights.up)) {
      releaseBuffer(isWeightBuffer(upWeight) ? upWeight.buffer : upWeight);
    }

    // 2. Activation: activation(gate) * up
    // The activation function handles both the activation and element-wise multiply
    const activationFn = hiddenActivation === 'gelu' ? doGeLU : doSiLU;
    activatedOutput = await activationFn(upOutput, {
      size: numTokens * intermediateSize,
      gate: gateOutput,
    }, recorder);

    if (recorder) {
      recorder.trackTemporaryBuffer(gateOutput.buffer);
      recorder.trackTemporaryBuffer(upOutput.buffer);
    } else {
      releaseBuffer(gateOutput.buffer);
      releaseBuffer(upOutput.buffer);
    }
  }

  // 3. FUSED: Down projection + RMSNorm + Residual (single kernel!)
  // This is the key optimization: combines matmul + norm + residual into one dispatch
  // Use pre-allocated decode buffer for output when available
  const outputTensor = await doMatmulRMSNormFused(
    activatedOutput,
    downWeight,
    normWeightBuf,
    {
      N: hiddenSize,
      K: intermediateSize,
      eps,
      residual: residualTensor,
      outputBuffer,
      transposeB,  // true for GGUF row-major, false for column-major
    },
    recorder
  );

  // Apply LoRA to output if needed (rare case)
  const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
  if (loraDown) {
    // LoRA needs separate application - for now, fall back to non-fused for LoRA
    // This is a rare case during fine-tuning only
    log.warn('Layer', `L${layerIdx} LoRA down_proj with fused kernel not yet optimized`);
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(outputTensor.buffer, 'fused_ffn_out', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: hiddenSize,
    });
  }

  if (!(layerWeights.down instanceof GPUBuffer) && !isWeightBuffer(layerWeights.down)) {
    releaseBuffer(isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
  }
  if (!(layerWeights.postFeedforwardNorm instanceof GPUBuffer)) releaseBuffer(normWeightBuf);
  if (recorder) {
    recorder.trackTemporaryBuffer(activatedOutput.buffer);
  } else {
    releaseBuffer(activatedOutput.buffer);
  }

  return outputTensor;
}

/**
 * Run MoE FFN on GPU.
 * Input and output are Tensor for dtype-aware processing.
 */
async function runMoEFFNGPU(
  layerIdx: number,
  inputTensor: Tensor,
  numTokens: number,
  context: LayerContext
): Promise<Tensor> {
  const { config, moeRouter, expertWeights, expertLoader, layerRouterWeights } = context;

  if (!moeRouter || !expertWeights || !expertLoader) {
    throw new Error('MoE components not initialized');
  }

  // Import dynamically to avoid circular dependency
  const { moeFeedForwardGPU } = await import('./moe-impl.js');

  // MoE implementation still uses GPUBuffer - wrap/unwrap at boundaries
  const outputBuffer = await moeFeedForwardGPU(
    inputTensor.buffer,
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

  // Wrap output in Tensor, preserving dtype from input
  return createTensor(outputBuffer, inputTensor.dtype, [...inputTensor.shape], 'moe_ffn_output');
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
  log.warn('Layer', `L${layerIdx} CPU fallback - returning input unchanged`);
  return new Float32Array(hiddenStates);
}
