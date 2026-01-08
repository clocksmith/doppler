/**
 * Kernel Operation Wrappers (Ops)
 *
 * This module provides high-level wrappers around GPU kernels (run/record variants)
 * and handles tensor creation, tracing, and buffer management.
 *
 * @module inference/pipeline/ops
 */

import {
  runRMSNorm, runResidualAdd, runMatmul, runSiLU, runGeLU,
  recordRMSNorm, recordResidualAdd, recordMatmul, recordSiLU, recordGeLU,
  runSiLURowSplit, recordSiLURowSplit,
  runMatmulRMSNormFused, recordMatmulRMSNormFused,
  type SiLURowSplitOptions,
  type CommandRecorder,
} from '../../gpu/kernel-selector.js';
import { Tensor, createTensor, type TensorDtype } from '../../gpu/tensor.js';
import { type WeightBuffer, type CpuWeightBuffer } from '../../gpu/weight-buffer.js';
import { releaseBuffer } from '../../gpu/buffer-pool.js';
import { kernelTrace, traceStep } from './kernel-trace.js';
import type { DecodeBufferManager } from '../decode-buffers.js';
import {
  runLayerAttentionGPU,
  recordLayerAttentionGPU,
  type AttentionConfig,
  type AttentionState,
  type AttentionDebugFlags,
  type AttentionResult
} from './attention.js';
import { getWeightBuffer, getNormWeightBuffer, type WeightBufferConfig, type WeightDebugFlags } from './weights.js';
import type { LayerWeights } from './types.js';
import type { LoRAAdapter } from './lora.js';

export function isDecodeBuffer(decodeBuffers: DecodeBufferManager | null | undefined, buffer: GPUBuffer): boolean {
  return !!decodeBuffers?.ownsBuffer(buffer);
}

export function releaseOrTrack(
  recorder: CommandRecorder | undefined,
  buffer: GPUBuffer,
  decodeBuffers?: DecodeBufferManager | null
): void {
  if (isDecodeBuffer(decodeBuffers, buffer)) {
    return;
  }
  if (recorder) {
    recorder.trackTemporaryBuffer(buffer);
  } else {
    releaseBuffer(buffer);
  }
}

/**
 * RMSNorm that uses record variant when recorder is provided.
 * Input and residual are Tensor, returns Tensor.
 */
export async function doRMSNorm(
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
export async function doResidualAdd(
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
export async function doMatmul(
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
export async function doSiLU(
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
export async function doGeLU(
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
export async function doSiLURowSplit(
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
export async function doMatmulRMSNormFused(
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
export async function doAttention(
  inputTensor: Tensor,
  layerWeights: LayerWeights | null,
  config: AttentionConfig,
  state: AttentionState,
  debug: boolean,
  debugFlags: AttentionDebugFlags,
  getWeightBufferFn: (weight: GPUBuffer | WeightBuffer | Float32Array | ArrayBuffer | CpuWeightBuffer, label: string) => GPUBuffer | WeightBuffer,
  getNormWeightBufferFn: (weight: GPUBuffer | Float32Array | ArrayBuffer | CpuWeightBuffer, label: string) => GPUBuffer,
  debugCheckBuffer?: (buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>,
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
      debugCheckBuffer,
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
    debugCheckBuffer,
    lora
  );
}
