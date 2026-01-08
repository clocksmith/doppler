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
} from '../../gpu/kernel-selector.js';
import { releaseBuffer } from '../../gpu/buffer-pool.js';
import { kernelTrace, traceStep } from './kernel-trace.js';
import {
  runLayerAttentionGPU,
  recordLayerAttentionGPU,
} from './attention.js';

/**
 * @param {import('../decode-buffers.js').DecodeBufferManager | null | undefined} decodeBuffers
 * @param {GPUBuffer} buffer
 * @returns {boolean}
 */
export function isDecodeBuffer(decodeBuffers, buffer) {
  return !!decodeBuffers?.ownsBuffer(buffer);
}

/**
 * @param {import('../../gpu/kernel-selector.js').CommandRecorder | undefined} recorder
 * @param {GPUBuffer} buffer
 * @param {import('../decode-buffers.js').DecodeBufferManager | null} [decodeBuffers]
 */
export function releaseOrTrack(recorder, buffer, decodeBuffers) {
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
 * @param {import('../../gpu/tensor.js').Tensor} input
 * @param {GPUBuffer} weight
 * @param {number} eps
 * @param {{ batchSize: number; hiddenSize: number; residual?: import('../../gpu/tensor.js').Tensor | null; outputBuffer?: GPUBuffer | null; label?: string; layerIdx?: number }} options
 * @param {import('../../gpu/kernel-selector.js').CommandRecorder} [recorder]
 * @returns {Promise<import('../../gpu/tensor.js').Tensor>}
 */
export async function doRMSNorm(input, weight, eps, options, recorder) {
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
 * @param {import('../../gpu/tensor.js').Tensor} a
 * @param {import('../../gpu/tensor.js').Tensor} b
 * @param {number} size
 * @param {import('../../gpu/kernel-selector.js').CommandRecorder} [recorder]
 * @param {{ label?: string; layerIdx?: number; outputBuffer?: GPUBuffer | null }} [traceOptions]
 * @returns {Promise<import('../../gpu/tensor.js').Tensor>}
 */
export async function doResidualAdd(a, b, size, recorder, traceOptions) {
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
 * @param {import('../../gpu/tensor.js').Tensor} A
 * @param {GPUBuffer | import('../../gpu/weight-buffer.js').WeightBuffer} B
 * @param {number} M
 * @param {number} N
 * @param {number} K
 * @param {{ transposeB?: boolean | 'auto'; label?: string; layerIdx?: number; outputDtype?: 'f16' | 'f32'; role?: string }} [options]
 * @param {import('../../gpu/kernel-selector.js').CommandRecorder} [recorder]
 * @returns {Promise<import('../../gpu/tensor.js').Tensor>}
 */
export async function doMatmul(A, B, M, N, K, options = {}, recorder) {
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
 * @param {import('../../gpu/tensor.js').Tensor} input
 * @param {{ size?: number; gate?: import('../../gpu/tensor.js').Tensor | null; label?: string; layerIdx?: number }} [options]
 * @param {import('../../gpu/kernel-selector.js').CommandRecorder} [recorder]
 * @returns {Promise<import('../../gpu/tensor.js').Tensor>}
 */
export async function doSiLU(input, options = {}, recorder) {
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
 * @param {import('../../gpu/tensor.js').Tensor} input
 * @param {{ size?: number; gate?: import('../../gpu/tensor.js').Tensor | null; label?: string; layerIdx?: number }} [options]
 * @param {import('../../gpu/kernel-selector.js').CommandRecorder} [recorder]
 * @returns {Promise<import('../../gpu/tensor.js').Tensor>}
 */
export async function doGeLU(input, options = {}, recorder) {
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
 * @param {import('../../gpu/tensor.js').Tensor} input
 * @param {Omit<import('../../gpu/kernel-selector.js').SiLURowSplitOptions, 'activationDtype'> & { label?: string; layerIdx?: number }} options
 * @param {import('../../gpu/kernel-selector.js').CommandRecorder} [recorder]
 * @returns {Promise<import('../../gpu/tensor.js').Tensor>}
 */
export async function doSiLURowSplit(input, options, recorder) {
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
 * @param {import('../../gpu/tensor.js').Tensor} input
 * @param {GPUBuffer | import('../../gpu/weight-buffer.js').WeightBuffer} weight
 * @param {GPUBuffer} normWeight
 * @param {{ N: number; K: number; eps: number; residual?: import('../../gpu/tensor.js').Tensor | null; outputBuffer?: GPUBuffer | null; transposeB?: boolean; label?: string; layerIdx?: number }} options
 * @param {import('../../gpu/kernel-selector.js').CommandRecorder} [recorder]
 * @returns {Promise<import('../../gpu/tensor.js').Tensor>}
 */
export async function doMatmulRMSNormFused(input, weight, normWeight, options, recorder) {
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
 * @param {import('../../gpu/tensor.js').Tensor} inputTensor
 * @param {import('./types.js').LayerWeights | null} layerWeights
 * @param {import('./attention.js').AttentionConfig} config
 * @param {import('./attention.js').AttentionState} state
 * @param {boolean} debug
 * @param {import('./attention.js').AttentionDebugFlags} debugFlags
 * @param {(weight: GPUBuffer | import('../../gpu/weight-buffer.js').WeightBuffer | Float32Array | ArrayBuffer | import('../../gpu/weight-buffer.js').CpuWeightBuffer, label: string) => GPUBuffer | import('../../gpu/weight-buffer.js').WeightBuffer} getWeightBufferFn
 * @param {(weight: GPUBuffer | Float32Array | ArrayBuffer | import('../../gpu/weight-buffer.js').CpuWeightBuffer, label: string) => GPUBuffer} getNormWeightBufferFn
 * @param {((buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>) | undefined} [debugCheckBuffer]
 * @param {import('../../gpu/kernel-selector.js').CommandRecorder} [recorder]
 * @param {import('./lora.js').LoRAAdapter | null} [lora]
 * @returns {Promise<import('./attention.js').AttentionResult>}
 */
export async function doAttention(
  inputTensor,
  layerWeights,
  config,
  state,
  debug,
  debugFlags,
  getWeightBufferFn,
  getNormWeightBufferFn,
  debugCheckBuffer,
  recorder,
  lora
) {
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
