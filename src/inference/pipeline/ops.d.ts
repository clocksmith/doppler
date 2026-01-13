/**
 * Kernel Operation Wrappers (Ops)
 *
 * This module provides high-level wrappers around GPU kernels (run/record variants)
 * and handles tensor creation, tracing, and buffer management.
 *
 * @module inference/pipeline/ops
 */

import type { SiLURowSplitOptions, CommandRecorder } from '../../gpu/kernel-selector.js';
import type { Tensor } from '../../gpu/tensor.js';
import type { WeightBuffer, CpuWeightBuffer } from '../../gpu/weight-buffer.js';
import type { DecodeBufferManager } from '../decode-buffers.js';
import type {
  AttentionConfig,
  AttentionState,
  AttentionDebugFlags,
  AttentionResult
} from './attention.js';
import type { LayerWeights } from './types.js';
import type { LoRAAdapter } from './lora.js';

export function isDecodeBuffer(decodeBuffers: DecodeBufferManager | null | undefined, buffer: GPUBuffer): boolean;

export function releaseOrTrack(
  recorder: CommandRecorder | undefined,
  buffer: GPUBuffer,
  decodeBuffers?: DecodeBufferManager | null
): void;

/**
 * RMSNorm that uses record variant when recorder is provided.
 * Input and residual are Tensor, returns Tensor.
 */
export function doRMSNorm(
  input: Tensor,
  weight: GPUBuffer,
  eps: number,
  options: { batchSize: number; hiddenSize: number; residual?: Tensor | null; outputBuffer?: GPUBuffer | null; label?: string; layerIdx?: number; rmsNormWeightOffset?: boolean },
  recorder?: CommandRecorder
): Promise<Tensor>;

/**
 * ResidualAdd that uses record variant when recorder is provided.
 * Accepts Tensor for inputs, returns Tensor.
 */
export function doResidualAdd(
  a: Tensor,
  b: Tensor,
  size: number,
  recorder?: CommandRecorder,
  traceOptions?: { label?: string; layerIdx?: number; outputBuffer?: GPUBuffer | null }
): Promise<Tensor>;

/**
 * Matmul that uses record variant when recorder is provided.
 * A is activation Tensor, B is weight (GPUBuffer or WeightBuffer), returns Tensor.
 */
export function doMatmul(
  A: Tensor,
  B: GPUBuffer | WeightBuffer,
  M: number,
  N: number,
  K: number,
  options?: { transposeB?: boolean | 'auto'; label?: string; layerIdx?: number; outputDtype?: 'f16' | 'f32'; role?: string },
  recorder?: CommandRecorder
): Promise<Tensor>;

/**
 * SiLU that uses record variant when recorder is provided.
 * Supports gated variant (SiLU with gate multiplication).
 */
export function doSiLU(
  input: Tensor,
  options?: { size?: number; gate?: Tensor | null; label?: string; layerIdx?: number },
  recorder?: CommandRecorder
): Promise<Tensor>;

/**
 * GeLU that uses record variant when recorder is provided.
 * Supports gated variant (GeGLU).
 */
export function doGeLU(
  input: Tensor,
  options?: { size?: number; gate?: Tensor | null; label?: string; layerIdx?: number },
  recorder?: CommandRecorder
): Promise<Tensor>;

/**
 * SiLURowSplit that uses record variant when recorder is provided.
 * Used for fused gate+up FFN path: splits combined output and applies activation.
 */
export function doSiLURowSplit(
  input: Tensor,
  options: Omit<SiLURowSplitOptions, 'activationDtype'> & { label?: string; layerIdx?: number },
  recorder?: CommandRecorder
): Promise<Tensor>;

/**
 * Fused Matmul + RMSNorm that uses record variant when recorder is provided.
 * Used for down projection + post-FFN norm fusion during decode (M=1).
 */
export function doMatmulRMSNormFused(
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  normWeight: GPUBuffer,
  options: { N: number; K: number; eps: number; residual?: Tensor | null; outputBuffer?: GPUBuffer | null; transposeB?: boolean; label?: string; layerIdx?: number; rmsNormWeightOffset?: boolean },
  recorder?: CommandRecorder
): Promise<Tensor>;

/**
 * Attention that uses record variant when recorder is provided.
 * Input is Tensor for dtype-aware processing.
 */
export function doAttention(
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
): Promise<AttentionResult>;
