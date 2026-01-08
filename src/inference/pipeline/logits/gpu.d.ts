/**
 * GPU implementations for logits computation.
 *
 * Provides GPU-accelerated implementations for computing logits,
 * including both immediate execution and recorded (batched) variants.
 *
 * @module inference/pipeline/logits/gpu
 */

import type { Tensor } from '../../../gpu/tensor.js';
import type { CpuWeightBuffer } from '../../../gpu/weight-buffer.js';
import type { CommandRecorder } from '../../../gpu/command-recorder.js';
import type { LargeWeightConfigSchema, ProbeConfigSchema } from '../../../config/schema/index.js';
import type { LogitsConfig, LogitsWeights, LogitsDebugFlags } from './types.js';

/**
 * Resolve CPU weight buffer dimensions for LM head.
 */
export function resolveCpuWeightDims(lmHead: CpuWeightBuffer): { vocabSize: number; hiddenSize: number };

/**
 * Calculate the maximum rows per chunk for LM head matmul.
 */
export function resolveLmHeadChunkRows(
  device: GPUDevice,
  numTokens: number,
  hiddenSize: number,
  config?: LargeWeightConfigSchema
): number;

/**
 * Extract a chunk of the LM head weight matrix.
 */
export function extractLmHeadChunk(
  data: Float32Array,
  layout: 'row' | 'column',
  hiddenSize: number,
  vocabSize: number,
  rowOffset: number,
  rowCount: number
): Float32Array;

/**
 * Write chunk logits to the full logits buffer.
 */
export function writeChunkLogits(
  target: Float32Array,
  chunk: Float32Array,
  numTokens: number,
  vocabSize: number,
  rowOffset: number,
  rowCount: number
): void;

/**
 * Compute logits using chunked GPU matmul for large LM heads.
 *
 * Used when LM head weights are CPU-resident and too large
 * to fit in a single GPU buffer binding.
 */
export function computeChunkedLogitsGPU(
  normedTensor: Tensor,
  lmHead: CpuWeightBuffer,
  numTokens: number,
  hiddenSize: number,
  vocabSize: number,
  weightVocabSize: number,
  debugProbes?: ProbeConfigSchema[] | null,
  largeWeightConfig?: LargeWeightConfigSchema
): Promise<Float32Array>;

/**
 * Compute logits and return GPU buffer directly (deferred readback).
 *
 * This variant avoids the ~1MB readback per token, enabling GPU-side sampling.
 * Use with runGPUSample or runArgmax to sample directly on GPU.
 *
 * @param hiddenStates - Hidden states from transformer [numTokens, hiddenSize]
 * @param numTokens - Number of tokens
 * @param weights - Final norm and LM head weights
 * @param config - Model configuration for logits
 * @param debugFlags - Debug flags to prevent repeated logging (optional)
 * @returns GPU buffer containing logits [numTokens, vocabSize]
 */
export function computeLogitsGPU(
  hiddenStates: GPUBuffer | Float32Array,
  numTokens: number,
  weights: LogitsWeights,
  config: LogitsConfig,
  debugFlags?: LogitsDebugFlags,
): Promise<{ logitsBuffer: GPUBuffer; vocabSize: number } | null>;

/**
 * Record logits computation (batched, no submit).
 *
 * This variant uses the CommandRecorder to batch logits computation with
 * preceding layer operations, avoiding a GPU sync point.
 *
 * @param recorder - CommandRecorder for batched operations
 * @param hiddenStates - Hidden states from transformer [numTokens, hiddenSize]
 * @param numTokens - Number of tokens
 * @param weights - Final norm and LM head weights
 * @param config - Model configuration for logits
 * @returns GPU buffer containing logits [numTokens, vocabSize] and vocab size
 */
export function recordLogitsGPU(
  recorder: CommandRecorder,
  hiddenStates: GPUBuffer,
  numTokens: number,
  weights: LogitsWeights,
  config: LogitsConfig,
): Promise<{ logitsBuffer: GPUBuffer; vocabSize: number }>;
