/**
 * Logits computation - final layer norm and LM head projection.
 *
 * This module handles the final steps of inference:
 * - Apply final RMS norm to hidden states
 * - Project to vocabulary size via LM head
 * - Handle tied embeddings (transposeB for HuggingFace format)
 * - CPU fallback for non-GPU execution
 *
 * @module inference/pipeline/logits
 */

import type { ProbeConfigSchema } from '../../../config/schema/index.js';
import type { LogitsConfig, LogitsWeights, LogitsDebugFlags } from './types.js';

// Re-export all types
export type { LogitsConfig, LogitsWeights, LogitsDebugFlags } from './types.js';

// Re-export CPU functions
export { rmsNormCPU, matmulCPU, applySoftcapping, f16ToF32, f16BufferToF32 } from './cpu.js';

// Re-export GPU functions
export { computeLogitsGPU, recordLogitsGPU, computeChunkedLogitsGPU, resolveCpuWeightDims, resolveLmHeadChunkRows, extractLmHeadChunk, writeChunkLogits } from './gpu.js';

// Re-export utilities
export { extractLastPositionLogits, finalizeLogits } from './utils.js';

/**
 * Compute logits from hidden states.
 *
 * This function:
 * 1. Applies final RMS normalization
 * 2. Projects to vocabulary via LM head matrix multiplication
 * 3. Handles tied embeddings (uses transposeB for HF format)
 * 4. Falls back to CPU if GPU unavailable
 *
 * @param hiddenStates - Hidden states from transformer [numTokens, hiddenSize]
 * @param numTokens - Number of tokens (required for GPU buffer input)
 * @param weights - Final norm and LM head weights
 * @param config - Model configuration for logits
 * @param useGPU - Whether to use GPU
 * @param debugFlags - Debug flags to prevent repeated logging
 * @param getNormWeightBuffer - Helper to get norm weight buffer (from pipeline)
 * @param debugCheckBuffer - Helper for debug buffer checking (from pipeline)
 * @returns Logits tensor [numTokens, vocabSize]
 */
export function computeLogits(
  hiddenStates: GPUBuffer | Float32Array,
  numTokens: number,
  weights: LogitsWeights,
  config: LogitsConfig,
  useGPU: boolean,
  debugFlags?: LogitsDebugFlags,
  getNormWeightBuffer?: (weight: GPUBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer,
  debugCheckBuffer?: (buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>,
  debugProbes?: ProbeConfigSchema[] | null
): Promise<Float32Array>;
