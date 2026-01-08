/**
 * Type definitions for logits computation.
 *
 * Contains interfaces for configuration, weights, and debug flags
 * used throughout the logits computation pipeline.
 *
 * @module inference/pipeline/logits/types
 */

import type { LargeWeightConfigSchema } from '../../../config/schema/index.js';
import type { WeightBuffer, CpuWeightBuffer } from '../../../gpu/weight-buffer.js';

/**
 * Configuration for logits computation.
 */
export interface LogitsConfig {
  hiddenSize: number;
  vocabSize: number;
  rmsNormEps: number;
  useTiedEmbeddings: boolean;
  embeddingVocabSize: number | null;
  finalLogitSoftcapping: number | null;  // Gemma 2: 30.0 - applies tanh(x/cap)*cap
  largeWeights?: LargeWeightConfigSchema;
  /** Dtype for hidden state activations */
  activationDtype?: 'f16' | 'f32';
}

/**
 * Weights required for logits computation.
 */
export interface LogitsWeights {
  finalNorm: GPUBuffer | Float32Array;
  lmHead: GPUBuffer | Float32Array | WeightBuffer | CpuWeightBuffer;
}

/**
 * Debug flags for logits computation.
 */
export interface LogitsDebugFlags {
  finalNormDebugDone?: boolean;
  afterFinalNormDebugDone?: boolean;
}
