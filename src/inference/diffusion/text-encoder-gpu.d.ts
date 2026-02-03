/**
 * Diffusion GPU text encoders (SD3).
 *
 * @module inference/diffusion/text-encoder-gpu
 */

import type { Tensor } from '../../gpu/tensor.js';
import type { DiffusionModelConfig, DiffusionRuntimeConfig } from './types.js';

export interface DiffusionTextEncoderWeightsEntry {
  weights: Map<string, any>;
  shapes: Map<string, number[]>;
  dtypes?: Map<string, string>;
}

export interface DiffusionTextEncoderWeights {
  text_encoder: DiffusionTextEncoderWeightsEntry;
  text_encoder_2: DiffusionTextEncoderWeightsEntry;
  text_encoder_3: DiffusionTextEncoderWeightsEntry;
  transformer?: DiffusionTextEncoderWeightsEntry;
}

export interface DiffusionTextTokens {
  text_encoder: number[];
  text_encoder_2: number[];
  text_encoder_3: number[];
}

export interface DiffusionTextConditioning {
  pooled: Float32Array;
  context: Tensor;
}

export declare function runTextEncodersForPrompt(
  tokensByEncoder: DiffusionTextTokens,
  weightsByComponent: DiffusionTextEncoderWeights,
  modelConfig: DiffusionModelConfig,
  runtime: DiffusionRuntimeConfig
): Promise<DiffusionTextConditioning>;

export declare function buildTimeTextEmbedding(
  pooled: Float32Array,
  weightsEntry: DiffusionTextEncoderWeightsEntry,
  modelConfig: DiffusionModelConfig,
  runtime: DiffusionRuntimeConfig
): Promise<Tensor>;

export declare function buildTimestepEmbedding(
  timestep: number,
  weightsEntry: DiffusionTextEncoderWeightsEntry,
  modelConfig: DiffusionModelConfig,
  runtime: DiffusionRuntimeConfig,
  options?: { dim?: number }
): Promise<Tensor>;

export declare function combineTimeTextEmbeddings(
  time: Tensor,
  text: Tensor,
  hiddenSize: number
): Promise<Tensor>;

export declare function projectContext(
  context: Tensor,
  weightsEntry: DiffusionTextEncoderWeightsEntry,
  modelConfig: DiffusionModelConfig,
  runtime: DiffusionRuntimeConfig
): Promise<Tensor>;

export declare function logQuickGeluWarning(config: { hidden_act?: string }): void;
