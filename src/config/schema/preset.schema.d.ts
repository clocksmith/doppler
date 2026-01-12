/**
 * Preset Schema Definitions
 *
 * Model family presets for config-as-code architecture.
 * Presets define defaults that can be overridden by manifest config.
 *
 * @module config/schema/preset
 */

import type { ModelType, ArchitectureSchema } from './manifest.schema.js';
import type {
  InferenceConfigSchema,
  TokenizerConfigSchema,
  SamplingSchema,
} from './inference.schema.js';
import type { LoadingConfigSchema } from './loading.schema.js';

/** Kernel path map for conversion defaults (weights quantization -> activation dtype). */
export type KernelPathMapSchema = Record<string, string | Record<string, string>>;

export type PresetInferenceSchema = InferenceConfigSchema & {
  kernelPaths?: KernelPathMapSchema;
};

/** Model family preset */
export interface PresetSchema {
  /** Preset identifier */
  id: string;

  /** Human-readable name */
  name?: string;

  /** Parent preset to extend */
  extends?: string;

  /** Model type hint */
  modelType?: ModelType;

  /** Architecture parameter overrides */
  architecture?: Partial<ArchitectureSchema>;

  /** Inference behavior */
  inference?: PresetInferenceSchema;

  /** Tokenizer configuration */
  tokenizer?: TokenizerConfigSchema;

  /** Default sampling parameters */
  sampling?: SamplingSchema;

  /** Tensor name patterns for this model family */
  tensorPatterns?: TensorPatternSchema;

  /** Model family detection patterns */
  detection?: DetectionPatternSchema;

  /** Loading behavior configuration */
  loading?: Partial<LoadingConfigSchema>;
}

/** Tensor naming patterns for model family */
export interface TensorPatternSchema {
  /** Embedding tensor patterns */
  embedding?: string[];
  /** LM head patterns */
  lmHead?: string[];
  /** Layer patterns (with {layer} placeholder) */
  layer?: string[];
  /** Attention patterns */
  attention?: {
    qProj?: string[];
    kProj?: string[];
    vProj?: string[];
    oProj?: string[];
    qkv?: string[];  // Fused QKV
  };
  /** FFN patterns */
  ffn?: {
    gate?: string[];
    up?: string[];
    down?: string[];
    gateUp?: string[];  // Fused gate+up
  };
  /** Norm patterns */
  norm?: {
    input?: string[];
    postAttn?: string[];
    final?: string[];
  };
}

/** Patterns for detecting model family */
export interface DetectionPatternSchema {
  /** Architecture name patterns (regex) */
  architecturePatterns?: string[];
  /** Model type patterns */
  modelTypePatterns?: string[];
  /** Config field patterns */
  configPatterns?: Record<string, unknown>;
}

/** Fully resolved configuration (preset + manifest merged) */
export interface ResolvedConfigSchema {
  /** Source preset ID */
  preset: string;

  /** Model type */
  modelType: ModelType;

  /** Architecture parameters */
  architecture: ArchitectureSchema;

  /** Inference configuration */
  inference: Required<InferenceConfigSchema>;

  /** Tokenizer configuration */
  tokenizer: TokenizerConfigSchema;

  /** Sampling defaults */
  sampling: SamplingSchema;

  /** Loading behavior configuration */
  loading: LoadingConfigSchema;
}
