/**
 * Inference Schema Definitions
 *
 * Configuration for model inference behavior.
 * These are runtime settings that affect how the model executes.
 *
 * @module config/schema/inference
 */

import type { ProbeStage } from './debug.schema.js';
import type { KernelPathRef } from './kernel-path.schema.js';

/** RoPE configuration for positional embeddings */
export interface RoPEConfigSchema {
  /** Base frequency for RoPE (default 10000, modern models use 1000000) */
  ropeTheta?: number;

  /** Local RoPE theta for sliding window layers (Gemma 3 uses 10000) */
  ropeLocalTheta?: number;

  /** RoPE scaling type */
  ropeScalingType?: 'linear' | 'dynamic' | 'yarn' | null;

  /** RoPE scaling factor */
  ropeScalingFactor?: number;

  /** YARN beta_fast parameter */
  yarnBetaFast?: number;

  /** YARN beta_slow parameter */
  yarnBetaSlow?: number;

  /** YARN original max position embeddings */
  yarnOriginalMaxPos?: number;
}

/** Attention mechanism configuration */
export interface AttentionSchema {
  /** Use sliding window attention */
  slidingWindow?: number | null;
  /** Softcap attention logits before softmax */
  attnLogitSoftcapping?: number | null;
  /** Use query-key normalization */
  queryKeyNorm?: boolean;
  /** @deprecated Use RoPEConfigSchema.ropeScalingType instead */
  ropeScalingType?: 'linear' | 'dynamic' | 'yarn' | null;
  /** @deprecated Use RoPEConfigSchema.ropeScalingFactor instead */
  ropeScalingFactor?: number;
}

/** Normalization configuration */
export interface NormalizationSchema {
  /** Add 1.0 to RMSNorm weights (Gemma-style) */
  rmsNormWeightOffset?: boolean;
  /** RMSNorm epsilon */
  rmsNormEps?: number;
  /** Use post-attention norm */
  postAttentionNorm?: boolean;
  /** Use pre-feedforward norm */
  preFeedforwardNorm?: boolean;
  /** Use post-feedforward norm */
  postFeedforwardNorm?: boolean;
}

/** Feed-forward network configuration */
export interface FFNSchema {
  /** Activation function */
  activation?: 'silu' | 'gelu' | 'relu' | 'swiglu';
  /** Use gated FFN (SwiGLU) */
  gatedFFN?: boolean;
  /** Use fused gate+up projection */
  fusedGateUp?: boolean;
}

/** Built-in chat template types */
export type ChatTemplateType = 'gemma' | 'llama3' | 'gpt-oss' | null;

/** Chat template configuration for instruct models */
export interface ChatTemplateSchema {
  /** Template type identifier (gemma, llama3, gpt-oss) */
  type?: ChatTemplateType;

  /** Whether to apply chat template by default (instruct models should set true) */
  enabled?: boolean;

  /** Custom template with {prompt} placeholder (overrides type) */
  custom?: string;
}

export type LayerPipelineOp =
  | 'save'
  | 'load'
  | 'attention'
  | 'rmsnorm'
  | 'ffn'
  | 'residual_add'
  | 'noop';

export type LayerPipelineNormWeight =
  | 'input'
  | 'post_attention'
  | 'post_attn'
  | 'pre_ffn'
  | 'post_ffn';

export interface LayerPipelineStepSchema {
  op: LayerPipelineOp;
  /** Source slot (default: "state") */
  src?: string;
  /** Destination slot (default: "state") */
  dst?: string;
  /** Slot name for save/load operations */
  name?: string;
  /** Norm weight selector (rmsnorm only) */
  weight?: LayerPipelineNormWeight;
  /** Residual slot for fused ops (optional) */
  residual?: string | null;
  /** Residual add inputs (defaults: a="state", b="residual") */
  a?: string;
  b?: string;
  /** FFN variant override */
  variant?: 'auto' | 'dense' | 'moe';
  /** Skip input norm inside attention (use when providing explicit rmsnorm) */
  skipInputNorm?: boolean;
  /** Optional probe stage to emit for this step */
  probeStage?: ProbeStage;
}

export interface LayerPipelineOverrideSchema {
  layers: number[];
  steps: LayerPipelineStepSchema[];
}

export interface LayerPipelineSchema {
  steps: LayerPipelineStepSchema[];
  overrides?: LayerPipelineOverrideSchema[];
}

/** Output/sampling configuration */
export interface OutputSchema {
  /** Softcap final logits */
  finalLogitSoftcapping?: number | null;
  /** Tie embeddings to output */
  tieWordEmbeddings?: boolean;
}

/** Layer type for hybrid models */
export type LayerType = 'attention' | 'mamba' | 'rwkv';

/** Global layer pattern (computed at runtime from numLayers) */
export type GlobalLayerPattern =
  | 'even'       // Layers 0, 2, 4, ... are global (Gemma 2)
  | 'odd'        // Layers 1, 3, 5, ... are global
  | 'every_n';   // Every Nth layer is global (Gemma 3: every 6th)

/** Layer pattern for hybrid architectures */
export interface LayerPatternSchema {
  /** Pattern type: 'all_attention', 'alternating', 'custom' */
  type: 'all_attention' | 'alternating' | 'custom';
  /** For 'alternating': pattern for global/full attention layers */
  globalPattern?: GlobalLayerPattern;
  /** For 'alternating' with 'every_n': the N value (e.g., 6 for Gemma 3) */
  globalPatternN?: number;
  /** @deprecated Use globalPattern instead */
  attentionLayers?: number[];
  /** For 'custom': explicit layer type mapping */
  layerTypes?: LayerType[];
}

/**
 * Compute global attention layer indices from pattern.
 * Used at runtime when numLayers is known.
 */
export declare function computeGlobalLayers(
  pattern: LayerPatternSchema,
  numLayers: number
): number[];

/** Complete inference configuration */
export interface InferenceConfigSchema {
  attention?: AttentionSchema;
  normalization?: NormalizationSchema;
  ffn?: FFNSchema;
  output?: OutputSchema;
  layerPattern?: LayerPatternSchema;
  rope?: RoPEConfigSchema;
  pipeline?: LayerPipelineSchema | null;
  /** Chat template for instruct models */
  chatTemplate?: ChatTemplateSchema;
  /**
   * Kernel path for explicit kernel dispatch ordering.
   * Specifies exactly which kernels run, in what order, with what configs.
   * Can be a preset ID (e.g., 'gemma2-q4k-fused-f32a') or inline KernelPathSchema.
   */
  kernelPath?: KernelPathRef;
}

/** Sampling parameters */
export interface SamplingSchema {
  temperature?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
}

/** Tokenizer runtime configuration */
export interface TokenizerConfigSchema {
  /** BOS token string */
  bosToken?: string;
  /** EOS token strings (can be multiple) */
  eosTokens?: string[];
  /** Pad token string */
  padToken?: string;
  /** Add BOS token to input */
  addBosToken?: boolean;
  /** Add EOS token to output */
  addEosToken?: boolean;
  /** Chat template (jinja2-style) */
  chatTemplate?: string;
}
