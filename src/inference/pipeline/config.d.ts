/**
 * Model configuration parsing and normalization.
 * Handles HuggingFace, GGUF, and llama.cpp config formats.
 *
 * Architecture: Manifest-First Config Resolution
 * - manifest.inference is the source of truth (populated by converter)
 * - mergeConfig() merges manifest with runtime overrides
 * - toParsedConfigFromMerged() adapts MergedConfig to ParsedModelConfig
 *
 * See: config/merge.ts, config/schema/manifest.schema.ts
 */

import type {
  LayerPipelineSchema,
  KernelPathRef,
  ManifestInferenceSchema,
  ArchitectureSchema,
} from '../../config/schema/index.js';
import type { MergedConfig, RuntimeInferenceOverrides } from '../../config/merge.js';

export type ActivationType = 'silu' | 'gelu';

export interface RawConfig {
  model_type?: string;
  text_config?: RawConfig;
  architectures?: string[];
  hidden_size?: number;
  n_embd?: number;
  embeddingLength?: number;
  num_hidden_layers?: number;
  n_layer?: number;
  blockCount?: number;
  num_attention_heads?: number;
  n_head?: number;
  attentionHeadCount?: number;
  num_key_value_heads?: number;
  attentionHeadCountKV?: number;
  head_dim?: number;
  intermediate_size?: number;
  n_inner?: number;
  feedForwardLength?: number;
  vocab_size?: number;
  max_position_embeddings?: number;
  contextLength?: number;
  rope_theta?: number;
  rope_local_base_freq?: number;
  ropeFreqBase?: number;
  rms_norm_eps?: number;
  attentionLayerNormRMSEpsilon?: number;
  hidden_activation?: string;
  hidden_act?: string;
  eos_token_id?: number | number[];
  rope_scaling?: RopeScalingConfig;
  sliding_window?: number;
  sliding_window_pattern?: number;
  num_local_experts?: number;
  num_experts?: number;
  experts_per_token?: number;
  num_experts_per_tok?: number;
  top_k?: number;
  layer_types?: string[];
  attention_bias?: boolean;
  quantization_config?: { quant_method?: string };
  scale_embeddings?: boolean;
  rms_norm_weight_offset?: boolean;
  final_logit_softcapping?: number;
  attn_logit_softcapping?: number;
  query_pre_attn_scalar?: number;
}

export interface RopeScalingConfig {
  type?: string;
  rope_type?: string;
  factor?: number;
  beta_fast?: number;
  beta_slow?: number;
  original_max_position_embeddings?: number;
}

export interface TensorInfo {
  shape?: number[];
  dtype?: string;
}

export interface Manifest {
  architecture?: string;
  config?: RawConfig | Record<string, unknown>;
  tensors?: Record<string, TensorInfo>;
  tokenizer?: Record<string, unknown> & { vocab_size?: number };
  quantization?: string;
  eos_token_id?: number | number[];
  modelId?: string;
  model_id?: string;
  name?: string;
  draftModel?: { numTokens?: number };
  optimizations?: {
    useBatching?: boolean;
    debug?: boolean;
    kernelPath?: KernelPathRef;
  };
  runtime?: {
    useBatching?: boolean;
    debug?: boolean;
  };
  quantizationInfo?: {
    weights?: string;
    embeddings?: string;
    lmHead?: string;
    compute?: string;
  };
  inference?: ManifestInferenceSchema;
}

export interface AttentionParams {
  numHeads: number;
  numKVHeads: number;
  headDim: number;
}

export interface ParsedModelConfig {
  numLayers: number;
  hiddenSize: number;
  intermediateSize: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  vocabSize: number;
  maxSeqLen: number;
  useMoE: boolean;
  numExperts: number;
  moeTopK: number;
  expertFormat: 'mixtral' | 'gpt-oss' | null;
  slidingWindow: number | null;
  ropeTheta: number;
  ropeLocalTheta: number | null;
  ropeScale: number;
  ropeScalingType: string | null;
  ropeScaling: RopeScalingConfig | null;
  quantization: string;
  quantMethod: string | null;
  rmsNormEps: number;
  rmsNormWeightOffset: boolean;
  postAttentionNorm: boolean;
  preFeedforwardNorm: boolean;
  postFeedforwardNorm: boolean;
  scaleEmbeddings: boolean;
  useTiedEmbeddings: boolean;
  embeddingTranspose: boolean;
  embeddingVocabSize: number | null;
  hiddenActivation: ActivationType;
  swigluLimit: number | null;
  stopTokenIds: number[];
  layerTypes: string[] | null;
  attentionBias: boolean;
  embeddingScale?: number;
  finalLogitSoftcapping: number | null;
  attnLogitSoftcapping: number | null;
  queryKeyNorm: boolean;
  queryPreAttnScalar: number;
  layerPipeline?: LayerPipelineSchema | null;
  chatTemplateType?: string | null;
  chatTemplateEnabled: boolean;
  kernelPath?: KernelPathRef;
}

export function getStopTokenIds(config: RawConfig, manifest: Manifest): number[];

/**
 * Extended manifest with inference config for manifest-first parsing.
 */
export interface ManifestWithInference {
  inference: ManifestInferenceSchema;
  architecture: ArchitectureSchema;
  config?: RawConfig | Record<string, unknown>;
  tensors?: Record<string, TensorInfo>;
  tokenizer?: Record<string, unknown> & { vocab_size?: number };
  quantization?: string;
  modelId?: string;
  model_id?: string;
  eos_token_id: number | number[];
}

/**
 * Check if manifest has inference config for manifest-first parsing.
 */
export function hasManifestInference(manifest: Manifest): manifest is Manifest & { inference: ManifestInferenceSchema };

/**
 * Convert MergedConfig to ParsedModelConfig.
 */
export function toParsedConfigFromMerged(
  merged: MergedConfig,
  manifest: ManifestWithInference
): ParsedModelConfig;

/**
 * Parse model config from manifest using manifest-first resolution.
 */
export function parseModelConfigFromManifest(
  manifest: ManifestWithInference,
  runtimeOverrides?: RuntimeInferenceOverrides
): ParsedModelConfig;

/**
 * Parse model configuration from manifest.
 */
export function parseModelConfig(
  manifest: Manifest,
  runtimeOverrides?: RuntimeInferenceOverrides
): ParsedModelConfig;
