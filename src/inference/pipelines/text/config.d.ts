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
  ManifestEmbeddingPostprocessorSchema,
  ArchitectureSchema,
  LinearNormMode,
} from '../../../config/schema/index.js';
import type { MergedConfig, RuntimeInferenceOverrides } from '../../../config/merge.js';

export type ActivationType = 'silu' | 'gelu';
export type ParsedLayerType =
  | 'full_attention'
  | 'sliding_attention'
  | 'linear_attention'
  | 'conv'
  | 'moe'
  | 'mamba'
  | 'rwkv';

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
  global_head_dim?: number;
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
  num_kv_shared_layers?: number;
  linear_num_key_heads?: number;
  linear_num_value_heads?: number;
  linear_key_head_dim?: number;
  linear_value_head_dim?: number;
  linear_conv_kernel_dim?: number;
  linear_norm_mode?: LinearNormMode;
  linear_norm_shared?: boolean;
  attention_bias?: boolean;
  quantization_config?: { quant_method?: string };
  scale_embeddings?: boolean;
  hidden_size_per_layer_input?: number;
  vocab_size_per_layer_input?: number;
  rms_norm_weight_offset?: boolean;
  final_logit_softcapping?: number;
  attn_logit_softcapping?: number;
  query_pre_attn_scalar?: number;
  attn_output_gate?: boolean;
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
  architecture?: ArchitectureSchema;
  config?: RawConfig | Record<string, unknown>;
  tensors?: Record<string, TensorInfo>;
  tokenizer?: Record<string, unknown> & { vocab_size?: number };
  quantization?: string;
  eos_token_id?: number | number[];
  modelId?: string;
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
  globalHeadDim: number | null;
  vocabSize: number;
  hiddenSizePerLayerInput: number | null;
  vocabSizePerLayerInput: number | null;
  numKvSharedLayers: number;
  maxSeqLen: number;
  useMoE: boolean;
  numExperts: number;
  moeTopK: number;
  expertFormat: 'mixtral' | 'gpt-oss' | null;
  slidingWindow: number | null;
  ropeTheta: number;
  ropeLocalTheta: number | null;
  ropeRotaryDim: number;
  ropeLocalRotaryDim: number;
  ropeInterleaved: boolean;
  mropeInterleaved: boolean;
  mropeSection: number[] | null;
  partialRotaryFactor: number | null;
  ropeLocalPartialRotaryFactor: number | null;
  ropeScale: number;
  ropeLocalScale: number;
  ropeScalingType: string | null;
  ropeLocalScalingType: string | null;
  ropeScaling: RopeScalingConfig | null;
  ropeLocalScaling: RopeScalingConfig | null;
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
  embeddingPostprocessor: ManifestEmbeddingPostprocessorSchema | null;
  hiddenActivation: ActivationType;
  swigluLimit: number | null;
  stopTokenIds: number[];
  layerTypes: ParsedLayerType[] | null;
  linearNumKeyHeads: number | null;
  linearNumValueHeads: number | null;
  linearKeyHeadDim: number | null;
  linearValueHeadDim: number | null;
  linearConvKernelDim: number | null;
  linearNormMode: LinearNormMode | null;
  attentionBias: boolean;
  causalAttention: boolean;
  embeddingScale?: number;
  finalLogitSoftcapping: number | null;
  attnLogitSoftcapping: number | null;
  queryKeyNorm: boolean;
  attentionOutputGate: boolean;
  queryPreAttnScalar: number;
  layerPipeline?: LayerPipelineSchema | null;
  chatTemplateType?: string | null;
  chatTemplateEnabled: boolean;
  decodeStrategy: 'incremental' | 'replay_prefill';
  kernelPath?: KernelPathRef;
}

export function getStopTokenIds(manifest: Manifest): number[];

/**
 * Validate that a chatTemplate.type is a known formatter type.
 * Logs a warning for unknown types. Returns true if valid or null.
 */
export function validateChatTemplateType(type: string | null | undefined, modelId: string): boolean;

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
  eos_token_id: number | number[];
}

/**
 * Check if manifest has inference config for manifest-first parsing.
 */
export function hasManifestInference(manifest: Manifest): manifest is Manifest & { inference: ManifestInferenceSchema };

export function validateRequiredInferenceFields(
  inf: ManifestInferenceSchema,
  modelId: string
): void;

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
