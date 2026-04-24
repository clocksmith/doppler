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
import type { ExecutionV1PerLayerInputsSessionSchema } from '../../../config/schema/execution-v1.schema.js';
import type { RuntimeModelContract } from '../../runtime-model.js';

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
  num_global_key_value_heads?: number;
  num_global_kv_heads?: number;
  attentionHeadCountKV?: number;
  head_dim?: number;
  global_head_dim?: number;
  intermediate_size?: number;
  use_double_wide_mlp?: boolean;
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

export type Manifest = RuntimeModelContract;

export interface ParsedModelConfig {
  numLayers: number;
  hiddenSize: number;
  intermediateSize: number;
  intermediateSizes: number[];
  maxIntermediateSize: number;
  numHeads: number;
  numKVHeads: number;
  numGlobalKVHeads: number | null;
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
  ropeFrequencyBaseDim: number;
  ropeLocalFrequencyBaseDim: number;
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
  useDoubleWideMlp: boolean;
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
  valueNorm: boolean;
  attentionOutputGate: boolean;
  queryPreAttnScalar: number;
  layerPipeline?: LayerPipelineSchema | null;
  chatTemplateType?: string | null;
  chatTemplateEnabled: boolean;
  chatTemplateThinking: boolean | null;
  decodeStrategy: 'incremental' | 'replay_prefill';
  perLayerInputsSession: ExecutionV1PerLayerInputsSessionSchema;
  kernelPath?: KernelPathRef;
  visionConfig?: VisionConfig | null;
  audioConfig?: AudioEncoderConfig | null;
}

export interface VisionConfig {
  depth: number;
  hiddenSize: number;
  intermediateSize: number;
  numHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  outHiddenSize: number | null;
  patchSize: number;
  poolingKernelSize: number;
  spatialMergeSize: number | null;
  temporalPatchSize: number | null;
  positionEmbeddingSize: number | null;
  defaultOutputLength: number | null;
  ropeTheta: number | null;
  eps: number;
  hiddenActivation: string;
  standardize: boolean;
  useClippedLinears: boolean;
  minPixels: number;
  maxPixels: number;
  normalization: {
    mean: number[];
    std: number[];
  };
  deepstackVisualIndexes: number[];
  imageTokenId: number | null;
  visionArchitecture: 'gemma4' | 'qwen3vl';
  softTokenBudgetTiers?: number[];
}

export interface AudioEncoderConfig {
  audioArchitecture: 'gemma4';
  depth: number;
  hiddenSize: number;
  numAttentionHeads: number;
  headDim: number;
  convKernelSize: number;
  subsamplingConvChannels: number[];
  outputProjDims: number;
  attentionContextLeft: number;
  attentionContextRight: number;
  attentionChunkSize: number;
  attentionLogitCap: number;
  attentionInvalidLogitsValue: number;
  residualWeight: number;
  rmsNormEps: number;
  hiddenAct: string;
  useClippedLinears: boolean;
  audioTokenId: number | null;
}

export function getStopTokenIds(manifest: Manifest): number[];
export function resolveLayerIntermediateSize(config: ParsedModelConfig, layerIdx: number): number;

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

export function validateRequiredInferenceFields(
  inf: ManifestInferenceSchema,
  modelId: string
): void;

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
