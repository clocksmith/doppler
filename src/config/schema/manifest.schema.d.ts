/**
 * Manifest Schema Definitions
 *
 * Single source of truth for RDRR manifest structure.
 * Schema = type definition (what fields exist)
 *
 * @module config/schema/manifest
 */

import type { KernelPathRef } from './kernel-path.schema.js';

/** Supported hash algorithms */
export type HashAlgorithm = 'sha256' | 'blake3';

/** RDRR format version */
export declare const RDRR_VERSION: number;

/** Default shard size (64MB) */
export declare const SHARD_SIZE: number;

/** External tensors filename */
export declare const TENSORS_FILENAME: string;

/** Supported model architectures */
export type ModelType =
  | 'transformer'  // Dense transformer (Llama, Gemma, Mistral, GPT)
  | 'mamba'        // Pure Mamba SSM
  | 'rwkv'         // RWKV architecture
  | 'jamba'        // Hybrid Mamba + Attention + MoE
  | 'mixtral'      // MoE transformer (Mixtral, Arctic)
  | 'deepseek'     // MoE with shared experts
  | string;        // Allow future extensions

/** Component group types */
export type ComponentGroupType =
  | 'embed'   // Embedding layer
  | 'layer'   // Dense layer (full transformer/mamba/rwkv layer)
  | 'head'    // Output head (lm_head + final_norm)
  | 'expert'  // MoE expert
  | 'shared'  // MoE shared components (router, etc.)
  | 'mamba'   // Mamba block in hybrid
  | 'rwkv'    // RWKV block
  | 'attn';   // Attention block in hybrid

/** Weight storage layout */
export type WeightLayout = 'row' | 'column';

/** Quantization value (string for forward compatibility) */
export type QuantizationValue =
  | 'q4k'      // Q4_K_M block quantization (canonical short form)
  | 'q6k'      // Q6_K block quantization
  | 'q8_0'     // Q8_0 quantization
  | 'f16'      // Float16
  | 'bf16'     // BFloat16
  | 'f32'      // Float32
  | 'fp8e4'    // Float8 E4M3
  | 'fp8e5'    // Float8 E5M2
  | 'i8'       // Int8
  | 'i4'       // Int4
  | string;    // Allow future extensions

/**
 * Quantization metadata for different weight groups.
 */
export interface QuantizationInfoSchema {
  // Core text model components
  weights: QuantizationValue;
  embeddings?: QuantizationValue;
  lmHead?: QuantizationValue;

  // Multimodal components
  vision?: QuantizationValue;      // Vision encoder (ViT, SigLIP, CLIP)
  audio?: QuantizationValue;       // Audio encoder (Whisper, wav2vec)
  tts?: QuantizationValue;         // TTS decoder
  projector?: QuantizationValue;   // Cross-modal projection layers

  // Runtime hints (NOT included in variantTag - these are runtime, not storage)
  kvCache?: QuantizationValue;
  compute?: QuantizationValue;

  // Generated variant tag for modelId suffix
  variantTag?: string;
}

/**
 * Adapter configuration for LoRA/QLoRA adapters.
 */
export interface AdapterConfigSchema {
  /** Adapter type */
  type: 'lora' | 'qlora';
  /** Adapter name/purpose (e.g., 'coding', 'roleplay', 'japanese') */
  name: string;
  /** LoRA rank */
  rank: number;
  /** LoRA alpha scaling factor */
  alpha?: number;
  /** Quantization of adapter weights */
  quant: QuantizationValue;
  /** Target modules */
  targetModules?: string[];
  /** Dropout rate during training */
  dropout?: number;
}

/**
 * Model provenance for frankenmodels and merges.
 */
export interface ProvenanceSchema {
  /** Source models used in merge */
  sources: string[];
  /** Merge method (e.g., 'slerp', 'ties', 'dare', 'linear') */
  method?: string;
  /** Merge parameters (method-specific) */
  params?: Record<string, unknown>;
  /** Adapters applied before merge */
  adapters?: string[];
  /** Original model this was derived from */
  baseModel?: string;
  /** Conversion/creation timestamp */
  createdAt?: string;
  /** Tool used for merge/conversion */
  tool?: string;
}

/** Model architecture parameters */
export interface ArchitectureSchema {
  numLayers: number;
  hiddenSize: number;
  intermediateSize: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  vocabSize: number;
  maxSeqLen: number;
  ropeTheta?: number;
  rmsNormEps?: number;
}

/**
 * Attention configuration for inference.
 * All fields required - converter must populate everything.
 * Use `null` to indicate "not applicable" (e.g., no softcapping).
 */
export interface ManifestAttentionSchema {
  /** Query pre-attention scalar (Gemma 2: 256, standard: sqrt(headDim)) */
  queryPreAttnScalar: number;
  /** Attention logit softcapping (Gemma 2: 50, null = disabled) */
  attnLogitSoftcapping: number | null;
  /** Sliding window size for local attention (null = full attention) */
  slidingWindow: number | null;
  /** Query-key normalization */
  queryKeyNorm: boolean;
}

/**
 * Normalization configuration for inference.
 * Controls RMSNorm behavior and sandwich norm architecture.
 */
export interface ManifestNormalizationSchema {
  /** RMSNorm epsilon for numerical stability (default: 1e-5) */
  rmsNormEps: number;
  /** Use (1 + weight) pattern for RMSNorm (Gemma models) */
  rmsNormWeightOffset: boolean;
  /** Has post-attention normalization (sandwich norm) */
  postAttentionNorm: boolean;
  /** Has pre-feedforward normalization (sandwich norm) */
  preFeedforwardNorm: boolean;
  /** Has post-feedforward normalization (sandwich norm) */
  postFeedforwardNorm: boolean;
}

/**
 * FFN configuration for inference.
 */
export interface ManifestFFNSchema {
  /** Activation function type */
  activation: 'silu' | 'gelu' | 'geglu' | 'swiglu' | 'relu';
  /** Whether activation is gated (e.g., SwiGLU, GeGLU) */
  gatedActivation: boolean;
  /** Clamp SwiGLU output (null = disabled) */
  swigluLimit: number | null;
}

/**
 * RoPE configuration for inference.
 * All fields required - converter must populate everything.
 * This is the canonical source for RoPE params (not architecture.ropeTheta).
 */
export interface ManifestRoPESchema {
  /** Base theta for rotary embeddings (canonical source for execution) */
  ropeTheta: number;
  /** Local theta for sliding window layers (null = same as ropeTheta) */
  ropeLocalTheta: number | null;
  /** RoPE scaling type (null = no scaling, 'linear', 'dynamic', 'yarn') */
  ropeScalingType: string | null;
  /** RoPE scaling factor (1.0 if no scaling) */
  ropeScalingFactor: number;
  /** YARN beta_fast parameter (null if not YARN scaling) */
  yarnBetaFast: number | null;
  /** YARN beta_slow parameter (null if not YARN scaling) */
  yarnBetaSlow: number | null;
  /** YARN original max position embeddings (null if not YARN scaling) */
  yarnOriginalMaxPos: number | null;
}

/**
 * Output configuration for inference.
 * All fields required - converter must populate everything.
 */
export interface ManifestOutputSchema {
  /** Final logit softcapping (Gemma 2: 30, null = disabled) */
  finalLogitSoftcapping: number | null;
  /** Whether embeddings and LM head share weights */
  tieWordEmbeddings: boolean;
  /** Scale embeddings by sqrt(hiddenSize) (Gemma models: true) */
  scaleEmbeddings: boolean;
  /** Whether embedding weights are stored as [hidden, vocab] (transpose on gather) */
  embeddingTranspose: boolean;
  /** Embedding vocab size from weight tensor (null = use architecture.vocabSize) */
  embeddingVocabSize: number | null;
}

/**
 * Layer pattern for hybrid attention models.
 * Defines which layers use global vs sliding window attention.
 */
export interface ManifestLayerPatternSchema {
  /** Pattern type */
  type: 'uniform' | 'alternating' | 'every_n';
  /** For alternating: which layers are global ('odd' or 'even'), null if not applicable */
  globalPattern: 'odd' | 'even' | null;
  /** For every_n: period of global layers, null if not applicable */
  period: number | null;
}

/**
 * Chat template configuration.
 */
export interface ManifestChatTemplateSchema {
  /** Chat template type (null = no chat template) */
  type: 'gemma' | 'llama3' | 'chatml' | 'qwen' | null;
  /** Whether chat template is enabled */
  enabled: boolean;
}

/**
 * Complete inference configuration embedded in manifest.
 * All fields are required - converter must populate everything.
 * Use `null` values to indicate "not applicable" or "disabled".
 */
export interface ManifestInferenceSchema {
  /** Attention configuration */
  attention: ManifestAttentionSchema;
  /** Normalization configuration */
  normalization: ManifestNormalizationSchema;
  /** FFN configuration */
  ffn: ManifestFFNSchema;
  /** RoPE configuration */
  rope: ManifestRoPESchema;
  /** Output configuration */
  output: ManifestOutputSchema;
  /** Layer pattern for hybrid attention */
  layerPattern: ManifestLayerPatternSchema;
  /** Chat template configuration */
  chatTemplate: ManifestChatTemplateSchema;
  /** Default kernel path for this model (null = no explicit path) */
  defaultKernelPath: string | null;
}

/**
 * Standard inference configuration template.
 */
export declare const DEFAULT_MANIFEST_INFERENCE: ManifestInferenceSchema;

/** Individual shard metadata */
export interface ShardSchema {
  index: number;
  filename: string;
  size: number;
  hash: string;
  hashAlgorithm?: HashAlgorithm;
  offset?: number;
}

/** Tensor span for multi-shard tensors */
export interface TensorSpanSchema {
  shardIndex: number;
  offset: number;
  size: number;
}

/** Tensor location in shards */
export interface TensorSchema {
  shard: number;
  offset: number;
  size: number;
  shape: number[];
  dtype: string;
  group?: string;
  spans?: TensorSpanSchema[];
  layout?: WeightLayout;
  originalShape?: number[];
}

/** External tensor map (tensors.json) */
export type TensorMapSchema = Record<string, TensorSchema>;

/** Component group for hot-swap capability */
export interface ComponentGroupSchema {
  type: ComponentGroupType;
  version: string;
  shards: number[];
  tensors: string[];
  hash: string;
  layerIndex?: number;
  expertIndex?: number;
}

/** Mixture of Experts configuration */
export interface MoEConfigSchema {
  numExperts: number;
  numExpertsPerToken: number;
  sharedExperts?: number[];
  expertShardMap?: Record<string, number[]>;
  expertTensors?: Record<string, string[]>;
  expertBytes?: number;
}

/** Tokenizer metadata */
export interface TokenizerSchema {
  type: string;
  file?: string;
  vocabSize: number;
}

/** Runtime optimization plan */
export interface RuntimeOptimizationsSchema {
  /** Preferred kernel path override */
  kernelPath?: KernelPathRef;
}

/** Conversion metadata */
export interface ConversionInfoSchema {
  source: string;
  convertedAt: string;
  tool?: string;
  version?: string;
}

/** Complete RDRR manifest structure */
export interface ManifestSchema {
  // Required fields
  version: number;
  modelId: string;
  modelType: ModelType;
  quantization: string;
  quantizationInfo?: QuantizationInfoSchema;
  hashAlgorithm: HashAlgorithm;
  totalSize: number;

  // Architecture (required, but can be string for legacy)
  architecture: ArchitectureSchema;

  // Inference configuration (required, populated by converter)
  inference: ManifestInferenceSchema;

  // Shards (required)
  shards: ShardSchema[];

  // v1: External tensor file
  tensorsFile?: string;
  tensorCount?: number;

  // v1: Component groups
  groups?: Record<string, ComponentGroupSchema>;

  // Legacy: Inline tensors (deprecated in v1)
  tensors?: TensorMapSchema;

  // Optional
  config?: Record<string, unknown>;
  tokenizer?: TokenizerSchema;
  moeConfig?: MoEConfigSchema | null;
  optimizations?: RuntimeOptimizationsSchema;
  conversion?: ConversionInfoSchema;

  // Adapter support (for LoRA/QLoRA)
  adapterType?: 'lora' | 'qlora';
  baseCompatibility?: string[];
  mergedAdapter?: AdapterConfigSchema;
  adapterConfig?: AdapterConfigSchema;

  // Provenance (for merged/frankenstein models)
  provenance?: ProvenanceSchema;

  // Legacy field aliases
  name?: string;
}

/** Check if manifest is v1 format (has groups) */
export declare function isV1Manifest(manifest: ManifestSchema): boolean;

/** Check if manifest has MoE config */
export declare function hasMoEConfig(manifest: ManifestSchema): boolean;

/**
 * Validate manifest has required inference configuration.
 * Throws if manifest is missing inference field (legacy manifest).
 */
export declare function validateManifestInference(
  manifest: { modelId: string; inference?: ManifestInferenceSchema }
): void;

/**
 * Type guard to check if manifest has inference config.
 */
export declare function hasInferenceConfig<T extends { inference?: ManifestInferenceSchema }>(
  manifest: T
): manifest is T & { inference: ManifestInferenceSchema };
