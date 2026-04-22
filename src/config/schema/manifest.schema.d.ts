/**
 * Manifest Schema Definitions
 *
 * Single source of truth for RDRR manifest structure.
 * Schema = type definition (what fields exist)
 *
 * @module config/schema/manifest
 */

import type { KernelPathRef } from './kernel-path.schema.js';
import type { LayerPipelineSchema, LayerType, LinearNormMode } from './inference.schema.js';
import type { EnergyModelConfigSchema } from './energy.schema.js';
import type {
  ExecutionV1GraphSchema,
  ExecutionV1SessionSchema,
} from './execution-v1.schema.js';

/** Supported hash algorithms */
export type HashAlgorithm = 'sha256' | 'blake3';

/** RDRR format version */
export declare const RDRR_VERSION: number;

/** Default shard size (64MB) */
export declare const SHARD_SIZE: number;

/** Maximum manifest header size in bytes (safety cap for `MAX_HEADER_SIZE`) */
export declare const MAX_HEADER_SIZE: number;

/** Bytes read from the head of an RDRR manifest when probing the header */
export declare const HEADER_READ_SIZE: number;

/** Default epsilon for RMSNorm when the manifest does not pin one */
export declare const DEFAULT_RMS_NORM_EPS: number;

/** Default epsilon for high-precision ops when the manifest does not pin one */
export declare const DEFAULT_HIGH_PRECISION_EPS: number;

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
  | 'diffusion'    // Diffusion pipelines (Stable Diffusion, SD3)
  | 'energy'       // Energy-based models (EBM/JEM-style demos)
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
  | 'attn'    // Attention block in hybrid
  | 'text_encoder' // Diffusion text encoders
  | 'transformer'  // Diffusion transformer (UNet/DiT)
  | 'vae';         // Diffusion VAE

/** Weight storage layout */
export type WeightLayout = 'row' | 'column';

/** Quantization value (string for forward compatibility) */
export type QuantizationValue =
  | 'q4k'      // Q4_K_M block quantization (canonical short form)
  | 'q6k'      // Q6_K block quantization
  | 'q8_0'     // Q8_0 quantization
  | 'mxfp4'    // MXFP4 quantization (MoE experts)
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
  experts?: QuantizationValue;
  expertsFormat?: string;

  // Multimodal components
  vision?: QuantizationValue;      // Vision encoder (ViT, SigLIP, CLIP)
  audio?: QuantizationValue;       // Audio encoder (Whisper, wav2vec)
  tts?: QuantizationValue;         // TTS decoder
  projector?: QuantizationValue;   // Cross-modal projection layers
  perLayerEmbeddings?: 'int4_per_row';

  // Runtime hints. `compute` may be included in variantTag when artifact naming
  // treats activation dtype as part of the published variant identity.
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

export type SourceArtifactFormat =
  | 'safetensors'
  | 'gguf'
  | 'tflite'
  | 'task'
  | 'litertlm'
  | 'direct-source'
  | (string & {});

export type ArtifactCompleteness =
  | 'complete'
  | 'weights-ref'
  | 'incomplete';

/**
 * Additive artifact identity metadata.
 *
 * During the migration, this section may be absent from legacy manifests.
 * When present, fields identify source bytes, converted weight pack identity,
 * and manifest/runtime-policy variant identity without overloading modelId.
 */
export interface ManifestArtifactIdentitySchema {
  sourceCheckpointId?: string;
  sourceRepo?: string;
  sourceRevision?: string;
  sourceFormat?: SourceArtifactFormat;
  conversionConfigPath?: string;
  conversionConfigDigest?: string;
  weightPackId?: string;
  weightPackHash?: string;
  manifestVariantId?: string;
  modalitySet?: string[];
  materializationProfile?: string;
  artifactCompleteness?: ArtifactCompleteness;
}

/**
 * Reference from a manifest variant to a shared/external weight pack.
 *
 * Runtime shard resolution does not consume this field yet. While migration is
 * in progress, a manifest with incomplete local shards must still fail unless
 * the loader path has explicit weightsRef support.
 */
export interface ManifestWeightsRefSchema {
  weightPackId: string;
  artifactRoot: string;
  manifestDigest: string;
  shardSetHash: string;
}

/** Model architecture parameters */
export interface ArchitectureSchema {
  numLayers: number;
  hiddenSize: number;
  intermediateSize: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  globalHeadDim?: number;
  vocabSize: number;
  maxSeqLen: number;
  hiddenSizePerLayerInput?: number;
  vocabSizePerLayerInput?: number;
  numKvSharedLayers?: number;
  ropeTheta?: number;
  rmsNormEps?: number;
  linearNumKeyHeads?: number;
  linearNumValueHeads?: number;
  linearKeyHeadDim?: number;
  linearValueHeadDim?: number;
  linearConvKernelDim?: number;
  linearNormMode?: LinearNormMode;
}

/**
 * Attention configuration for inference.
 * All fields required - converter must populate everything.
 * Use `null` to indicate "not applicable" (e.g., no softcapping).
 */
export interface ManifestAttentionSchema {
  /** Query pre-attention scalar: attnScale = 1/sqrt(scalar). Standard = headDim. */
  queryPreAttnScalar: number;
  /** Attention logit softcapping (Gemma 2: 50, null = disabled) */
  attnLogitSoftcapping: number | null;
  /** Sliding window size for local attention (null = full attention) */
  slidingWindow: number | null;
  /** Query-key normalization */
  queryKeyNorm: boolean;
  /** Value RMSNorm with implicit unit scale (Gemma 4 text attention) */
  valueNorm: boolean;
  /** Whether attention mask is causal (false = bidirectional attention) */
  causal: boolean;
  /** Attention bias mask enabled */
  attentionBias: boolean;
  /** Apply sigmoid gate from q_proj second half before o_proj */
  attentionOutputGate: boolean;
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
  /** Double the FFN intermediate width on KV-shared layers. */
  useDoubleWideMlp: boolean;
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
  /** Use adjacent-pair rotary layout instead of rotate-half layout for standard RoPE. */
  ropeInterleaved: boolean;
  /** Use adjacent-pair rotary layout for mRoPE section pairing. */
  mropeInterleaved: boolean;
  /** mRoPE section sizes before the Qwen doubling step. */
  mropeSection: number[] | null;
  /** Fraction of the head dimension that participates in rotary embedding. */
  partialRotaryFactor: number | null;
  /** Fraction of the local/sliding head dimension that participates in rotary embedding. */
  ropeLocalPartialRotaryFactor: number | null;
  /** Frequency exponent base dimension for full/global attention RoPE (null = use rotary dim). */
  ropeFrequencyBaseDim: number | null;
  /** Frequency exponent base dimension for local/sliding attention RoPE (null = use local rotary dim). */
  ropeLocalFrequencyBaseDim: number | null;
  /** RoPE scaling type (null = no scaling, 'linear', 'dynamic', 'yarn') */
  ropeScalingType: string | null;
  /** RoPE scaling factor (1.0 if no scaling) */
  ropeScalingFactor: number;
  /** Local RoPE scaling type for sliding window layers (null = no local scaling) */
  ropeLocalScalingType: string | null;
  /** Local RoPE scaling factor for sliding window layers (1.0 if no local scaling) */
  ropeLocalScalingFactor: number;
  /** YARN beta_fast parameter (null if not YARN scaling) */
  yarnBetaFast: number | null;
  /** YARN beta_slow parameter (null if not YARN scaling) */
  yarnBetaSlow: number | null;
  /** YARN original max position embeddings (null if not YARN scaling) */
  yarnOriginalMaxPos: number | null;
  /** Local YARN beta_fast parameter (null if not local YARN scaling) */
  ropeLocalYarnBetaFast: number | null;
  /** Local YARN beta_slow parameter (null if not local YARN scaling) */
  ropeLocalYarnBetaSlow: number | null;
  /** Local YARN original max position embeddings (null if not local YARN scaling) */
  ropeLocalYarnOriginalMaxPos: number | null;
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
  /** Optional embedding-only postprocessor stack applied after pooled hidden states. */
  embeddingPostprocessor: ManifestEmbeddingPostprocessorSchema | null;
}

export interface ManifestEmbeddingProjectionSchema {
  weightTensor: string;
  biasTensor: string | null;
  inputSize: number;
  outputSize: number;
  activation: 'identity';
}

export interface ManifestEmbeddingPostprocessorSchema {
  poolingMode: 'mean' | 'last';
  includePrompt: boolean;
  projections: ManifestEmbeddingProjectionSchema[];
  normalize: 'l2' | null;
}

/**
 * Layer pattern for hybrid attention models.
 * Defines which layers use global vs sliding window attention.
 */
export interface ManifestLayerPatternSchema {
  /** Pattern type */
  type: 'uniform' | 'alternating' | 'every_n' | 'custom';
  /** For alternating: which layers are global ('odd' or 'even'), null if not applicable */
  globalPattern: 'odd' | 'even' | null;
  /** For every_n: period of global layers, null if not applicable */
  period: number | null;
  /** For every_n: first global layer index modulo period, null if not applicable */
  offset: number | null;
  /** For custom: explicit per-layer architecture tags, null if not applicable */
  layerTypes: LayerType[] | null;
}

/**
 * Chat template configuration.
 */
export interface ManifestChatTemplateSchema {
  /** Chat template type (null = no chat template) */
  type: 'gemma' | 'gemma4' | 'llama3' | 'gpt-oss' | 'chatml' | 'qwen' | 'translategemma' | null;
  /** Whether chat template is enabled */
  enabled: boolean;
}

/**
 * Complete inference configuration embedded in manifest.
 * All fields are required - converter must populate everything.
 * Use `null` values to indicate "not applicable" or "disabled".
 */
export interface ManifestInferenceSchema {
  /** Execution contract discriminator (null = legacy inference contract). */
  schema: string | null;
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
  /** Layer pipeline override (null = use optimized hardcoded path) */
  pipeline: LayerPipelineSchema | null;
  /** Explicit session policy for execution v1 manifests */
  session: ExecutionV1SessionSchema | null;
  /** Explicit execution graph (v1 compact tuple format) */
  execution: ExecutionV1GraphSchema | null;
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

/** Canonical tensor role classification (for manifest-first loading) */
export type TensorRole =
  | 'embedding'
  | 'lm_head'
  | 'norm'
  | 'matmul'
  | 'conv'
  | 'expert'
  | 'router'
  | 'other';

/** Tensor location in shards */
export interface TensorSchema {
  shard: number;
  offset: number;
  size: number;
  shape: number[];
  dtype: string;
  role: TensorRole;
  group?: string;
  spans?: TensorSpanSchema[];
  layout?: WeightLayout;
  originalShape?: number[];
  sourceTransform?: {
    kind: 'affine_dequant';
    scheme: 'per_tensor_affine';
    sourceDtype: 'INT8' | 'UINT8' | 'INT4' | 'INT2';
    targetDtype: 'F16';
    scale: number;
    zeroPoint: number;
  } | {
    kind: 'litert_rowwise_dequant';
    scheme: 'per_row_affine';
    sourceDtype: 'INT8' | 'INT4' | 'INT2' | 'UINT8';
    targetDtype: 'F16';
    storageEncoding: 'signed' | 'offset_binary';
    scaleSemantics: 'step' | 'qmax_abs';
    scaleDivisor?: number;
    scaleSource: {
      shard: number;
      shardIndex?: number;
      offset: number;
      size: number;
      spans?: TensorSpanSchema[];
    };
    rowSumSource?: {
      shard: number;
      shardIndex?: number;
      offset: number;
      size: number;
      spans?: TensorSpanSchema[];
    };
  } | {
    kind: 'litert_axis_dequant';
    scheme: 'per_axis_affine';
    sourceDtype: 'INT8' | 'INT4' | 'INT2' | 'UINT8';
    targetDtype: 'F16';
    storageEncoding: 'signed' | 'offset_binary';
    scaleSemantics: 'step' | 'qmax_abs';
    scaleDivisor?: number;
    storageShape: [number, number];
    quantAxis: 0 | 1;
    scaleCompanionDtype?: 'UINT8';
    scaleCompanionDequant?: {
      scale: number;
      zeroPoint: number;
    };
    scaleSource: {
      shard: number;
      shardIndex?: number;
      offset: number;
      size: number;
      spans?: TensorSpanSchema[];
    };
    sumSource?: {
      shard: number;
      shardIndex?: number;
      offset: number;
      size: number;
      spans?: TensorSpanSchema[];
    };
  };
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
  /** Expert tensor format (required for MoE models) */
  expertFormat: 'mixtral' | 'gpt-oss';
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
  modelId?: string;
  sentencepieceModel?: string;
  hfModel?: string;
  allowArchFallback?: boolean;
  bosTokenId?: number;
  eosTokenId?: number;
  eosTokens?: number[];
  padTokenId?: number;
  unkTokenId?: number;
  addBosToken?: boolean;
  addEosToken?: boolean;
  specialTokens?: {
    pad?: number;
    bos?: number;
    eos?: number;
    unk?: number;
  };
}

/** Runtime optimization plan */
export interface RuntimeOptimizationsSchema {
  /** Inline kernel path override. String registry IDs were removed. */
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
  artifactIdentity?: ManifestArtifactIdentitySchema;
  weightsRef?: ManifestWeightsRefSchema;
  hashAlgorithm: HashAlgorithm;
  totalSize: number;
  eos_token_id: number | number[] | null;
  image_token_id?: number;
  audio_token_id?: number;
  video_token_id?: number;
  visionArchitecture?: 'qwen3vl' | 'gemma4' | 'siglip' | (string & {});

  // Architecture (required)
  architecture: ArchitectureSchema | string;

  // Inference configuration (required, populated by converter)
  inference: ManifestInferenceSchema;

  // Shards (required)
  shards: ShardSchema[];

  // v1: External tensor file
  tensorsFile?: string;
  tensorCount?: number;

  // v1: Component groups
  groups?: Record<string, ComponentGroupSchema>;

  // Inline tensors (deprecated in v1)
  tensors?: TensorMapSchema;

  // Optional
  config?: Record<string, unknown>;
  tokenizer?: TokenizerSchema;
  moeConfig?: MoEConfigSchema | null;
  optimizations?: RuntimeOptimizationsSchema;
  conversion?: ConversionInfoSchema;
  energy?: EnergyModelConfigSchema;

  // Adapter support (for LoRA/QLoRA)
  adapterType?: 'lora' | 'qlora';
  baseCompatibility?: string[];
  mergedAdapter?: AdapterConfigSchema;
  adapterConfig?: AdapterConfigSchema;

  // Provenance (for merged/frankenstein models)
  provenance?: ProvenanceSchema;

}

/** Check if manifest is v1 format (has groups) */
export declare function isV1Manifest(manifest: ManifestSchema): boolean;

/** Check if manifest has MoE config */
export declare function hasMoEConfig(manifest: ManifestSchema): boolean;

/**
 * Validate manifest has required inference configuration.
 * Throws if manifest is missing inference field.
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
