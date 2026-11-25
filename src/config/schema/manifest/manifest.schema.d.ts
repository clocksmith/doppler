export declare const DEFAULT_LINEAR_ATTENTION_SOURCE_LAYOUT: 'hf';
import type { FP8Policy } from './fp8.schema.js';
import type { W8A8Policy } from './w8a8.schema.js';
/**
 * Manifest Schema Definitions
 *
 * Single source of truth for RDRR manifest structure.
 * Schema = type definition (what fields exist)
 *
 * @module config/schema/manifest
 */

import type { ManifestRoPESchema } from './manifest-rope.schema.js';
export type { ManifestRoPESchema } from './manifest-rope.schema.js';
import type { KernelPathRef } from "../execution/kernel-path.schema.js";
import type { LayerPipelineSchema, LayerType, LinearNormGateActivation, LinearNormMode } from "../execution/inference.schema.js";
import type { EnergyModelConfigSchema } from "../execution/energy.schema.js";
import type { HyperConnectionsSchema, IndexedAttentionSchema } from "../execution/hyper-connection-decoder.schema.js";
import type { NgramEmbeddingSchema } from "../execution/ngram-embedding.schema.js";
import type {
  ExecutionV1GraphSchema,
  ExecutionV1SessionSchema,
} from "../execution/execution-v1.schema.js";

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
  | 'diffusion_gemma' // Block-diffusion Gemma text generation
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
  | 'q4_1' | 'q5_0' | 'q8_0' | 'q6k' | 'q5k'
  | 'q4_0'     // GGUF Q4_0 block quantization
  | 'q6k' | 'q5k'      // Q6_K block quantization
  | 'q8_0'     // Q8_0 quantization
  | 'w8a8'     // Symmetric per-channel weights and per-token activations
  | 'w4a16'    // Packed W4A16 quantization
  | 'wna8o8'   // Mobile wNa8o8 quantization
  | 'mxfp4'    // MXFP4 quantization (MoE experts)
  | 'f16'      // Float16
  | 'bf16'     // BFloat16
  | 'f32'      // Float32
  | 'fp8e4'    // Float8 E4M3
  | 'fp8e5'    // Float8 E5M2
  | 'i8'       // Int8
  | 'i4'       // Int4
  | string;    // Allow future extensions

export type SourceTrainingQuantization =
  | 'qat'
  | 'ptq';

export type SourceQuantizationTarget =
  | 'q4_1' | 'q5_0' | 'q8_0' | 'q6k' | 'q5k'
  | 'q4_0'
  | 'w4a16'
  | 'wNa8o8';

export type SourceQuantizationFormat =
  | 'compressed-tensors'
  | 'gguf';

/**
 * Quantization metadata for different weight groups.
 */
export interface QuantizationInfoSchema {
  // Core text model components
  weights: QuantizationValue;
  w8a8?: W8A8Policy;
  fp8?: FP8Policy;
  embeddings?: QuantizationValue;
  lmHead?: QuantizationValue;
  sourceTrainingQuantization?: SourceTrainingQuantization;
  sourceQuantizationTarget?: SourceQuantizationTarget;
  preserveSourceTensors?: boolean;
  sourceQuantizationFormat?: SourceQuantizationFormat;
  experts?: QuantizationValue;
  expertsFormat?: string;

  // Multimodal components
  vision?: QuantizationValue;      // Vision encoder (ViT, SigLIP, CLIP)
  audio?: QuantizationValue;       // Audio encoder (Whisper, wav2vec)
  tts?: QuantizationValue;         // TTS decoder
  projector?: QuantizationValue;   // Cross-modal projection layers
  perLayerEmbeddings?: 'int4-per-row';

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
 * When present, fields identify source bytes, converted weight capsule identity,
 * and manifest/runtime-policy variant identity without overloading modelId.
 */
export interface ManifestArtifactIdentitySchema {
  sourceCheckpointId?: string;
  sourceRepo?: string;
  sourceRevision?: string;
  sourceFormat?: SourceArtifactFormat;
  conversionConfigPath?: string;
  conversionConfigDigest?: string;
  weightCapsuleId?: string;
  weightCapsuleHash?: string;
  shardSetHash?: string;
  manifestVariantId?: string;
  modalitySet?: string[];
  materializationProfile?: string;
  artifactCompleteness?: ArtifactCompleteness;
}

/**
 * Reference from a manifest variant to a shared/external weight capsule.
 *
 * Runtime shard resolution does not consume this field yet. While migration is
 * in progress, a manifest with incomplete local shards must still fail unless
 * the loader path has explicit weightsRef support.
 */
export interface ManifestWeightsRefSchema {
  weightCapsuleId: string;
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
  numGlobalKeyValueHeads?: number;
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
  linearAttentionSourceLayout?: 'hf' | 'gguf-v1';
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
  storage?: {
    packing: 'dense' | 'q4k' | 'q4_0' | 'w4a16' | 'gguf-block-v2' | string;
    blockShape?: number[];
    blockBytes?: number;
    companions?: Array<{
      role: string;
      tensorId: string;
    }>;
    shardSpans?: Array<{
      shardIndex: number;
      byteStart: number;
      byteEnd: number;
    }>;
  };
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
    storageEncoding: 'signed' | 'offset-binary';
    scaleSemantics: 'step' | 'qmax-abs';
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
    storageEncoding: 'signed' | 'offset-binary';
    scaleSemantics: 'step' | 'qmax-abs';
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
  expertFormat: 'mixtral' | 'gpt-oss' | 'gemma4' | 'qwen3_5';
  /** Expert hidden width when it differs from architecture.intermediateSize. */
  expertIntermediateSize?: number;
  sharedExpertIntermediateSize?: number;
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

/** Check if a manifest supports embedding workloads. */
export declare function modelSupportsEmbedding(
  manifest: Partial<ManifestSchema> | null | undefined
): boolean;

/** Check if a manifest supports typed biological sequence workloads. */
export declare function modelSupportsSequence(
  manifest: Partial<ManifestSchema> | null | undefined
): boolean;

/** Check if a manifest supports rerank workloads. */
export declare function modelSupportsRerank(
  manifest: Partial<ManifestSchema> | null | undefined
): boolean;

/** Check if a manifest supports audio transcription workloads. */
export declare function modelSupportsTranscription(
  manifest: Partial<ManifestSchema> | null | undefined
): boolean;

/** Check if a manifest supports vision workloads. */
export declare function modelSupportsVision(
  manifest: Partial<ManifestSchema> | null | undefined
): boolean;

/** Supported linear-attention gated-RMSNorm activations, in canonical spelling. */
export declare const LINEAR_NORM_GATE_ACTIVATIONS: readonly LinearNormGateActivation[];

/**
 * Resolve the declared linear-attention norm-gate activation of a manifest
 * normalization section. An absent field resolves to the schema default
 * (`silu`, the meaning of every manifest converted before the field existed);
 * any other value, including null, must be a supported activation or throws.
 */
export declare function resolveLinearNormGateActivation(
  normalization: Partial<ManifestNormalizationSchema>,
  label?: string
): LinearNormGateActivation;

export type { ManifestAttentionSchema, ManifestNormalizationSchema, ManifestFFNSchema, ManifestOutputSchema, ManifestEmbeddingNormalizationSchema, ManifestSequencePoolingSchema, ManifestSequenceSchema, ManifestEmbeddingProjectionSchema, ManifestEmbeddingPostprocessorSchema, ManifestLayerPatternSchema, ManifestChatTemplateSchema, ManifestUnsupportedSchema, ManifestDiffusionGemmaSchema, ManifestRerankSchema, ManifestInferenceSchema } from './manifest-inference.schema.js';
import type { ManifestInferenceSchema, ManifestNormalizationSchema } from './manifest-inference.schema.js';
