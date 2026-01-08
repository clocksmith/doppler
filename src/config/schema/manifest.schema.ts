/**
 * Manifest Schema Definitions
 *
 * Single source of truth for RDRR manifest structure.
 * Schema = type definition (what fields exist)
 *
 * @module config/schema/manifest
 */

import type { KernelPathRef } from './kernel-path.schema.js';

// =============================================================================
// Hash & Versioning
// =============================================================================

/** Supported hash algorithms */
export type HashAlgorithm = 'sha256' | 'blake3';

/** RDRR format version */
export const RDRR_VERSION = 1;

/** Default shard size (64MB) */
export const SHARD_SIZE = 64 * 1024 * 1024;

/** External tensors filename */
export const TENSORS_FILENAME = 'tensors.json';

// =============================================================================
// Model Types
// =============================================================================

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
 *
 * Naming convention (storage-only, no runtime info):
 * - Base model: `{name}-w{weights}[-e{embeddings}][-h{head}][-v{vision}][-a{audio}][-t{tts}][-p{projector}]`
 * - Standalone adapter: `{base}-w{quant}+{type}-{name}-{quant}r{rank}`
 * - Merged adapter: `{name}-w{weights}~{type}-{name}-{quant}r{rank}`
 *
 * Examples:
 * - `gemma-2b-wq4k` (weights Q4K, embeddings default to weights)
 * - `gemma-2b-wq4k-ef16` (weights Q4K, embeddings F16)
 * - `llama-8b-wq4k-ef16-hf16` (with separate head)
 * - `qwen2-vl-7b-wq4k-vf16-pf16` (multimodal with vision + projector)
 * - `gemma-2b-wq4k+lora-coding-f16r16` (standalone adapter)
 * - `gemma-2b-wq4k~lora-coding-f16r16` (merged adapter)
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

// =============================================================================
// Architecture Schema
// =============================================================================

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

// =============================================================================
// Inference Schema (Model-Specific Inference Parameters)
// =============================================================================

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
}

/**
 * Layer pattern for hybrid attention models.
 * Defines which layers use global vs sliding window attention.
 */
export interface ManifestLayerPatternSchema {
  /** Pattern type */
  type: 'uniform' | 'alternating' | 'every_n';
  /** For alternating: which layers are global ('odd' or 'even') */
  globalPattern?: 'odd' | 'even';
  /** For every_n: period of global layers */
  period?: number;
}

/**
 * Complete inference configuration embedded in manifest.
 *
 * This captures all model-specific inference parameters that were previously
 * scattered across model presets. By embedding these in the manifest at
 * conversion time, the manifest becomes the single source of truth for
 * how to run inference on this model.
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
  layerPattern?: ManifestLayerPatternSchema;
  /** Default kernel path for this model (e.g., 'gemma2-q4k-fused') */
  defaultKernelPath?: string;
}

/**
 * Standard inference configuration template.
 *
 * PURPOSE: Converter template and test fixtures ONLY.
 * NOT a runtime fallback - if manifest is missing fields, validation fails.
 *
 * These values represent a "standard transformer" (no special features).
 * Converter uses this as a base, then overrides for specific model families.
 */
export const DEFAULT_MANIFEST_INFERENCE: ManifestInferenceSchema = {
  attention: {
    queryPreAttnScalar: 8,  // sqrt(64) for standard 64-dim heads
    attnLogitSoftcapping: null,  // No softcapping
    slidingWindow: null,  // Full attention
    queryKeyNorm: false,
  },
  normalization: {
    rmsNormWeightOffset: false,
    postAttentionNorm: false,
    preFeedforwardNorm: false,
    postFeedforwardNorm: false,
  },
  ffn: {
    activation: 'silu',
    gatedActivation: true,
  },
  rope: {
    ropeTheta: 10000,
    ropeLocalTheta: null,  // Same as ropeTheta
    ropeScalingType: null,  // No scaling
    ropeScalingFactor: 1.0,
    yarnBetaFast: null,  // No YARN
    yarnBetaSlow: null,
    yarnOriginalMaxPos: null,
  },
  output: {
    finalLogitSoftcapping: null,  // No softcapping
    tieWordEmbeddings: false,
    scaleEmbeddings: false,
  },
};

// =============================================================================
// Shard Schema
// =============================================================================

/** Individual shard metadata */
export interface ShardSchema {
  index: number;
  filename: string;
  size: number;
  hash: string;
  hashAlgorithm?: HashAlgorithm;
  offset?: number;
}

// =============================================================================
// Tensor Schema
// =============================================================================

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

// =============================================================================
// Component Group Schema
// =============================================================================

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

// =============================================================================
// MoE Schema
// =============================================================================

/** Mixture of Experts configuration */
export interface MoEConfigSchema {
  numExperts: number;
  numExpertsPerToken: number;
  sharedExperts?: number[];
  expertShardMap?: Record<string, number[]>;
  expertTensors?: Record<string, string[]>;
  expertBytes?: number;
}

// =============================================================================
// Tokenizer Schema
// =============================================================================

/** Tokenizer metadata */
export interface TokenizerSchema {
  type: string;
  file?: string;
  vocabSize: number;
}

// =============================================================================
// Runtime Optimizations Schema
// =============================================================================

/** Runtime optimization plan */
export interface RuntimeOptimizationsSchema {
  /** Preferred kernel path override */
  kernelPath?: KernelPathRef;
}

// =============================================================================
// Conversion Metadata Schema
// =============================================================================

/** Conversion metadata */
export interface ConversionInfoSchema {
  source: string;
  convertedAt: string;
  tool?: string;
  version?: string;
}

// =============================================================================
// Full Manifest Schema
// =============================================================================

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
  architecture: ArchitectureSchema | string;

  // Inference configuration (required, populated by converter)
  // Contains all model-specific inference parameters
  // Manifests without this field are invalid and must be re-converted
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
  /** Adapter type - present only for adapter manifests */
  adapterType?: 'lora' | 'qlora';
  /** Base model compatibility - required for adapter manifests */
  baseCompatibility?: string[];
  /** Merged adapter info - present when adapter is baked into weights */
  mergedAdapter?: AdapterConfigSchema;
  /** Adapter config - full config for standalone adapter manifests */
  adapterConfig?: AdapterConfigSchema;

  // Provenance (for merged/frankenstein models)
  provenance?: ProvenanceSchema;

  // Legacy field aliases
  name?: string;
}

// =============================================================================
// Validation Helpers
// =============================================================================

/** Check if manifest is v1 format (has groups) */
export function isV1Manifest(manifest: ManifestSchema): boolean {
  return manifest.version === 1 && !!manifest.groups;
}

/** Check if manifest has MoE config */
export function hasMoEConfig(manifest: ManifestSchema): boolean {
  return manifest.moeConfig != null && manifest.moeConfig.numExperts > 1;
}

/**
 * Validate manifest has required inference configuration.
 * Throws if manifest is missing inference field (legacy manifest).
 *
 * @throws Error if manifest.inference is missing
 */
export function validateManifestInference(
  manifest: { modelId: string; inference?: ManifestInferenceSchema }
): void {
  if (!manifest.inference) {
    throw new Error(
      `Manifest for "${manifest.modelId}" is missing required 'inference' field. ` +
      `This model was converted with an older version of DOPPLER. ` +
      `Please re-convert the model using the latest converter.`
    );
  }
}

/**
 * Type guard to check if manifest has inference config.
 * Use validateManifestInference() to fail fast; this is for conditional checks.
 */
export function hasInferenceConfig<T extends { inference?: ManifestInferenceSchema }>(
  manifest: T
): manifest is T & { inference: ManifestInferenceSchema } {
  return manifest.inference != null;
}
