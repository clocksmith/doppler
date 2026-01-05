/**
 * Manifest Schema Definitions
 *
 * Single source of truth for RDRR manifest structure.
 * Schema = type definition (what fields exist)
 *
 * @module config/schema/manifest
 */

import type { KernelPlanSchema } from './kernel-plan.schema.js';

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
  kernelPlan?: KernelPlanSchema;
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
