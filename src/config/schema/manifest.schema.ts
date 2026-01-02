/**
 * Manifest Schema Definitions
 *
 * Single source of truth for RDRR manifest structure.
 * Schema = type definition (what fields exist)
 *
 * @module config/schema/manifest
 */

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
  | 'q4_k_m'
  | 'q6_k'
  | 'q8_0'
  | 'f16'
  | 'bf16'
  | 'f32'
  | string;

/** Quantization metadata for different weight groups */
export interface QuantizationInfoSchema {
  weights: QuantizationValue;
  embeddings?: QuantizationValue;
  lmHead?: QuantizationValue;
  activations?: QuantizationValue;
  kvCache?: QuantizationValue;
  compute?: QuantizationValue;
  variantTag?: string;
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

/** Kernel hints for runtime optimization */
export interface KernelHintsSchema {
  preferredKernels?: Record<string, string>;
  workgroupOverrides?: Record<string, [number, number, number]>;
  disableFeatures?: string[];
  forceF32Accumulation?: boolean;
  attentionTier?: 'tiled' | 'streaming' | 'basic';
  targetDevice?: string;
  q4kMatmul?: 'fused' | 'dequant';
}

/** Runtime optimization hints */
export interface RuntimeOptimizationsSchema {
  kernelHints?: KernelHintsSchema;
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
