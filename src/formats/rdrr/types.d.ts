/**
 * RDRR Format Types
 *
 * Core type definitions for the RDRR model format.
 *
 * @module formats/rdrr/types
 */

import type {
  HashAlgorithm as SchemaHashAlgorithm,
  ModelType as SchemaModelType,
  ComponentGroupType as SchemaComponentGroupType,
  WeightLayout as SchemaWeightLayout,
  QuantizationInfoSchema,
  ComponentGroupSchema,
  MoEConfigSchema,
  AdapterConfigSchema,
  ProvenanceSchema,
  ManifestArtifactIdentitySchema,
  ManifestWeightsRefSchema,
  KernelPathRef,
  ManifestInferenceSchema,
  type TensorRole as SchemaTensorRole,
} from '../../config/schema/index.js';

// =============================================================================
// Re-exports from Schema
// =============================================================================

export declare const RDRR_VERSION: number;
export declare const SHARD_SIZE: number;
export declare const MANIFEST_FILENAME: string;
export declare const TENSORS_FILENAME: string;

export type HashAlgorithm = SchemaHashAlgorithm;
export type ModelType = SchemaModelType;
export type ComponentGroupType = SchemaComponentGroupType;
export type WeightLayout = SchemaWeightLayout;
export type QuantizationInfo = QuantizationInfoSchema;
export type TensorRole = SchemaTensorRole;
export type ArtifactIdentity = ManifestArtifactIdentitySchema;
export type WeightsRef = ManifestWeightsRefSchema;

// =============================================================================
// Kernel Types
// =============================================================================

export type Q4KLayout = 'row' | 'col' | null;

export interface TensorSourceLocationRef {
  shard: number;
  shardIndex?: number;
  offset: number;
  size: number;
  spans?: Array<{ shard?: number; shardIndex?: number; offset: number; size: number }>;
}

export interface TensorSourceTransform {
  kind: 'affine_dequant' | 'litert_rowwise_dequant' | 'litert_axis_dequant' | 'litert_axis_blocked_dequant';
  scheme: 'per_tensor_affine' | 'per_row_affine' | 'per_axis_affine';
  sourceDtype: 'INT8' | 'UINT8' | 'INT4' | 'INT2';
  targetDtype: 'F16';
  scale?: number;
  zeroPoint?: number;
  storageEncoding?: 'signed' | 'offset_binary';
  scaleSemantics?: 'step' | 'qmax_abs';
  scaleDivisor?: number;
  storageShape?: number[];
  quantAxis?: 0 | 1;
  storageBlockSize?: number;
  storageLaneOrder?: number[];
  scaleSource?: TensorSourceLocationRef;
  rowSumSource?: TensorSourceLocationRef;
  sumSource?: TensorSourceLocationRef;
  scaleCompanionDtype?: 'UINT8';
  scaleCompanionDequant?: {
    scale: number;
    zeroPoint: number;
  };
}

export interface TensorStorageCompanion {
  role: string;
  tensorId: string;
}

export interface TensorStorageShardSpan {
  shardIndex: number;
  byteStart: number;
  byteEnd: number;
}

export interface TensorPhysicalStorageDescriptor {
  packing: 'dense' | 'q4k' | 'q4_0' | 'gguf-block-v2';
  blockShape?: number[];
  blockBytes?: number;
  companions?: TensorStorageCompanion[];
  shardSpans?: TensorStorageShardSpan[];
}

// =============================================================================
// Manifest Types
// =============================================================================

export interface ShardInfo {
  index: number;
  filename: string;
  size: number;
  hash: string;
  offset: number;
  hashAlgorithm?: HashAlgorithm;
}

export interface MoEConfig extends MoEConfigSchema {
  expertSize?: number;
}

export interface LayerConfig {
  numLayers: number;
  hiddenSize: number;
  intermediateSize: number;
  numAttentionHeads: number;
  numKeyValueHeads?: number;
  headDim?: number;
  globalHeadDim?: number;
  vocabSize: number;
  maxSeqLen: number;
  hiddenSizePerLayerInput?: number;
  vocabSizePerLayerInput?: number;
  numKvSharedLayers?: number;
}

export interface ComponentGroup extends ComponentGroupSchema {}

export interface TensorLocation {
  shard: number;
  shardIndex?: number;
  offset: number;
  size: number;
  shape: number[];
  dtype: string;
  role: TensorRole;
  group?: string;
  spans?: Array<{ shard?: number; shardIndex?: number; offset: number; size: number }>;
  layout?: WeightLayout;
  originalShape?: number[];
  sourceTransform?: TensorSourceTransform;
  storage?: TensorPhysicalStorageDescriptor;
}

export interface ConversionInfo {
  source: string;
  convertedAt: string;
  converterVersion: string;
  command?: string;
  quantization: {
    type: string;
    layout?: Q4KLayout;
    fuseGateUp?: boolean;
    quantizeEmbeddings?: boolean;
  };
  originalDtype?: string;
  notes?: string;
}

export interface RuntimeOptimizations {
  /** Preferred kernel path override */
  kernelPath?: KernelPathRef;
}

export interface IntegrityExtensionsBlockMerkle {
  blockSize: number;
  roots: Record<string, string>;
}

/**
 * Parity exactness taxonomy shared with RDRR distributed collectives and Doe TSIR
 * lowerings. Values are canonical and must not drift between the two lanes.
 *   - bit-exact-solo: hex-identical bytes against the reference interpreter.
 *   - algorithm-exact: hex-identical under a declared reduction tree (same vocabulary,
 *     declared associativity, declared accumulation dtype).
 *   - tolerance-bounded: declared metric within declared epsilon.
 */
export type LoweringExactnessClass =
  | 'bit-exact-solo'
  | 'algorithm-exact'
  | 'tolerance-bounded';

/**
 * One lowering receipt for a (kernel, backend) pair. Either digests are populated
 * (lowering succeeded) or rejectionReasons[] is populated (backend cannot honor the
 * kernel). A successful entry with rejectionReasons and a rejection entry with digests
 * are both invalid.
 */
export interface IntegrityExtensionsLoweringEntry {
  /** Logical reference into the manifest's execution graph. */
  kernelRef: string;
  /** Backend identifier, e.g. "webgpu-generic", "wse3", "csl-classifier-legacy". */
  backend: string;
  /** Hash of the target descriptor used to produce this realization. */
  targetDescriptorHash: string | null;
  /** Frontend-version identity that produced tsir.semantic (null for rejection). */
  frontendVersion: string | null;
  /** Stable source-meaning digest (null for rejection). */
  tsirSemanticDigest: string | null;
  /** Target-and-policy-dependent realization digest (null for rejection). */
  tsirRealizationDigest: string | null;
  /** Emitter-code-version digest (null for rejection). */
  emitterDigest: string | null;
  /** Doe compiler version pin (null for pre-TSIR classifier-legacy receipts). */
  doeCompilerVersion: string | null;
  /** Exactness class under which this lowering is declared equivalent to source. */
  exactnessClass: LoweringExactnessClass | null;
  /**
   * Present iff the backend refused this kernel. Canonical codes include:
   *   TSIR_SUBGROUP_UNLOWERABLE, TSIR_PE_BUDGET_EXHAUSTED,
   *   TSIR_COLLECTIVE_NOT_REPRESENTABLE, TSIR_DEPENDENCE_UNANALYZABLE,
   *   TSIR_SOURCE_NOT_AFFINE, TSIR_TARGET_UNFIT.
   */
  rejectionReasons: string[] | null;
}

export interface IntegrityExtensionsLowerings {
  /** Own contractVersion — additive, independent of integrityExtensions.contractVersion. */
  contractVersion: 1;
  entries: IntegrityExtensionsLoweringEntry[];
}

export interface IntegrityExtensions {
  contractVersion: 1;
  blockMerkle: IntegrityExtensionsBlockMerkle;
  /**
   * Optional Doe TSIR lowering receipts. Absent on artifacts built before the
   * lowering binding lands. When present, each entry is either a successful
   * lowering (digests populated, rejectionReasons null) or a rejection
   * (digests null, rejectionReasons non-empty).
   */
  lowerings?: IntegrityExtensionsLowerings;
}

export interface RDRRManifest {
  version: number;
  modelId: string;
  modelType: ModelType;
  quantization: string;
  quantizationInfo?: QuantizationInfo;
  artifactIdentity?: ArtifactIdentity;
  weightsRef?: WeightsRef;
  hashAlgorithm: HashAlgorithm;
  eos_token_id: number | number[] | null;
  image_token_id?: number;
  audio_token_id?: number;
  video_token_id?: number;
  architecture: LayerConfig | string;
  groups?: Record<string, ComponentGroup>;
  shards: ShardInfo[];
  totalSize: number;
  tensorsFile?: string;
  tensorCount?: number;
  tokenizer?: {
    type: string;
    file: string;
    vocabSize: number;
  };
  moeConfig?: MoEConfig;
  optimizations?: RuntimeOptimizations;
  config?: Record<string, unknown>;
  conversion?: ConversionInfo;
  integrityExtensions?: IntegrityExtensions;

  // Required inference configuration (populated by converter)
  inference: ManifestInferenceSchema;
  blake3Full?: string;
  defaultWeightLayout?: WeightLayout;
  metadata?: Record<string, unknown>;

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

  // LoRA adapter fields (used by adapter loading system)
  baseModel?: string;
  loraConfig?: {
    rank: number;
    alpha: number;
    targetModules?: string[];
    dropout?: number;
  };

  // Legacy inline tensors (use tensorsFile for new manifests)
  tensors?: Record<string, TensorLocation>;
}

export type TensorMap = Record<string, TensorLocation>;

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

export interface CreateManifestOptions {
  modelId: string;
  modelType: ModelType;
  quantization: string;
  quantizationInfo?: QuantizationInfo;
  artifactIdentity?: ArtifactIdentity;
  weightsRef?: WeightsRef;
  hashAlgorithm?: HashAlgorithm;
  eos_token_id?: number | number[];
  architecture: LayerConfig | string;
  groups?: Record<string, ComponentGroup>;
  shards: ShardInfo[];
  totalSize: number;
  tensorCount?: number;
  tensorsFile?: string;
  tensors?: Record<string, TensorLocation>;
  tokenizer?: { type: string; file: string; vocabSize: number };
  moeConfig?: MoEConfig;
  config?: Record<string, unknown>;
  conversion?: ConversionInfo;
  integrityExtensions?: IntegrityExtensions;
  blake3Full?: string;
  metadata?: Record<string, unknown>;
  // Required inference configuration
  inference: ManifestInferenceSchema;
  // Adapter support
  adapterType?: 'lora' | 'qlora';
  baseCompatibility?: string[];
  mergedAdapter?: AdapterConfigSchema;
  adapterConfig?: AdapterConfigSchema;
  provenance?: ProvenanceSchema;
}
