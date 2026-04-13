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
  kind: 'affine_dequant' | 'litert_rowwise_dequant' | 'litert_axis_dequant';
  scheme: 'per_tensor_affine' | 'per_row_affine' | 'per_axis_affine';
  sourceDtype: 'INT8' | 'UINT8' | 'INT4' | 'INT2';
  targetDtype: 'F16';
  scale?: number;
  zeroPoint?: number;
  storageEncoding?: 'signed' | 'offset_binary';
  storageShape?: number[];
  quantAxis?: 0 | 1;
  scaleSource?: TensorSourceLocationRef;
  rowSumSource?: TensorSourceLocationRef;
  sumSource?: TensorSourceLocationRef;
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

export interface RDRRManifest {
  version: number;
  modelId: string;
  modelType: ModelType;
  quantization: string;
  quantizationInfo?: QuantizationInfo;
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
