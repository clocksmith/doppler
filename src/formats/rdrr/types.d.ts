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

export type Q4KLayout = 'flat' | 'row_wise' | 'column_wise';

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
  vocabSize: number;
  maxSeqLen: number;
}

export interface ComponentGroup extends ComponentGroupSchema {}

export interface TensorLocation {
  shard: number;
  offset: number;
  size: number;
  shape: number[];
  dtype: string;
  role: TensorRole;
  group?: string;
  spans?: Array<{ shardIndex: number; offset: number; size: number }>;
  layout?: WeightLayout;
  originalShape?: number[];
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
  eos_token_id: number | number[];
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
