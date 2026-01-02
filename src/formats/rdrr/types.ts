/**
 * RDRR Format Types
 *
 * Core type definitions for the RDRR model format.
 *
 * @module formats/rdrr/types
 */

import {
  RDRR_VERSION as SCHEMA_VERSION,
  SHARD_SIZE as SCHEMA_SHARD_SIZE,
  TENSORS_FILENAME as SCHEMA_TENSORS_FILENAME,
  type HashAlgorithm as SchemaHashAlgorithm,
  type ModelType as SchemaModelType,
  type ComponentGroupType as SchemaComponentGroupType,
  type WeightLayout as SchemaWeightLayout,
  type QuantizationInfoSchema,
  type ShardSchema,
  type ComponentGroupSchema,
  type MoEConfigSchema,
} from '../../config/schema/index.js';

// =============================================================================
// Re-exports from Schema
// =============================================================================

export const RDRR_VERSION = SCHEMA_VERSION;
export const SHARD_SIZE = SCHEMA_SHARD_SIZE;
export const MANIFEST_FILENAME = 'manifest.json';
export const TENSORS_FILENAME = SCHEMA_TENSORS_FILENAME;

export type HashAlgorithm = SchemaHashAlgorithm;
export type ModelType = SchemaModelType;
export type ComponentGroupType = SchemaComponentGroupType;
export type WeightLayout = SchemaWeightLayout;
export type QuantizationInfo = QuantizationInfoSchema;

// =============================================================================
// Kernel Types
// =============================================================================

export type AttentionKernel = 'auto' | 'tiled_large' | 'tiled_small' | 'streaming';
export type MatmulKernel = 'auto' | 'fused_q4k' | 'dequant_f16' | 'dequant_f32' | 'gemv_subgroup';
export type Q4KLayout = 'flat' | 'row_wise' | 'column_wise';
export type ComputePrecision = 'f16' | 'f32' | 'auto';

// =============================================================================
// Manifest Types
// =============================================================================

export interface ShardInfo extends Omit<ShardSchema, 'hashAlgorithm'> {
  blake3?: string;
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
  group?: string;
  spans?: Array<{ shardIndex: number; offset: number; size: number }>;
  layout?: WeightLayout;
  originalShape?: number[];
}

export interface KernelHints {
  computePrecision?: ComputePrecision;
  q4kMatmul?: MatmulKernel;
  f16Matmul?: MatmulKernel;
  attentionPrefill?: AttentionKernel;
  attentionDecode?: AttentionKernel;
  tunedDevice?: string;
  benchmarkTokPerSec?: number;
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
  attentionKernel?: AttentionKernel;
  kernelHints?: KernelHints;
}

export interface LoRAConfig {
  rank: number;
  alpha: number;
  targetModules?: string[];
  dropout?: number;
}

export interface RDRRManifest {
  version: number | string;
  modelId: string;
  modelType: ModelType;
  quantization: string;
  quantizationInfo?: QuantizationInfo;
  hashAlgorithm?: HashAlgorithm;
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
  blake3Full?: string;
  defaultWeightLayout?: WeightLayout;
  metadata?: Record<string, unknown>;
  adapterType?: 'lora';
  baseModel?: string;
  loraConfig?: LoRAConfig;
  /** @deprecated Use tensorsFile */
  tensors?: Record<string, TensorLocation>;
  /** @deprecated Use modelId */
  name?: string;
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
}
