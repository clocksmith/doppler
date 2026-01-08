/**
 * RDRR Writer Type Definitions
 *
 * All interfaces and types used by the RDRR model format writer.
 * Types imported from config/schema for single source of truth.
 *
 * @module converter/writer/types
 */

import {
  // Constants
  SHARD_SIZE as SCHEMA_SHARD_SIZE,
  DEFAULT_STORAGE_ALIGNMENT_CONFIG,
  // Schema types
  type HashAlgorithm,
  type ModelType,
  type WeightLayout,
  type MoEConfigSchema,
  type ComponentGroupSchema,
  type TensorMapSchema,
  type ConversionInfoSchema,
  type RuntimeOptimizationsSchema,
  type QuantizationInfoSchema,
  type TensorInfoSchema,
  type WriterOptionsSchema,
  type WriteResultSchema,
} from '../../config/schema/index.js';

import type { ManifestInferenceSchema } from '../../config/schema/manifest.schema.js';

// ============================================================================
// Constants
// ============================================================================

export const DEFAULT_SHARD_SIZE = SCHEMA_SHARD_SIZE;
// Use config value for alignment (default: 4KB for optimal disk I/O)
export const ALIGNMENT = DEFAULT_STORAGE_ALIGNMENT_CONFIG.bufferAlignmentBytes;

// ============================================================================
// Re-exports for Backward Compatibility
// ============================================================================

/** @deprecated Use WriterOptionsSchema from config/schema */
export type WriterOptions = WriterOptionsSchema;

/** @deprecated Use WriteResultSchema from config/schema */
export type WriteResult = WriteResultSchema;

/** @deprecated Use TensorInfoSchema from config/schema */
export type TensorInfo = TensorInfoSchema;

/** @deprecated Use MoEConfigSchema from config/schema */
export type MoEConfig = MoEConfigSchema;

/** @deprecated Use ComponentGroupSchema from config/schema */
export type ComponentGroup = ComponentGroupSchema;

/** @deprecated Use TensorMapSchema from config/schema */
export type TensorMap = TensorMapSchema;

/** @deprecated Use ConversionInfoSchema from config/schema */
export type ConversionInfo = ConversionInfoSchema;

/** @deprecated Use RuntimeOptimizationsSchema from config/schema */
export type RuntimeOptimizations = RuntimeOptimizationsSchema;

// ============================================================================
// Local Types (Writer-specific)
// ============================================================================

/** Tensor shape and data type metadata */
export interface TensorMetadata {
  shape: number[];
  dtype: string;
}

/** Location of a tensor within shards */
export interface TensorLocation extends TensorMetadata {
  shardIndex: number;
  offset: number;
  size: number;
  spans?: Array<{ shardIndex: number; offset: number; size: number }>;
  layout?: WeightLayout;
  originalShape?: number[];
  group?: string;
}

/** In-memory shard data being built */
export interface ShardData {
  index: number;
  data: Uint8Array[];
  size: number;
}

/** Finalized shard record with hash */
export interface ShardRecord {
  index: number;
  fileName: string;
  size: number;
  hash: string;
  hashAlgorithm: HashAlgorithm;
}

/** Tokenizer configuration extracted from source model */
export interface TokenizerConfig {
  model?: string;
  tokens?: string[];
  merges?: string[];
  scores?: number[];
  tokenTypes?: number[];
  bosTokenId?: number;
  eosTokenId?: number;
  padTokenId?: number;
  unkTokenId?: number;
  sepTokenId?: number;
  clsTokenId?: number;
  maskTokenId?: number;
  addBosToken?: boolean;
  addEosToken?: boolean;
  addSpacePrefix?: boolean;
}

/** HuggingFace tokenizer.json format */
export interface HuggingFaceTokenizer {
  model?: {
    type?: string;
    vocab?: Record<string, number> | Array<[string, number]>;
  };
}

/** Model information for conversion */
export interface ModelInfo {
  modelName?: string;
  architecture?: string;
  quantization?: string;
  quantizationInfo?: QuantizationInfoSchema;
  config?: Record<string, unknown>;
  tokenizer?: TokenizerConfig;
  tokenizerConfig?: TokenizerConfig;
  tokenizerJson?: HuggingFaceTokenizer;
  tensors: TensorInfo[];
}

/** Progress event during conversion */
export interface ProgressEvent {
  stage: 'writing' | 'complete';
  current?: number;
  total?: number;
  tensorName?: string;
  manifestPath?: string;
  shardCount?: number;
  totalSize?: number;
  tensorCount?: number;
}

/** Options for writeRDRR function */
export interface WriteRDRROptions extends WriterOptionsSchema {
  onProgress?: (event: ProgressEvent) => void;
  /** Conversion metadata - how the model was generated */
  conversion?: ConversionInfoSchema;
  /** Runtime optimizations including kernel path overrides */
  optimizations?: RuntimeOptimizationsSchema;
  /** Model-specific inference configuration (from preset) */
  inference?: ManifestInferenceSchema;
}

// Re-export schema types that are commonly used
export type {
  HashAlgorithm,
  ModelType,
  WeightLayout,
  MoEConfigSchema,
  ComponentGroupSchema,
  TensorMapSchema,
  ConversionInfoSchema,
  RuntimeOptimizationsSchema,
  QuantizationInfoSchema,
  TensorInfoSchema,
  WriterOptionsSchema,
  WriteResultSchema,
  ManifestInferenceSchema,
};
