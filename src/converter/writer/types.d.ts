/**
 * RDRR Writer Type Definitions
 *
 * @module converter/writer/types
 */

import type {
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
} from '../../config/schema/index.js';

import type { ManifestInferenceSchema } from '../../config/schema/manifest.schema.js';

export declare const DEFAULT_SHARD_SIZE: number;
export declare const ALIGNMENT: number;

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

export interface TensorMetadata {
  shape: number[];
  dtype: string;
}

export interface TensorLocation extends TensorMetadata {
  shardIndex: number;
  offset: number;
  size: number;
  spans?: Array<{ shardIndex: number; offset: number; size: number }>;
  layout?: WeightLayout;
  originalShape?: number[];
  group?: string;
}

export interface ShardData {
  index: number;
  data: Uint8Array[];
  size: number;
}

export interface ShardRecord {
  index: number;
  fileName: string;
  size: number;
  hash: string;
  hashAlgorithm: HashAlgorithm;
}

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

export interface HuggingFaceTokenizer {
  model?: {
    type?: string;
    vocab?: Record<string, number> | Array<[string, number]>;
  };
}

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

export interface WriteRDRROptions extends WriterOptionsSchema {
  onProgress?: (event: ProgressEvent) => void;
  conversion?: ConversionInfoSchema;
  optimizations?: RuntimeOptimizationsSchema;
  inference?: ManifestInferenceSchema;
}

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
