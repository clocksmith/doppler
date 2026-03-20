/**
 * core.ts - Platform-agnostic Model Conversion Core
 *
 * Shared types, pure functions, and conversion logic for RDRR format.
 * Used by both CLI (Node.js) and browser converters.
 *
 * Types are imported from config/schema for single source of truth.
 *
 * @module converter/core
 */

import type {
  TensorInfoSchema,
  ParsedModelSchema,
  RawModelConfigSchema,
  ConversionStageType as SchemaConversionStageType,
  ConversionProgressSchema,
  ConversionOptionsSchema,
  ConversionIOSchema,
  ArchitectureSchema,
  ManifestInferenceSchema,
  ShardSchema,
  TensorSpanSchema,
  TensorRole,
  TokenizerSchema,
  QuantizationInfoSchema,
  MoEConfigSchema,
  ConversionInfoSchema,
} from '../config/schema/index.js';
import type { ExecutionContractArtifact } from '../config/execution-contract-check.js';
import type {
  ManifestRequiredInferenceFieldsArtifact,
  RequiredInferenceFieldsContractArtifact,
} from '../config/required-inference-fields-contract-check.js';

export { generateShardFilename } from '../formats/rdrr/index.js';

// Re-exports for Backward Compatibility
/** @deprecated Use TensorInfoSchema from config/schema */
export type TensorInfo = TensorInfoSchema;

/** @deprecated Use ParsedModelSchema from config/schema */
export type ParsedModel = ParsedModelSchema;

/** @deprecated Use RawModelConfigSchema from config/schema */
export type ModelConfig = RawModelConfigSchema;

/** @deprecated Use ConversionStage from config/schema */
export declare const ConvertStage: {
  readonly DETECTING: 'detecting';
  readonly PARSING: 'parsing';
  readonly WRITING: 'writing';
  readonly MANIFEST: 'manifest';
  readonly COMPLETE: 'complete';
  readonly ERROR: 'error';
};

/** @deprecated Use ConversionStageType from config/schema */
export type ConvertStageType = SchemaConversionStageType;

/** @deprecated Use ConversionProgressSchema from config/schema */
export type ConvertProgress = ConversionProgressSchema;

/** @deprecated Use ConversionOptionsSchema from config/schema */
export type ConvertOptions = ConversionOptionsSchema;

/** @deprecated Use ShardSchema from config/schema */
export type ShardInfo = ShardSchema;

/** @deprecated Use TensorSpanSchema from config/schema */
export type TensorSpan = TensorSpanSchema;

/**
 * Tensor location (single shard) - local type for conversion output
 */
export interface TensorLocationSingle {
  shard: number;
  offset: number;
  size: number;
  shape: number[];
  dtype: string;
  role: TensorRole;
}

/**
 * Tensor location (multi shard) - local type for conversion output
 */
export interface TensorLocationMulti {
  spans: TensorSpan[];
  size: number;
  shape: number[];
  dtype: string;
  role: TensorRole;
}

export type TensorLocation = TensorLocationSingle | TensorLocationMulti;

/** @deprecated Use ArchitectureSchema from config/schema */
export type ArchitectureConfig = ArchitectureSchema;

/** @deprecated Use TokenizerSchema from config/schema */
export type TokenizerInfo = TokenizerSchema;

/**
 * RDRR manifest structure for conversion output
 */
export interface RDRRManifest {
  version: number | string;
  modelId: string;
  modelType: string;
  quantization: string;
  quantizationInfo?: QuantizationInfoSchema;
  architecture: ArchitectureConfig | string;
  moeConfig?: MoEConfigSchema | null;
  inference: ManifestInferenceSchema;
  shards: ShardInfo[];
  tensors: Record<string, TensorLocation>;
  totalSize: number;
  hashAlgorithm: string;
  eos_token_id: number | number[] | null;
  conversion?: ConversionInfoSchema;
  tokenizer?: TokenizerInfo;
  metadata: {
    source: string;
    convertedAt: string;
    hasTokenizer?: boolean;
  };
}

export interface CreateManifestOptions {
  source: string;
  inference?: ManifestInferenceSchema;
  modelType?: string;
  quantization?: string;
  quantizationInfo?: QuantizationInfoSchema;
  moeConfig?: MoEConfigSchema | null;
  hashAlgorithm: string;
  architecture?: ArchitectureConfig | string;
  eosTokenId?: number | number[] | null;
  convertedAt?: string | null;
  conversionInfo?: ConversionInfoSchema | null;
}

/**
 * Conversion result
 */
export interface ConvertResult {
  manifest: RDRRManifest;
  shardCount: number;
  tensorCount: number;
  totalSize: number;
  executionContractArtifact: ExecutionContractArtifact | null;
  executionV0GraphContractArtifact: null;
  layerPatternContractArtifact: Record<string, unknown> | null;
  requiredInferenceFieldsArtifact: ManifestRequiredInferenceFieldsArtifact | RequiredInferenceFieldsContractArtifact | null;
}

/** @deprecated Use ConversionIOSchema from config/schema */
export type ConvertIO = ConversionIOSchema;

// Re-export constants
export declare const SHARD_SIZE: number;
export declare const RDRR_VERSION: number;

/**
 * Sanitize model ID for filesystem/URL safety
 */
export declare function sanitizeModelId(name: string): string | null;

/**
 * Resolve bundled tokenizer vocab size from Hugging Face tokenizer.json payloads.
 */
export declare function resolveBundledTokenizerVocabSize(
  tokenizerJson: Record<string, unknown> | null | undefined
): number;

/**
 * Format bytes for human-readable display
 */
export declare function formatBytes(bytes: number): string;

/**
 * Check if tensor should be quantized based on name and shape
 */
export declare function shouldQuantize(
  tensorName: string,
  shape: number[],
  options?: { quantizeEmbeddings?: boolean | null }
): boolean;

export declare function normalizeStorageQuant(value: unknown): string | null;

export declare function resolveTensorTargetQuant(
  tensorName: string,
  fallbackQuant: unknown,
  quantizationInfo: Record<string, unknown> | null | undefined
): string | null;

export declare function transformTensorBytes(
  tensor: {
    name: string;
    shape: number[];
    dtype: string;
  },
  rawData: ArrayBuffer | Uint8Array,
  options?: {
    targetQuant?: unknown;
    quantization?: unknown;
    quantizationInfo?: Record<string, unknown> | null;
    q4kLayout?: string | null;
    quantizeEmbeddings?: boolean | null;
    forceQuantizeDecision?: boolean | null;
  }
): {
  tensorData: Uint8Array;
  outDtype: string;
  outLayout: string | null;
  sourceDtype: string;
  tensorTargetQuant: string | null;
};

/**
 * Extract architecture configuration from model config
 */
export declare function extractArchitecture(
  config: ModelConfig,
  ggufConfig?: Record<string, unknown>
): ArchitectureConfig;

/**
 * Build tensor location map for manifest
 */
export declare function buildTensorMap(
  tensors: Array<{ name: string; shape: number[]; dtype: string; size: number }>,
  shardSize: number
): Record<string, TensorLocation>;

/**
 * Create RDRR manifest from model info and shards
 */
export declare function createManifest(
  modelId: string,
  model: ParsedModel,
  shards: ShardInfo[],
  tensorLocations: Record<string, TensorLocation>,
  source: string
): RDRRManifest;
export declare function createManifest(
  modelId: string,
  model: ParsedModel,
  shards: ShardInfo[],
  tensorLocations: Record<string, TensorLocation>,
  options: CreateManifestOptions
): RDRRManifest;

/**
 * Convert a parsed model to RDRR format
 */
export declare function convertModel(
  model: ParsedModel,
  io: ConvertIO,
  options?: ConvertOptions
): Promise<ConvertResult>;
