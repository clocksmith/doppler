/**
 * Conversion Schema Definitions
 *
 * Types for model format conversion (GGUF/SafeTensors → RDRR).
 *
 * @module config/schema/conversion
 */

import type {
  ArchitectureSchema,
  HashAlgorithm,
  ModelType,
  WeightLayout,
  QuantizationInfoSchema,
} from './manifest.schema.js';

/** Tensor information from source format */
export interface TensorInfoSchema {
  name: string;
  shape: number[];
  dtype: string;
  size: number;
  offset?: number;
  /** Platform-specific source reference */
  _source?: unknown;
}

/** Parsed model ready for conversion */
export interface ParsedModelSchema {
  tensors: TensorInfoSchema[];
  config: RawModelConfigSchema;
  architecture?: string;
  quantization?: string;
  tokenizerJson?: unknown;
}

/** Raw config from source (HuggingFace or GGUF style) */
export interface RawModelConfigSchema {
  // HuggingFace style
  architectures?: string[];
  model_type?: string;
  hidden_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  intermediate_size?: number;
  vocab_size?: number;
  max_position_embeddings?: number;
  rope_theta?: number;
  rms_norm_eps?: number;
  head_dim?: number;
  _name_or_path?: string;

  // GGUF style
  n_layer?: number;
  n_embd?: number;
  n_head?: number;
  n_inner?: number;
  n_positions?: number;

  // MoE
  num_local_experts?: number;
  num_experts?: number;
  n_shared_experts?: number;

  // Allow additional fields
  [key: string]: unknown;
}

/** Quantization target types */
export type QuantizationType = 'q4_k_m' | 'q6_k' | 'q8_0' | 'f16' | 'f32' | null;

/** Conversion options */
export interface ConversionOptionsSchema {
  /** Output model ID */
  modelId?: string;
  /** Target quantization */
  quantize?: QuantizationType;
  /** Also quantize embeddings */
  quantizeEmbeddings?: boolean;
  /** Shard size in bytes */
  shardSize?: number;
  /** Progress callback */
  onProgress?: (progress: ConversionProgressSchema) => void;
  /** Abort signal */
  signal?: AbortSignal;
}

/** Conversion stages */
export declare const ConversionStage: {
  readonly DETECTING: 'detecting';
  readonly PARSING: 'parsing';
  readonly QUANTIZING: 'quantizing';
  readonly WRITING: 'writing';
  readonly MANIFEST: 'manifest';
  readonly COMPLETE: 'complete';
  readonly ERROR: 'error';
};

export type ConversionStageType = (typeof ConversionStage)[keyof typeof ConversionStage];

/** Conversion progress */
export interface ConversionProgressSchema {
  stage: ConversionStageType;
  message: string;
  format?: string;
  modelId?: string;
  tensorCount?: number;
  totalSize?: string;
  current?: number;
  total?: number;
  percent?: number;
  shardCount?: number;
  error?: Error;
}

/** RDRR writer options */
export interface WriterOptionsSchema {
  shardSize?: number;
  hashAlgorithm?: HashAlgorithm;
  modelId?: string;
  modelType?: ModelType;
  architecture?: ArchitectureSchema;
  quantization?: string;
  quantizationInfo?: QuantizationInfoSchema;
  /** Pre-transpose weights for column-major access */
  transposeWeights?: boolean;
  /** Fuse gate+up projections for FFN */
  fuseGateUp?: boolean;
}

/** Tensor location after writing */
export interface TensorLocationSchema {
  shardIndex: number;
  offset: number;
  size: number;
  shape: number[];
  dtype: string;
  spans?: Array<{ shardIndex: number; offset: number; size: number }>;
  layout?: WeightLayout;
  originalShape?: number[];
  group?: string;
}

/** Result of RDRR write operation */
export interface WriteResultSchema {
  manifestPath: string;
  shardCount: number;
  totalSize: number;
  tensorCount: number;
}

/** Platform-specific I/O adapter */
export interface ConversionIOSchema {
  /** Read tensor data from source */
  readTensorData(tensor: TensorInfoSchema): Promise<ArrayBuffer>;
  /** Write shard data, returns hash */
  writeShard(index: number, data: Uint8Array): Promise<string>;
  /** Write manifest JSON */
  writeManifest(manifest: unknown): Promise<void>;
  /** Optional: compute hash */
  computeHash?(data: Uint8Array): Promise<string>;
}
