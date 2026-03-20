/**
 * Converter Config Schema Definitions
 *
 * Converter configuration for GGUF/SafeTensors -> RDRR output.
 *
 * @module config/schema/converter
 */

import type {
  HashAlgorithm,
  QuantizationValue,
  RuntimeOptimizationsSchema,
  ConversionInfoSchema,
} from './manifest.schema.js';
import type {
  ExecutionV1GraphSchema,
  ExecutionV1SessionDefaultsSchema,
} from './execution-v1.schema.js';

export type ComputePrecision = 'f16' | 'f32' | 'auto' | null;
/** Q4K layout: 'row' = fused kernel (fast), 'col' = dequant fallback */
export type Q4KLayout = 'row' | 'col' | null;
export type ConverterWorkerCountPolicy = 'cap' | 'error';

export interface ConverterQuantizationConfigSchema {
  weights: QuantizationValue | null;
  embeddings: QuantizationValue | null;
  lmHead: QuantizationValue | null;
  vision: QuantizationValue | null;
  audio: QuantizationValue | null;
  projector: QuantizationValue | null;
  q4kLayout: Q4KLayout;
  computePrecision: ComputePrecision;
}

export interface ConverterShardingConfigSchema {
  shardSizeBytes: number;
}

export interface ConverterStreamingConfigSchema {
  chunkSizeBytes: number;
}

export interface ConverterHttpConfigSchema {
  allowDownloadFallback: boolean;
  maxDownloadBytes: number | null;
}

export interface ConverterWeightLayoutConfigSchema {
  transposeWeights: boolean;
  fuseGateUp: boolean;
}

export interface ConverterManifestConfigSchema {
  hashAlgorithm: HashAlgorithm;
  optimizations: RuntimeOptimizationsSchema | null;
  conversion: ConversionInfoSchema | null;
}

export interface ConverterInferenceConfigSchema {
  defaultKernelPath: string | null;
  sessionDefaults: ExecutionV1SessionDefaultsSchema | null;
  execution: ExecutionV1GraphSchema | null;
}

export interface ConverterOutputConfigSchema {
  modelBaseId: string | null;
  dir: string | null;
  baseDir: string | null;
  textOnly: boolean;
  fast: boolean;
}

export interface GGUFParserDefaultsSchema {
  contextLength: number;
  attentionLayerNormEpsilon: number;
  attentionLayerNormRMSEpsilon: number;
  ropeFreqBase: number;
}

/** Node convert command execution defaults (worker scheduling policy) */
export interface ConverterExecutionConfigSchema {
  workers: number;
  workerCountPolicy: ConverterWorkerCountPolicy;
  rowChunkRows: number | null;
  rowChunkMinTensorBytes: number;
  maxInFlightJobs: number | null;
  useGpuCast: boolean;
  gpuCastMinTensorBytes: number;
}

export interface ConverterConfigSchema {
  quantization: ConverterQuantizationConfigSchema;
  sharding: ConverterShardingConfigSchema;
  streaming: ConverterStreamingConfigSchema;
  http: ConverterHttpConfigSchema;
  weightLayout: ConverterWeightLayoutConfigSchema;
  manifest: ConverterManifestConfigSchema;
  inference: ConverterInferenceConfigSchema;
  output: ConverterOutputConfigSchema;
}

export declare const DEFAULT_CONVERTER_QUANTIZATION_CONFIG: ConverterQuantizationConfigSchema;
export declare const DEFAULT_CONVERTER_SHARDING_CONFIG: ConverterShardingConfigSchema;
export declare const DEFAULT_CONVERTER_STREAMING_CONFIG: ConverterStreamingConfigSchema;
export declare const DEFAULT_CONVERTER_HTTP_CONFIG: ConverterHttpConfigSchema;
export declare const DEFAULT_CONVERTER_WEIGHT_LAYOUT_CONFIG: ConverterWeightLayoutConfigSchema;
export declare const DEFAULT_CONVERTER_MANIFEST_CONFIG: ConverterManifestConfigSchema;
export declare const DEFAULT_CONVERTER_INFERENCE_CONFIG: ConverterInferenceConfigSchema;
export declare const DEFAULT_CONVERTER_OUTPUT_CONFIG: ConverterOutputConfigSchema;
export declare const DEFAULT_GGUF_PARSER_DEFAULTS: GGUFParserDefaultsSchema;
export declare const DEFAULT_CONVERTER_EXECUTION_CONFIG: ConverterExecutionConfigSchema;
export declare const DEFAULT_CONVERTER_CONFIG: ConverterConfigSchema;

export declare function createConverterConfig(
  overrides?: Partial<ConverterConfigSchema>
): ConverterConfigSchema;
