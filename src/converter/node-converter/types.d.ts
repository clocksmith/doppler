/**
 * Type definitions for the Node.js Model Converter.
 *
 * @module converter/node-converter/types
 */

import type { ConverterConfigSchema, QuantizationInfoSchema } from '../../config/index.js';

export interface ConvertOptions {
  input: string;
  output: string;
  weightQuant: string | null;
  embedQuant: string | null;
  headQuant: string | null;
  visionQuant: string | null;
  audioQuant: string | null;
  projectorQuant: string | null;
  computePrecision: 'f16' | 'f32' | 'auto' | null;
  shardSize: number;
  shardSizeBytes?: number;
  modelId: string | null;
  textOnly: boolean;
  fast: boolean;
  verbose: boolean;
  test: boolean;
  help: boolean;
  converterConfig?: Partial<ConverterConfigSchema>;
}

export interface TensorInfo {
  name: string;
  shape: number[];
  dtype: string;
  size: number;
}

export interface ModelInfo {
  modelName?: string;
  architecture?: string;
  quantization?: string;
  quantizationInfo?: QuantizationInfoSchema;
  config?: Record<string, unknown>;
  tokenizer?: Record<string, unknown>;
  tokenizerConfig?: Record<string, unknown>;
  tokenizerJson?: Record<string, unknown>;
  tensors: TensorInfo[];
}

export type InputFormat = 'gguf' | 'safetensors' | 'unknown';

export type TensorDataGetter = (info: TensorInfo) => Promise<ArrayBuffer>;
