/**
 * Type definitions for the Node.js Model Converter.
 *
 * Contains CLI options interfaces, conversion config types, and model info structures.
 *
 * @module converter/node-converter/types
 */

import type { QuantizationInfoSchema } from '../../config/index.js';

/**
 * CLI options parsed from command-line arguments.
 */
export interface ConvertOptions {
  /** Input path (HuggingFace directory or GGUF file) */
  input: string;
  /** Output directory for RDRR model */
  output: string;
  /** Weight quantization (e.g., 'q4k', 'f16') */
  weightQuant: string | null;
  /** Embedding quantization */
  embedQuant: string | null;
  /** LM head quantization */
  headQuant: string | null;
  /** Vision encoder quantization (multimodal models) */
  visionQuant: string | null;
  /** Audio encoder quantization (speech models) */
  audioQuant: string | null;
  /** Cross-modal projector quantization */
  projectorQuant: string | null;
  /** Runtime compute precision hint */
  computePrecision: 'f16' | 'f32' | 'auto' | null;
  /** Shard size in MB */
  shardSize: number;
  /** Base model ID (variant tag auto-appended) */
  modelId: string | null;
  /** Extract only text model from multimodal */
  textOnly: boolean;
  /** Pre-load all shards into memory (faster, more RAM) */
  fast: boolean;
  /** Show detailed progress */
  verbose: boolean;
  /** Create tiny test fixture */
  test: boolean;
  /** Show help */
  help: boolean;
}

/**
 * Information about a single tensor.
 */
export interface TensorInfo {
  /** Tensor name */
  name: string;
  /** Tensor shape dimensions */
  shape: number[];
  /** Data type (e.g., 'F16', 'Q4_K_M') */
  dtype: string;
  /** Size in bytes */
  size: number;
}

/**
 * Model information extracted from source format.
 */
export interface ModelInfo {
  /** Model name/identifier */
  modelName?: string;
  /** Model architecture (e.g., 'gemma2', 'llama') */
  architecture?: string;
  /** Primary quantization type */
  quantization?: string;
  /** Detailed quantization info per component */
  quantizationInfo?: QuantizationInfoSchema;
  /** Model configuration (HuggingFace-style) */
  config?: Record<string, unknown>;
  /** Tokenizer configuration */
  tokenizer?: Record<string, unknown>;
  /** Tokenizer config (separate from tokenizer.json) */
  tokenizerConfig?: Record<string, unknown>;
  /** Raw tokenizer.json contents */
  tokenizerJson?: Record<string, unknown>;
  /** List of tensors in the model */
  tensors: TensorInfo[];
}

/**
 * Detected input format.
 */
export type InputFormat = 'gguf' | 'safetensors' | 'unknown';

/**
 * Function type for reading tensor data.
 */
export type TensorDataGetter = (info: TensorInfo) => Promise<ArrayBuffer>;
