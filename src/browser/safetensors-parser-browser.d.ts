/**
 * safetensors-parser-browser.ts - Browser Safetensors Parser
 *
 * Parses Hugging Face safetensors files in the browser using File API.
 * Supports both single files and sharded models (multiple files + index).
 *
 * @module browser/safetensors-parser-browser
 */

import type { SafetensorsTensor as CoreSafetensorsTensor, SafetensorsIndexJson } from '../formats/safetensors.js';

export { DTYPE_SIZE, DTYPE_MAP } from '../formats/safetensors.js';

export type { SafetensorsDtype, SafetensorsIndexJson } from '../formats/safetensors.js';

/**
 * Tensor information from safetensors file
 */
export type SafetensorsTensor = CoreSafetensorsTensor & {
  file?: File;
  elemSize: number;
  dtypeOriginal: string;
};

/**
 * Parsed safetensors file result
 */
export interface ParsedSafetensorsFile {
  headerSize: number;
  dataOffset: number;
  metadata: Record<string, unknown>;
  tensors: SafetensorsTensor[];
  file: File;
  fileSize: number;
  fileName: string;
  config?: ModelConfig;
}

/**
 * Shard information
 */
export interface ShardInfo {
  file: string;
  size: number;
  tensorCount: number;
}

/**
 * Parsed sharded safetensors model
 */
export interface ParsedSafetensorsSharded {
  metadata: Record<string, unknown>;
  shards: ShardInfo[];
  tensors: SafetensorsTensor[];
  fileMap: Map<string, File>;
  config?: ModelConfig;
}

/**
 * Model format detection result
 */
export interface ModelFormatInfo {
  type: 'single' | 'sharded' | 'sharded-no-index' | 'gguf' | 'unknown';
  indexFile?: File;
  safetensorsFile?: File;
  safetensorsFiles?: File[];
  ggufFile?: File;
  files?: File[];
}

/**
 * Auxiliary files from model directory
 */
export interface AuxiliaryFiles {
  config?: File;
  tokenizerConfig?: File;
  tokenizer?: File;
  tokenizerModel?: File;
  specialTokensMap?: File;
  generationConfig?: File;
}

/**
 * Parse safetensors header from File object
 */
export declare function parseSafetensorsFile(file: File): Promise<ParsedSafetensorsFile>;

/**
 * Parse sharded safetensors model from multiple files
 */
export declare function parseSafetensorsSharded(
  files: File[],
  indexJson?: SafetensorsIndexJson | null
): Promise<ParsedSafetensorsSharded>;

/**
 * Read tensor data from File
 */
export declare function readTensorData(tensor: SafetensorsTensor): Promise<ArrayBuffer>;

/**
 * Stream tensor data for large files
 */
export declare function streamTensorData(
  tensor: SafetensorsTensor,
  chunkSize?: number
): AsyncGenerator<Uint8Array>;

/**
 * Parse config.json from File
 */
export declare function parseConfigJson(configFile: File): Promise<Record<string, unknown>>;

export declare function parseTokenizerConfigJson(tokenizerConfigFile: File): Promise<Record<string, unknown>>;

/**
 * Parse tokenizer.json from File
 */
export declare function parseTokenizerJson(tokenizerFile: File): Promise<Record<string, unknown>>;

/**
 * Parse model.safetensors.index.json from File
 */
export declare function parseIndexJson(indexFile: File): Promise<SafetensorsIndexJson>;

/**
 * Detect model format from selected files
 */
export declare function detectModelFormat(files: File[]): ModelFormatInfo;

/**
 * Get auxiliary files from selection
 */
export declare function getAuxiliaryFiles(files: File[]): AuxiliaryFiles;

/**
 * Calculate total model size
 */
export declare function calculateTotalSize(parsed: { tensors: SafetensorsTensor[] }): number;

/**
 * Group tensors by layer
 */
export declare function groupTensorsByLayer(
  parsed: { tensors: SafetensorsTensor[] }
): Map<number, SafetensorsTensor[]>;

/**
 * @deprecated Use ParsedSafetensorsFile instead
 */
export type SafetensorsParseResult = ParsedSafetensorsFile;

/**
 * @deprecated Use ModelFormatInfo instead
 */
export type ModelFormat = ModelFormatInfo;

/**
 * @deprecated Use SafetensorsTensor instead
 */
export type TensorInfo = SafetensorsTensor;

/**
 * Model configuration type (extracted from config.json)
 */
export interface ModelConfig {
  architectures?: string[];
  model_type?: string;
  hidden_size?: number;
  intermediate_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  vocab_size?: number;
  max_position_embeddings?: number;
  rms_norm_eps?: number;
  rope_theta?: number;
  rope_scaling?: {
    type?: string;
    factor?: number;
  };
  _name_or_path?: string;
  n_layer?: number;
  n_embd?: number;
  n_inner?: number;
  n_head?: number;
  n_positions?: number;
  head_dim?: number;
  [key: string]: unknown;
}
