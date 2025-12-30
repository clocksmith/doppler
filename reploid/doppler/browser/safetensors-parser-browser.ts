/**
 * safetensors-parser-browser.ts - Browser Safetensors Parser
 *
 * Parses Hugging Face safetensors files in the browser using File API.
 * Supports both single files and sharded models (multiple files + index).
 *
 * @module browser/safetensors-parser-browser
 */

import {
  DTYPE_SIZE,
  parseSafetensorsHeader,
  parseSafetensorsIndexJsonText,
  groupTensorsByLayer as groupTensorsByLayerCore,
  calculateTotalSize as calculateTotalSizeCore,
  type SafetensorsDtype,
  type SafetensorsTensor as CoreSafetensorsTensor,
  type SafetensorsIndexJson,
} from '../formats/safetensors.js';
import {
  parseConfigJsonText,
  parseTokenizerJsonText,
  parseTokenizerConfigJsonText,
} from '../formats/tokenizer.js';

export { DTYPE_SIZE, DTYPE_MAP } from '../formats/safetensors.js';

// ============================================================================
// Types and Interfaces
// ============================================================================

export type { SafetensorsDtype, SafetensorsIndexJson };

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

// ============================================================================
// Public API
// ============================================================================

/**
 * Parse safetensors header from File object
 */
export async function parseSafetensorsFile(file: File): Promise<ParsedSafetensorsFile> {
  const headerSizeBuffer = await file.slice(0, 8).arrayBuffer();
  const headerSizeView = new DataView(headerSizeBuffer);
  const headerSizeLow = headerSizeView.getUint32(0, true);
  const headerSizeHigh = headerSizeView.getUint32(4, true);
  const headerSize = headerSizeHigh * 0x100000000 + headerSizeLow;
  if (headerSize > 100 * 1024 * 1024) {
    throw new Error(`Header too large: ${headerSize} bytes`);
  }

  const headerBuffer = await file.slice(8, 8 + headerSize).arrayBuffer();
  const combined = new Uint8Array(8 + headerSize);
  combined.set(new Uint8Array(headerSizeBuffer), 0);
  combined.set(new Uint8Array(headerBuffer), 8);
  const parsedHeader = parseSafetensorsHeader(combined.buffer);

  const tensors: SafetensorsTensor[] = parsedHeader.tensors.map((tensor) => ({
    ...tensor,
    elemSize: tensor.elemSize ?? DTYPE_SIZE[tensor.dtype as SafetensorsDtype] ?? 4,
    dtypeOriginal: tensor.dtypeOriginal ?? tensor.dtype,
    file,
  }));

  return {
    headerSize: parsedHeader.headerSize,
    dataOffset: parsedHeader.dataOffset,
    metadata: parsedHeader.metadata,
    tensors,
    file,
    fileSize: file.size,
    fileName: file.name,
  };
}

/**
 * Parse sharded safetensors model from multiple files
 */
export async function parseSafetensorsSharded(
  files: File[],
  indexJson: SafetensorsIndexJson | null = null
): Promise<ParsedSafetensorsSharded> {
  const fileMap = new Map<string, File>();
  for (const file of files) {
    fileMap.set(file.name, file);
  }

  // If we have an index, use it to determine tensor locations
  let metadata: Record<string, unknown> = {};

  if (indexJson) {
    metadata = indexJson.metadata || {};
  }

  // Parse each safetensors file
  const shards: ShardInfo[] = [];
  const allTensors: SafetensorsTensor[] = [];

  for (const file of files) {
    if (!file.name.endsWith('.safetensors')) continue;

    const parsed = await parseSafetensorsFile(file);
    shards.push({
      file: file.name,
      size: file.size,
      tensorCount: parsed.tensors.length,
    });

    // Add shard info to tensors
    for (const tensor of parsed.tensors) {
      tensor.shardFile = file.name;
      allTensors.push(tensor);
    }
  }

  return {
    metadata,
    shards,
    tensors: allTensors,
    fileMap,
  };
}

/**
 * Read tensor data from File
 */
export async function readTensorData(tensor: SafetensorsTensor): Promise<ArrayBuffer> {
  const file = tensor.file;
  if (!file) {
    throw new Error('No file reference for tensor');
  }

  const blob = file.slice(tensor.offset, tensor.offset + tensor.size);
  return blob.arrayBuffer();
}

/**
 * Stream tensor data for large files
 */
export async function* streamTensorData(
  tensor: SafetensorsTensor,
  chunkSize = 64 * 1024 * 1024
): AsyncGenerator<Uint8Array> {
  const file = tensor.file;
  if (!file) {
    throw new Error('No file reference for tensor');
  }

  let offset = tensor.offset;
  const endOffset = tensor.offset + tensor.size;

  while (offset < endOffset) {
    const end = Math.min(offset + chunkSize, endOffset);
    const blob = file.slice(offset, end);
    const buffer = await blob.arrayBuffer();
    yield new Uint8Array(buffer);
    offset = end;
  }
}

/**
 * Parse config.json from File
 */
export async function parseConfigJson(configFile: File): Promise<Record<string, unknown>> {
  const text = await configFile.text();
  return parseConfigJsonText(text);
}

export async function parseTokenizerConfigJson(tokenizerConfigFile: File): Promise<Record<string, unknown>> {
  const text = await tokenizerConfigFile.text();
  return parseTokenizerConfigJsonText(text);
}

/**
 * Parse tokenizer.json from File
 */
export async function parseTokenizerJson(tokenizerFile: File): Promise<Record<string, unknown>> {
  const text = await tokenizerFile.text();
  return parseTokenizerJsonText(text);
}

/**
 * Parse model.safetensors.index.json from File
 */
export async function parseIndexJson(indexFile: File): Promise<SafetensorsIndexJson> {
  const text = await indexFile.text();
  return parseSafetensorsIndexJsonText(text);
}

/**
 * Detect model format from selected files
 */
export function detectModelFormat(files: File[]): ModelFormatInfo {
  // Check for index file (sharded model)
  const indexFile = files.find(f => f.name === 'model.safetensors.index.json');
  if (indexFile) {
    return {
      type: 'sharded',
      indexFile,
      safetensorsFiles: files.filter(f => f.name.endsWith('.safetensors')),
    };
  }

  // Check for single safetensors file
  const safetensorsFiles = files.filter(f => f.name.endsWith('.safetensors'));
  if (safetensorsFiles.length === 1) {
    return {
      type: 'single',
      safetensorsFile: safetensorsFiles[0],
    };
  }

  if (safetensorsFiles.length > 1) {
    return {
      type: 'sharded-no-index',
      safetensorsFiles,
    };
  }

  // Check for GGUF
  const ggufFile = files.find(f => f.name.endsWith('.gguf'));
  if (ggufFile) {
    return {
      type: 'gguf',
      ggufFile,
    };
  }

  return { type: 'unknown', files };
}

/**
 * Get auxiliary files from selection
 */
export function getAuxiliaryFiles(files: File[]): AuxiliaryFiles {
  return {
    config: files.find(f => f.name === 'config.json'),
    tokenizerConfig: files.find(f => f.name === 'tokenizer_config.json'),
    tokenizer: files.find(f => f.name === 'tokenizer.json'),
    tokenizerModel: files.find(f => f.name === 'tokenizer.model'),
    specialTokensMap: files.find(f => f.name === 'special_tokens_map.json'),
    generationConfig: files.find(f => f.name === 'generation_config.json'),
  };
}

/**
 * Calculate total model size
 */
export function calculateTotalSize(parsed: { tensors: SafetensorsTensor[] }): number {
  return calculateTotalSizeCore(parsed);
}

/**
 * Group tensors by layer
 */
export function groupTensorsByLayer(
  parsed: { tensors: SafetensorsTensor[] }
): Map<number, SafetensorsTensor[]> {
  // Cast to browser type - core tensors are a subset of browser tensors
  return groupTensorsByLayerCore(parsed) as Map<number, SafetensorsTensor[]>;
}

// ============================================================================
// Type Aliases for API Compatibility
// ============================================================================

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
