/**
 * safetensors-parser-browser.ts - Browser Safetensors Parser
 *
 * Parses Hugging Face safetensors files in the browser using File API.
 * Supports both single files and sharded models (multiple files + index).
 *
 * @module browser/safetensors-parser-browser
 */

// ============================================================================
// Types and Interfaces
// ============================================================================

/**
 * Data types supported by safetensors
 */
export type SafetensorsDtype =
  | 'F64'
  | 'F32'
  | 'F16'
  | 'BF16'
  | 'I64'
  | 'I32'
  | 'I16'
  | 'I8'
  | 'U8'
  | 'BOOL';

/**
 * Tensor information from safetensors file
 */
export interface SafetensorsTensor {
  name: string;
  shape: number[];
  dtype: string;
  dtypeOriginal: string;
  offset: number;
  size: number;
  elemSize: number;
  file?: File;
  shardFile?: string;
}

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
 * Index JSON structure
 */
export interface SafetensorsIndexJson {
  weight_map: Record<string, string>;
  metadata?: Record<string, unknown>;
}

// ============================================================================
// Constants
// ============================================================================

/**
 * Safetensors dtype to byte size
 */
export const DTYPE_SIZE: Record<SafetensorsDtype, number> = {
  F64: 8,
  F32: 4,
  F16: 2,
  BF16: 2,
  I64: 8,
  I32: 4,
  I16: 2,
  I8: 1,
  U8: 1,
  BOOL: 1,
};

/**
 * Map safetensors dtype to internal names
 */
export const DTYPE_MAP: Record<string, string> = {
  F64: 'F64',
  F32: 'F32',
  F16: 'F16',
  BF16: 'BF16',
  I64: 'I64',
  I32: 'I32',
  I16: 'I16',
  I8: 'I8',
  U8: 'U8',
  BOOL: 'BOOL',
};

// ============================================================================
// Public API
// ============================================================================

/**
 * Parse safetensors header from File object
 */
export async function parseSafetensorsFile(file: File): Promise<ParsedSafetensorsFile> {
  // Read first 8 bytes to get header size
  const headerSizeBlob = file.slice(0, 8);
  const headerSizeBuffer = await headerSizeBlob.arrayBuffer();
  const headerSizeView = new DataView(headerSizeBuffer);

  const headerSizeLow = headerSizeView.getUint32(0, true);
  const headerSizeHigh = headerSizeView.getUint32(4, true);
  const headerSize = headerSizeHigh * 0x100000000 + headerSizeLow;

  if (headerSize > 100 * 1024 * 1024) {
    throw new Error(`Header too large: ${headerSize} bytes`);
  }

  // Read JSON header
  const headerBlob = file.slice(8, 8 + headerSize);
  const headerBuffer = await headerBlob.arrayBuffer();
  const headerJson = new TextDecoder().decode(headerBuffer);
  const header = JSON.parse(headerJson) as Record<string, unknown>;

  // Data starts after header
  const dataOffset = 8 + headerSize;

  // Extract metadata (special __metadata__ key)
  const metadata = (header.__metadata__ || {}) as Record<string, unknown>;
  delete header.__metadata__;

  // Parse tensor info
  const tensors: SafetensorsTensor[] = [];
  for (const [name, info] of Object.entries(header)) {
    const tensorInfo = info as {
      dtype: string;
      shape: number[];
      data_offsets: [number, number];
    };
    const { dtype, shape, data_offsets } = tensorInfo;
    const [startOffset, endOffset] = data_offsets;

    tensors.push({
      name,
      shape,
      dtype: DTYPE_MAP[dtype] || dtype,
      dtypeOriginal: dtype,
      offset: dataOffset + startOffset,
      size: endOffset - startOffset,
      elemSize: DTYPE_SIZE[dtype as SafetensorsDtype] || 4,
      file,
    });
  }

  // Sort by offset for sequential reading
  tensors.sort((a, b) => a.offset - b.offset);

  return {
    headerSize,
    dataOffset,
    metadata,
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
  return JSON.parse(text);
}

/**
 * Parse tokenizer.json from File
 */
export async function parseTokenizerJson(tokenizerFile: File): Promise<Record<string, unknown>> {
  const text = await tokenizerFile.text();
  return JSON.parse(text);
}

/**
 * Parse model.safetensors.index.json from File
 */
export async function parseIndexJson(indexFile: File): Promise<SafetensorsIndexJson> {
  const text = await indexFile.text();
  return JSON.parse(text);
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
  return parsed.tensors.reduce((sum, t) => sum + t.size, 0);
}

/**
 * Group tensors by layer
 */
export function groupTensorsByLayer(
  parsed: { tensors: SafetensorsTensor[] }
): Map<number, SafetensorsTensor[]> {
  const layers = new Map<number, SafetensorsTensor[]>();

  for (const tensor of parsed.tensors) {
    const match = tensor.name.match(/layers?\.(\d+)\./);
    if (match) {
      const layerIdx = parseInt(match[1], 10);
      if (!layers.has(layerIdx)) {
        layers.set(layerIdx, []);
      }
      layers.get(layerIdx)!.push(tensor);
    }
  }

  return layers;
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
