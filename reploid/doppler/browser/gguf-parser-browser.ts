/**
 * gguf-parser-browser.ts - Browser-safe GGUF Parser
 *
 * Parses GGUF files (llama.cpp format) in the browser.
 * Based on tools/gguf-parser.js but without Node.js dependencies.
 *
 * @module browser/gguf-parser-browser
 */

// ============================================================================
// Types and Interfaces
// ============================================================================

/**
 * GGUF value types for metadata
 */
export enum GGUFValueType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

/**
 * GGML tensor types (quantization formats)
 */
export const GGMLType = {
  F32: 0,
  F16: 1,
  Q4_0: 2,
  Q4_1: 3,
  Q5_0: 6,
  Q5_1: 7,
  Q8_0: 8,
  Q8_1: 9,
  Q2_K: 10,
  Q3_K: 11,
  Q4_K: 12,
  Q5_K: 13,
  Q6_K: 14,
  Q8_K: 15,
  IQ2_XXS: 16,
  IQ2_XS: 17,
  IQ3_XXS: 18,
  IQ1_S: 19,
  IQ4_NL: 20,
  IQ3_S: 21,
  IQ2_S: 22,
  IQ4_XS: 23,
  I8: 24,
  I16: 25,
  I32: 26,
  I64: 27,
  F64: 28,
  BF16: 29,
} as const;

export type GGMLTypeValue = (typeof GGMLType)[keyof typeof GGMLType];

/**
 * Tensor information from GGUF file
 */
export interface GGUFTensor {
  name: string;
  shape: number[];
  dtype: string;
  dtypeId: number;
  offset: number;
  size: number;
}

/**
 * Tokenizer configuration from GGUF
 */
export interface GGUFTokenizer {
  model?: string;
  bosTokenId?: number;
  eosTokenId?: number;
  padTokenId?: number;
  addBosToken?: boolean;
  addEosToken?: boolean;
}

/**
 * Model configuration extracted from GGUF metadata
 */
export interface GGUFModelConfig {
  architecture: string;
  vocabSize?: number;
  contextLength?: number;
  embeddingLength?: number;
  blockCount?: number;
  feedForwardLength?: number;
  attentionHeadCount?: number;
  attentionHeadCountKV?: number;
  attentionLayerNormEpsilon?: number;
  attentionLayerNormRMSEpsilon?: number;
  ropeFreqBase?: number;
  ropeScalingType?: string;
  ropeScalingFactor?: number;
  expertCount?: number;
  expertUsedCount?: number;
  tokenizer?: GGUFTokenizer;
  [key: string]: unknown;
}

/**
 * Parsed GGUF result
 */
export interface ParsedGGUF {
  version: number;
  architecture: string;
  modelName: string;
  metadata: Record<string, unknown>;
  config: GGUFModelConfig;
  tensors: GGUFTensor[];
  quantization: string;
  tensorDataOffset: number;
  totalTensorSize: number;
  headerSize: number;
}

// ============================================================================
// Constants
// ============================================================================

// GGUF Magic number: "GGUF" in little-endian
const GGUF_MAGIC = 0x46554747; // 'GGUF'

// GGUF versions supported
const GGUF_VERSION_MIN = 2;
const GGUF_VERSION_MAX = 3;

// Reverse lookup for type names
export const GGMLTypeName: Record<number, string> = Object.fromEntries(
  Object.entries(GGMLType).map(([k, v]) => [v, k])
);

// Block sizes for quantized types (elements per block)
export const GGML_BLOCK_SIZE: Record<number, number> = {
  [GGMLType.Q4_0]: 32,
  [GGMLType.Q4_1]: 32,
  [GGMLType.Q5_0]: 32,
  [GGMLType.Q5_1]: 32,
  [GGMLType.Q8_0]: 32,
  [GGMLType.Q8_1]: 32,
  [GGMLType.Q2_K]: 256,
  [GGMLType.Q3_K]: 256,
  [GGMLType.Q4_K]: 256,
  [GGMLType.Q5_K]: 256,
  [GGMLType.Q6_K]: 256,
  [GGMLType.Q8_K]: 256,
};

// Bytes per block for quantized types
export const GGML_TYPE_SIZE: Record<number, number> = {
  [GGMLType.F32]: 4,
  [GGMLType.F16]: 2,
  [GGMLType.Q4_0]: 18,
  [GGMLType.Q4_1]: 20,
  [GGMLType.Q5_0]: 22,
  [GGMLType.Q5_1]: 24,
  [GGMLType.Q8_0]: 34,
  [GGMLType.Q8_1]: 36,
  [GGMLType.Q2_K]: 84,
  [GGMLType.Q3_K]: 110,
  [GGMLType.Q4_K]: 144,
  [GGMLType.Q5_K]: 176,
  [GGMLType.Q6_K]: 210,
  [GGMLType.Q8_K]: 292,
  [GGMLType.BF16]: 2,
  [GGMLType.I8]: 1,
  [GGMLType.I16]: 2,
  [GGMLType.I32]: 4,
  [GGMLType.I64]: 8,
  [GGMLType.F64]: 8,
};

// ============================================================================
// GGUFReader Class
// ============================================================================

/**
 * Reader class for parsing binary GGUF data
 */
class GGUFReader {
  private buffer: ArrayBuffer;
  private view: DataView;
  offset: number;

  constructor(buffer: ArrayBuffer) {
    this.buffer = buffer;
    this.view = new DataView(buffer);
    this.offset = 0;
  }

  readUint8(): number {
    const val = this.view.getUint8(this.offset);
    this.offset += 1;
    return val;
  }

  readInt8(): number {
    const val = this.view.getInt8(this.offset);
    this.offset += 1;
    return val;
  }

  readUint16(): number {
    const val = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return val;
  }

  readInt16(): number {
    const val = this.view.getInt16(this.offset, true);
    this.offset += 2;
    return val;
  }

  readUint32(): number {
    const val = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return val;
  }

  readInt32(): number {
    const val = this.view.getInt32(this.offset, true);
    this.offset += 4;
    return val;
  }

  readUint64(): number {
    const low = this.view.getUint32(this.offset, true);
    const high = this.view.getUint32(this.offset + 4, true);
    this.offset += 8;
    return high * 0x100000000 + low;
  }

  readInt64(): number {
    const low = this.view.getUint32(this.offset, true);
    const high = this.view.getInt32(this.offset + 4, true);
    this.offset += 8;
    return high * 0x100000000 + low;
  }

  readFloat32(): number {
    const val = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return val;
  }

  readFloat64(): number {
    const val = this.view.getFloat64(this.offset, true);
    this.offset += 8;
    return val;
  }

  readBool(): boolean {
    return this.readUint8() !== 0;
  }

  readString(): string {
    const length = this.readUint64();
    if (length > 1024 * 1024) {
      throw new Error(`String too long: ${length}`);
    }
    const bytes = new Uint8Array(this.buffer, this.offset, length);
    this.offset += length;
    return new TextDecoder().decode(bytes);
  }

  readValue(type: GGUFValueType): unknown {
    switch (type) {
      case GGUFValueType.UINT8:
        return this.readUint8();
      case GGUFValueType.INT8:
        return this.readInt8();
      case GGUFValueType.UINT16:
        return this.readUint16();
      case GGUFValueType.INT16:
        return this.readInt16();
      case GGUFValueType.UINT32:
        return this.readUint32();
      case GGUFValueType.INT32:
        return this.readInt32();
      case GGUFValueType.UINT64:
        return this.readUint64();
      case GGUFValueType.INT64:
        return this.readInt64();
      case GGUFValueType.FLOAT32:
        return this.readFloat32();
      case GGUFValueType.FLOAT64:
        return this.readFloat64();
      case GGUFValueType.BOOL:
        return this.readBool();
      case GGUFValueType.STRING:
        return this.readString();
      case GGUFValueType.ARRAY:
        return this.readArray();
      default:
        throw new Error(`Unknown value type: ${type}`);
    }
  }

  readArray(): unknown[] {
    const elementType = this.readUint32();
    const length = this.readUint64();
    if (length > 10000000) {
      throw new Error(`Array too long: ${length}`);
    }
    const arr: unknown[] = [];
    for (let i = 0; i < length; i++) {
      arr.push(this.readValue(elementType));
    }
    return arr;
  }

  align(boundary: number): void {
    const remainder = this.offset % boundary;
    if (remainder !== 0) {
      this.offset += boundary - remainder;
    }
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Calculate tensor size in bytes
 */
function calculateTensorSize(shape: number[], type: number): number {
  const numElements = shape.reduce((a, b) => a * b, 1);

  if (type === GGMLType.F32) return numElements * 4;
  if (type === GGMLType.F16 || type === GGMLType.BF16) return numElements * 2;
  if (type === GGMLType.I8) return numElements;
  if (type === GGMLType.I16) return numElements * 2;
  if (type === GGMLType.I32) return numElements * 4;
  if (type === GGMLType.I64) return numElements * 8;
  if (type === GGMLType.F64) return numElements * 8;

  // Quantized types
  const blockSize = GGML_BLOCK_SIZE[type];
  const typeSize = GGML_TYPE_SIZE[type];
  if (blockSize && typeSize) {
    const numBlocks = Math.ceil(numElements / blockSize);
    return numBlocks * typeSize;
  }

  throw new Error(`Unknown tensor type: ${type}`);
}

/**
 * Extract model configuration from metadata
 */
function extractModelConfig(
  metadata: Record<string, unknown>,
  architecture: string
): GGUFModelConfig {
  const prefix = `${architecture}.`;

  const config: GGUFModelConfig = {
    architecture,
    vocabSize: (metadata[`${prefix}vocab_size`] || metadata['tokenizer.ggml.vocab_size']) as number | undefined,
    contextLength: (metadata[`${prefix}context_length`] as number) || 2048,
    embeddingLength: metadata[`${prefix}embedding_length`] as number | undefined,
    blockCount: metadata[`${prefix}block_count`] as number | undefined,
    feedForwardLength: metadata[`${prefix}feed_forward_length`] as number | undefined,
    attentionHeadCount: metadata[`${prefix}attention.head_count`] as number | undefined,
    attentionHeadCountKV: metadata[`${prefix}attention.head_count_kv`] as number | undefined,
    attentionLayerNormEpsilon: (metadata[`${prefix}attention.layer_norm_epsilon`] as number) || 1e-5,
    attentionLayerNormRMSEpsilon: (metadata[`${prefix}attention.layer_norm_rms_epsilon`] as number) || 1e-5,
    ropeFreqBase: (metadata[`${prefix}rope.freq_base`] as number) || 10000,
    ropeScalingType: metadata[`${prefix}rope.scaling.type`] as string | undefined,
    ropeScalingFactor: metadata[`${prefix}rope.scaling.factor`] as number | undefined,
  };

  // MoE specific
  if (metadata[`${prefix}expert_count`]) {
    config.expertCount = metadata[`${prefix}expert_count`] as number;
    config.expertUsedCount = metadata[`${prefix}expert_used_count`] as number;
  }

  // Tokenizer info
  config.tokenizer = {
    model: metadata['tokenizer.ggml.model'] as string | undefined,
    bosTokenId: metadata['tokenizer.ggml.bos_token_id'] as number | undefined,
    eosTokenId: metadata['tokenizer.ggml.eos_token_id'] as number | undefined,
    padTokenId: metadata['tokenizer.ggml.padding_token_id'] as number | undefined,
    addBosToken: metadata['tokenizer.ggml.add_bos_token'] as boolean | undefined,
    addEosToken: metadata['tokenizer.ggml.add_eos_token'] as boolean | undefined,
  };

  return config;
}

/**
 * Detect primary quantization type from tensors
 */
function detectQuantization(tensors: GGUFTensor[]): string {
  const typeCounts: Record<string, number> = {};

  for (const tensor of tensors) {
    // Skip embedding and output layers for quant detection
    if (tensor.name.includes('embed') || tensor.name.includes('output')) {
      continue;
    }
    const dtype = tensor.dtype;
    typeCounts[dtype] = (typeCounts[dtype] || 0) + tensor.size;
  }

  let dominantType = 'F16';
  let maxSize = 0;
  for (const [dtype, size] of Object.entries(typeCounts)) {
    if (size > maxSize) {
      maxSize = size;
      dominantType = dtype;
    }
  }

  return dominantType;
}

// ============================================================================
// Public API
// ============================================================================

/**
 * Parse GGUF buffer (browser-safe)
 */
export function parseGGUF(buffer: ArrayBuffer): ParsedGGUF {
  const reader = new GGUFReader(buffer);

  // Read header
  const magic = reader.readUint32();
  if (magic !== GGUF_MAGIC) {
    throw new Error(`Invalid GGUF magic: 0x${magic.toString(16)} (expected 0x${GGUF_MAGIC.toString(16)})`);
  }

  const version = reader.readUint32();
  if (version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX) {
    throw new Error(`Unsupported GGUF version: ${version}`);
  }

  const tensorCount = reader.readUint64();
  const metadataKVCount = reader.readUint64();

  // Read metadata
  const metadata: Record<string, unknown> = {};
  for (let i = 0; i < metadataKVCount; i++) {
    const key = reader.readString();
    const valueType = reader.readUint32();
    const value = reader.readValue(valueType);
    metadata[key] = value;
  }

  // Extract architecture info
  const architecture = (metadata['general.architecture'] as string) || 'unknown';
  const modelName = (metadata['general.name'] as string) || 'unknown';

  // Extract model config based on architecture
  const config = extractModelConfig(metadata, architecture);

  // Read tensor info
  const tensors: GGUFTensor[] = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = reader.readString();
    const nDims = reader.readUint32();
    const shape: number[] = [];
    for (let d = 0; d < nDims; d++) {
      shape.push(reader.readUint64());
    }
    const type = reader.readUint32();
    const offset = reader.readUint64();

    tensors.push({
      name,
      shape,
      dtype: GGMLTypeName[type] || `unknown_${type}`,
      dtypeId: type,
      offset,
      size: calculateTensorSize(shape, type),
    });
  }

  // Align to get tensor data offset
  reader.align(32);
  const tensorDataOffset = reader.offset;

  // Update tensor offsets to be absolute
  for (const tensor of tensors) {
    tensor.offset += tensorDataOffset;
  }

  // Determine primary quantization type
  const quantization = detectQuantization(tensors);

  // Calculate total tensor data size
  const totalTensorSize = tensors.reduce((sum, t) => sum + t.size, 0);

  return {
    version,
    architecture,
    modelName,
    metadata,
    config,
    tensors,
    quantization,
    tensorDataOffset,
    totalTensorSize,
    headerSize: tensorDataOffset,
  };
}

/**
 * Parse just the GGUF header (first ~1MB is usually enough)
 * Use this for large files to avoid loading the entire file into memory
 */
export function parseGGUFHeader(headerBuffer: ArrayBuffer): ParsedGGUF {
  return parseGGUF(headerBuffer);
}

/**
 * Get tensor by name
 */
export function getTensor(parsed: ParsedGGUF, name: string): GGUFTensor | null {
  return parsed.tensors.find((t) => t.name === name) || null;
}

/**
 * Get all tensors matching pattern
 */
export function getTensors(parsed: ParsedGGUF, pattern: RegExp): GGUFTensor[] {
  return parsed.tensors.filter((t) => pattern.test(t.name));
}

// ============================================================================
// Type Aliases for API Compatibility
// ============================================================================

/**
 * @deprecated Use ParsedGGUF instead
 */
export type GGUFParseResult = ParsedGGUF;
