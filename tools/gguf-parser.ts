/**
 * GGUF Model Format Parser
 * Parses GGUF files (llama.cpp format) for metadata, tensors, and quantization info.
 */

import { readFile } from 'fs/promises';

const GGUF_MAGIC = 0x46554747;
const GGUF_VERSION_MIN = 2;
const GGUF_VERSION_MAX = 3;

export const GGUFValueType = {
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12,
} as const;

export type GGUFValueTypeId = (typeof GGUFValueType)[keyof typeof GGUFValueType];

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

export type GGMLTypeId = (typeof GGMLType)[keyof typeof GGMLType];

export const GGMLTypeName: Record<number, string> = Object.fromEntries(
  Object.entries(GGMLType).map(([k, v]) => [v, k])
);

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

export interface GGUFTensor {
  name: string;
  shape: number[];
  dtype: string;
  dtypeId: number;
  offset: number;
  size: number;
}

export interface GGUFTokenizer {
  model?: string;
  tokens?: string[];
  scores?: number[];
  tokenTypes?: number[];
  merges?: string[];
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

export interface GGUFConfig {
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
  tokenizer: GGUFTokenizer;
}

export interface GGUFParseResult {
  version: number;
  architecture: string;
  modelName: string;
  metadata: Record<string, unknown>;
  config: GGUFConfig;
  tensors: GGUFTensor[];
  quantization: string;
  tensorDataOffset: number;
  fileSize: number;
}

type GGUFValue = string | number | boolean | GGUFValue[];

class GGUFReader {
  private buffer: ArrayBuffer;
  private view: DataView;
  offset = 0;

  constructor(buffer: ArrayBuffer) {
    this.buffer = buffer;
    this.view = new DataView(buffer);
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

  readValue(type: GGUFValueTypeId): GGUFValue {
    switch (type) {
      case GGUFValueType.UINT8: return this.readUint8();
      case GGUFValueType.INT8: return this.readInt8();
      case GGUFValueType.UINT16: return this.readUint16();
      case GGUFValueType.INT16: return this.readInt16();
      case GGUFValueType.UINT32: return this.readUint32();
      case GGUFValueType.INT32: return this.readInt32();
      case GGUFValueType.UINT64: return this.readUint64();
      case GGUFValueType.INT64: return this.readInt64();
      case GGUFValueType.FLOAT32: return this.readFloat32();
      case GGUFValueType.FLOAT64: return this.readFloat64();
      case GGUFValueType.BOOL: return this.readBool();
      case GGUFValueType.STRING: return this.readString();
      case GGUFValueType.ARRAY: return this.readArray();
      default: throw new Error(`Unknown value type: ${type}`);
    }
  }

  readArray(): GGUFValue[] {
    const elementType = this.readUint32() as GGUFValueTypeId;
    const length = this.readUint64();
    if (length > 10000000) {
      throw new Error(`Array too long: ${length}`);
    }
    const arr: GGUFValue[] = [];
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

function calculateTensorSize(shape: number[], type: number): number {
  const numElements = shape.reduce((a, b) => a * b, 1);

  if (type === GGMLType.F32) return numElements * 4;
  if (type === GGMLType.F16 || type === GGMLType.BF16) return numElements * 2;
  if (type === GGMLType.I8) return numElements;
  if (type === GGMLType.I16) return numElements * 2;
  if (type === GGMLType.I32) return numElements * 4;
  if (type === GGMLType.I64) return numElements * 8;
  if (type === GGMLType.F64) return numElements * 8;

  const blockSize = GGML_BLOCK_SIZE[type];
  const typeSize = GGML_TYPE_SIZE[type];
  if (blockSize && typeSize) {
    const numBlocks = Math.ceil(numElements / blockSize);
    return numBlocks * typeSize;
  }

  throw new Error(`Unknown tensor type: ${type}`);
}

function extractModelConfig(metadata: Record<string, unknown>, architecture: string): GGUFConfig {
  const prefix = `${architecture}.`;
  const get = <T>(key: string): T | undefined => metadata[key] as T | undefined;

  return {
    architecture,
    vocabSize: get<number>(`${prefix}vocab_size`) ?? get<number>('tokenizer.ggml.vocab_size'),
    contextLength: get<number>(`${prefix}context_length`) ?? 2048,
    embeddingLength: get<number>(`${prefix}embedding_length`),
    blockCount: get<number>(`${prefix}block_count`),
    feedForwardLength: get<number>(`${prefix}feed_forward_length`),
    attentionHeadCount: get<number>(`${prefix}attention.head_count`),
    attentionHeadCountKV: get<number>(`${prefix}attention.head_count_kv`),
    attentionLayerNormEpsilon: get<number>(`${prefix}attention.layer_norm_epsilon`) ?? 1e-5,
    attentionLayerNormRMSEpsilon: get<number>(`${prefix}attention.layer_norm_rms_epsilon`) ?? 1e-5,
    ropeFreqBase: get<number>(`${prefix}rope.freq_base`) ?? 10000,
    ropeScalingType: get<string>(`${prefix}rope.scaling.type`),
    ropeScalingFactor: get<number>(`${prefix}rope.scaling.factor`),
    expertCount: get<number>(`${prefix}expert_count`),
    expertUsedCount: get<number>(`${prefix}expert_used_count`),
    tokenizer: {
      model: get<string>('tokenizer.ggml.model'),
      tokens: get<string[]>('tokenizer.ggml.tokens'),
      scores: get<number[]>('tokenizer.ggml.scores'),
      tokenTypes: get<number[]>('tokenizer.ggml.token_type'),
      merges: get<string[]>('tokenizer.ggml.merges'),
      bosTokenId: get<number>('tokenizer.ggml.bos_token_id'),
      eosTokenId: get<number>('tokenizer.ggml.eos_token_id'),
      padTokenId: get<number>('tokenizer.ggml.padding_token_id'),
      unkTokenId: get<number>('tokenizer.ggml.unknown_token_id'),
      sepTokenId: get<number>('tokenizer.ggml.seperator_token_id'),
      clsTokenId: get<number>('tokenizer.ggml.cls_token_id'),
      maskTokenId: get<number>('tokenizer.ggml.mask_token_id'),
      addBosToken: get<boolean>('tokenizer.ggml.add_bos_token'),
      addEosToken: get<boolean>('tokenizer.ggml.add_eos_token'),
      addSpacePrefix: get<boolean>('tokenizer.ggml.add_space_prefix'),
    },
  };
}

function detectQuantization(tensors: GGUFTensor[]): string {
  const typeCounts: Record<string, number> = {};

  for (const tensor of tensors) {
    if (tensor.name.includes('embed') || tensor.name.includes('output')) continue;
    typeCounts[tensor.dtype] = (typeCounts[tensor.dtype] || 0) + tensor.size;
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

export function parseGGUF(buffer: ArrayBuffer): GGUFParseResult {
  const reader = new GGUFReader(buffer);

  const magic = reader.readUint32();
  if (magic !== GGUF_MAGIC) {
    throw new Error(`Invalid GGUF magic: 0x${magic.toString(16)}`);
  }

  const version = reader.readUint32();
  if (version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX) {
    throw new Error(`Unsupported GGUF version: ${version}`);
  }

  const tensorCount = reader.readUint64();
  const metadataKVCount = reader.readUint64();

  const metadata: Record<string, unknown> = {};
  for (let i = 0; i < metadataKVCount; i++) {
    const key = reader.readString();
    const valueType = reader.readUint32() as GGUFValueTypeId;
    metadata[key] = reader.readValue(valueType);
  }

  const architecture = (metadata['general.architecture'] as string) || 'unknown';
  const modelName = (metadata['general.name'] as string) || 'unknown';
  const config = extractModelConfig(metadata, architecture);

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

  reader.align(32);
  const tensorDataOffset = reader.offset;

  for (const tensor of tensors) {
    tensor.offset += tensorDataOffset;
  }

  return {
    version,
    architecture,
    modelName,
    metadata,
    config,
    tensors,
    quantization: detectQuantization(tensors),
    tensorDataOffset,
    fileSize: buffer.byteLength,
  };
}

export async function parseGGUFFile(filePath: string): Promise<GGUFParseResult> {
  const buffer = await readFile(filePath);
  return parseGGUF(buffer.buffer as ArrayBuffer);
}

export function getTensor(parsed: GGUFParseResult, name: string): GGUFTensor | null {
  return parsed.tensors.find(t => t.name === name) || null;
}

export function getTensors(parsed: GGUFParseResult, pattern: RegExp): GGUFTensor[] {
  return parsed.tensors.filter(t => pattern.test(t.name));
}

export function groupTensorsByLayer(parsed: GGUFParseResult): Map<number, GGUFTensor[]> {
  const layers = new Map<number, GGUFTensor[]>();

  for (const tensor of parsed.tensors) {
    const match = tensor.name.match(/(?:blk|layers?)\.(\d+)\./);
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

export function identifyMoETensors(parsed: GGUFParseResult): Map<number, Map<number, GGUFTensor[]>> {
  const moe = new Map<number, Map<number, GGUFTensor[]>>();

  for (const tensor of parsed.tensors) {
    const match = tensor.name.match(/(?:blk|layers?)\.(\d+)\..*(?:expert|exp)s?\.(\d+)\./);
    if (match) {
      const layerIdx = parseInt(match[1], 10);
      const expertIdx = parseInt(match[2], 10);

      if (!moe.has(layerIdx)) {
        moe.set(layerIdx, new Map());
      }
      if (!moe.get(layerIdx)!.has(expertIdx)) {
        moe.get(layerIdx)!.set(expertIdx, []);
      }
      moe.get(layerIdx)!.get(expertIdx)!.push(tensor);
    }
  }

  return moe;
}
