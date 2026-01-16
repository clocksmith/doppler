

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
};

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
};

export const GGMLTypeName = Object.fromEntries(
  Object.entries(GGMLType).map(([key, value]) => [value, key])
);

export const GGML_BLOCK_SIZE = {
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

export const GGML_TYPE_SIZE = {
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

const GGUF_MAGIC = 0x46554747;
const GGUF_VERSION_MIN = 2;
const GGUF_VERSION_MAX = 3;

class GGUFReader {
  constructor(buffer) {
    this.view = new DataView(buffer);
    this.offset = 0;
  }

  readUint8() {
    const value = this.view.getUint8(this.offset);
    this.offset += 1;
    return value;
  }

  readInt8() {
    const value = this.view.getInt8(this.offset);
    this.offset += 1;
    return value;
  }

  readUint16() {
    const value = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return value;
  }

  readInt16() {
    const value = this.view.getInt16(this.offset, true);
    this.offset += 2;
    return value;
  }

  readUint32() {
    const value = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return value;
  }

  readInt32() {
    const value = this.view.getInt32(this.offset, true);
    this.offset += 4;
    return value;
  }

  readUint64() {
    const low = this.view.getUint32(this.offset, true);
    const high = this.view.getUint32(this.offset + 4, true);
    this.offset += 8;
    return high * 0x100000000 + low;
  }

  readInt64() {
    const low = this.view.getUint32(this.offset, true);
    const high = this.view.getInt32(this.offset + 4, true);
    this.offset += 8;
    return high * 0x100000000 + low;
  }

  readFloat32() {
    const value = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return value;
  }

  readFloat64() {
    const value = this.view.getFloat64(this.offset, true);
    this.offset += 8;
    return value;
  }

  readBool() {
    return this.readUint8() !== 0;
  }

  readString() {
    const length = this.readUint64();
    const bytes = new Uint8Array(this.view.buffer, this.offset, length);
    this.offset += length;
    return new TextDecoder().decode(bytes);
  }

  readValue(type) {
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

  readArray() {
    const elementType = this.readUint32();
    const length = this.readUint64();
    if (length > 10000000) {
      throw new Error(`Array too long: ${length}`);
    }
    const result = [];
    for (let i = 0; i < length; i++) {
      result.push(this.readValue(elementType));
    }
    return result;
  }

  align(boundary) {
    const remainder = this.offset % boundary;
    if (remainder !== 0) {
      this.offset += boundary - remainder;
    }
  }
}

function calculateTensorSize(shape, type) {
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

function extractModelConfig(metadata, architecture) {
  const prefix = `${architecture}.`;
  const get = (key) => metadata[key];

  return {
    architecture,
    vocabSize: get(`${prefix}vocab_size`) ?? get('tokenizer.ggml.vocab_size'),
    contextLength: get(`${prefix}context_length`) ?? 2048,
    embeddingLength: get(`${prefix}embedding_length`),
    blockCount: get(`${prefix}block_count`),
    feedForwardLength: get(`${prefix}feed_forward_length`),
    attentionHeadCount: get(`${prefix}attention.head_count`),
    attentionHeadCountKV: get(`${prefix}attention.head_count_kv`),
    attentionLayerNormEpsilon: get(`${prefix}attention.layer_norm_epsilon`) ?? 1e-5,
    attentionLayerNormRMSEpsilon: get(`${prefix}attention.layer_norm_rms_epsilon`) ?? 1e-5,
    ropeFreqBase: get(`${prefix}rope.freq_base`) ?? 10000,
    ropeScalingType: get(`${prefix}rope.scaling.type`),
    ropeScalingFactor: get(`${prefix}rope.scaling.factor`),
    expertCount: get(`${prefix}expert_count`),
    expertUsedCount: get(`${prefix}expert_used_count`),
    tokenizer: {
      model: get('tokenizer.ggml.model'),
      tokens: get('tokenizer.ggml.tokens'),
      scores: get('tokenizer.ggml.scores'),
      tokenTypes: get('tokenizer.ggml.token_type'),
      merges: get('tokenizer.ggml.merges'),
      bosTokenId: get('tokenizer.ggml.bos_token_id'),
      eosTokenId: get('tokenizer.ggml.eos_token_id'),
      padTokenId: get('tokenizer.ggml.padding_token_id'),
      unkTokenId: get('tokenizer.ggml.unknown_token_id'),
      sepTokenId: get('tokenizer.ggml.seperator_token_id'),
      clsTokenId: get('tokenizer.ggml.cls_token_id'),
      maskTokenId: get('tokenizer.ggml.mask_token_id'),
      addBosToken: get('tokenizer.ggml.add_bos_token'),
      addEosToken: get('tokenizer.ggml.add_eos_token'),
      addSpacePrefix: get('tokenizer.ggml.add_space_prefix'),
    },
  };
}

function detectQuantization(tensors) {
  const typeCounts = {};

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

export function parseGGUF(buffer) {
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

  const metadata = {};
  for (let i = 0; i < metadataKVCount; i++) {
    const key = reader.readString();
    const valueType = reader.readUint32();
    metadata[key] = reader.readValue(valueType);
  }

  const architecture = metadata['general.architecture'] || 'unknown';
  const modelName = metadata['general.name'] || 'unknown';
  const config = extractModelConfig(metadata, architecture);

  const tensors = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = reader.readString();
    const nDims = reader.readUint32();
    const shape = [];
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

  const totalTensorSize = tensors.reduce((sum, tensor) => sum + tensor.size, 0);

  return {
    version,
    architecture,
    modelName,
    metadata,
    config,
    tensors,
    quantization: detectQuantization(tensors),
    tensorDataOffset,
    totalTensorSize,
    headerSize: tensorDataOffset,
    fileSize: buffer.byteLength,
  };
}

export function parseGGUFHeader(buffer) {
  return parseGGUF(buffer);
}

export function getTensor(parsed, name) {
  return parsed.tensors.find((tensor) => tensor.name === name) || null;
}

export function getTensors(parsed, pattern) {
  return parsed.tensors.filter((tensor) => pattern.test(tensor.name));
}

export function groupTensorsByLayer(parsed) {
  const layers = new Map();

  for (const tensor of parsed.tensors) {
    const match = tensor.name.match(/(?:blk|layers?)\.(\d+)\./);
    if (match) {
      const layerIdx = parseInt(match[1], 10);
      if (!layers.has(layerIdx)) {
        layers.set(layerIdx, []);
      }
      layers.get(layerIdx).push(tensor);
    }
  }

  return layers;
}

export function identifyMoETensors(parsed) {
  const result = new Map();
  const layers = groupTensorsByLayer(parsed);

  for (const [layerIdx, tensors] of layers.entries()) {
    const experts = new Map();
    for (const tensor of tensors) {
      const match = tensor.name.match(/experts\.(\d+)\./);
      if (!match) continue;
      const expertIdx = parseInt(match[1], 10);
      if (!experts.has(expertIdx)) {
        experts.set(expertIdx, []);
      }
      experts.get(expertIdx).push(tensor);
    }
    if (experts.size > 0) {
      result.set(layerIdx, experts);
    }
  }

  return result;
}
