import { describe, expect, it, beforeAll } from 'vitest';
import { readFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

import {
  parseGGUF,
  parseGGUFHeader,
  getTensor,
  getTensors,
  groupTensorsByLayer,
  GGUFValueType,
  GGMLType,
  GGMLTypeName,
  GGML_BLOCK_SIZE,
  GGML_TYPE_SIZE,
} from '../../src/formats/gguf.js';
import { parseGGUFFile } from '../../src/formats/gguf/parser.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = join(__dirname, '..', 'fixtures');

function createMinimalGGUFBuffer() {
  const encoder = new TextEncoder();

  const metadataKVs = [
    { key: 'general.architecture', type: GGUFValueType.STRING, value: 'llama' },
    { key: 'general.name', type: GGUFValueType.STRING, value: 'test-model' },
    { key: 'llama.context_length', type: GGUFValueType.UINT32, value: 2048 },
    { key: 'llama.embedding_length', type: GGUFValueType.UINT32, value: 256 },
    { key: 'llama.block_count', type: GGUFValueType.UINT32, value: 4 },
  ];

  const tensors = [
    { name: 'token_embd.weight', shape: [256, 1000], dtype: GGMLType.F16 },
    { name: 'blk.0.attn_q.weight', shape: [256, 256], dtype: GGMLType.F16 },
    { name: 'blk.0.attn_k.weight', shape: [256, 256], dtype: GGMLType.F16 },
    { name: 'blk.1.attn_q.weight', shape: [256, 256], dtype: GGMLType.Q4_K },
  ];

  let totalSize = 0;
  totalSize += 4 + 4 + 8 + 8;

  for (const kv of metadataKVs) {
    totalSize += 8 + encoder.encode(kv.key).length;
    totalSize += 4;
    if (kv.type === GGUFValueType.STRING) {
      totalSize += 8 + encoder.encode(kv.value).length;
    } else if (kv.type === GGUFValueType.UINT32) {
      totalSize += 4;
    }
  }

  for (const tensor of tensors) {
    totalSize += 8 + encoder.encode(tensor.name).length;
    totalSize += 4;
    totalSize += tensor.shape.length * 8;
    totalSize += 4;
    totalSize += 8;
  }

  const padding = (32 - (totalSize % 32)) % 32;
  totalSize += padding;

  let tensorDataSize = 0;
  for (const tensor of tensors) {
    const numElements = tensor.shape.reduce((a, b) => a * b, 1);
    if (tensor.dtype === GGMLType.F16) {
      tensorDataSize += numElements * 2;
    } else if (tensor.dtype === GGMLType.Q4_K) {
      const numBlocks = Math.ceil(numElements / GGML_BLOCK_SIZE[GGMLType.Q4_K]);
      tensorDataSize += numBlocks * GGML_TYPE_SIZE[GGMLType.Q4_K];
    }
  }

  totalSize += tensorDataSize;

  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  let offset = 0;

  function writeUint32(val) {
    view.setUint32(offset, val, true);
    offset += 4;
  }

  function writeUint64(val) {
    view.setUint32(offset, val & 0xffffffff, true);
    view.setUint32(offset + 4, Math.floor(val / 0x100000000), true);
    offset += 8;
  }

  function writeString(str) {
    const encoded = encoder.encode(str);
    writeUint64(encoded.length);
    bytes.set(encoded, offset);
    offset += encoded.length;
  }

  writeUint32(0x46554747);
  writeUint32(3);
  writeUint64(tensors.length);
  writeUint64(metadataKVs.length);

  for (const kv of metadataKVs) {
    writeString(kv.key);
    writeUint32(kv.type);
    if (kv.type === GGUFValueType.STRING) {
      writeString(kv.value);
    } else if (kv.type === GGUFValueType.UINT32) {
      writeUint32(kv.value);
    }
  }

  let tensorOffset = 0;
  for (const tensor of tensors) {
    writeString(tensor.name);
    writeUint32(tensor.shape.length);
    for (const dim of tensor.shape) {
      writeUint64(dim);
    }
    writeUint32(tensor.dtype);
    writeUint64(tensorOffset);

    const numElements = tensor.shape.reduce((a, b) => a * b, 1);
    if (tensor.dtype === GGMLType.F16) {
      tensorOffset += numElements * 2;
    } else if (tensor.dtype === GGMLType.Q4_K) {
      const numBlocks = Math.ceil(numElements / GGML_BLOCK_SIZE[GGMLType.Q4_K]);
      tensorOffset += numBlocks * GGML_TYPE_SIZE[GGMLType.Q4_K];
    }
  }

  const remainder = offset % 32;
  if (remainder !== 0) {
    offset += 32 - remainder;
  }

  return buffer;
}

describe('formats/gguf', () => {
  let testBuffer;

  beforeAll(() => {
    testBuffer = createMinimalGGUFBuffer();
  });

  describe('parseGGUF', () => {
    it('parses GGUF header correctly', () => {
      const result = parseGGUF(testBuffer);

      expect(result.version).toBe(3);
      expect(result.architecture).toBe('llama');
      expect(result.modelName).toBe('test-model');
    });

    it('extracts metadata fields', () => {
      const result = parseGGUF(testBuffer);

      expect(result.metadata['general.architecture']).toBe('llama');
      expect(result.metadata['general.name']).toBe('test-model');
      expect(result.metadata['llama.context_length']).toBe(2048);
      expect(result.metadata['llama.embedding_length']).toBe(256);
      expect(result.metadata['llama.block_count']).toBe(4);
    });

    it('parses tensor metadata', () => {
      const result = parseGGUF(testBuffer);

      expect(result.tensors.length).toBe(4);

      const embedTensor = result.tensors.find(t => t.name === 'token_embd.weight');
      expect(embedTensor).toBeDefined();
      expect(embedTensor.shape).toEqual([256, 1000]);
      expect(embedTensor.dtype).toBe('F16');
      expect(embedTensor.dtypeId).toBe(GGMLType.F16);

      const q4Tensor = result.tensors.find(t => t.name === 'blk.1.attn_q.weight');
      expect(q4Tensor).toBeDefined();
      expect(q4Tensor.dtype).toBe('Q4_K');
      expect(q4Tensor.dtypeId).toBe(GGMLType.Q4_K);
    });

    it('calculates tensor sizes correctly', () => {
      const result = parseGGUF(testBuffer);

      const embedTensor = result.tensors.find(t => t.name === 'token_embd.weight');
      expect(embedTensor.size).toBe(256 * 1000 * 2);

      const f16Tensor = result.tensors.find(t => t.name === 'blk.0.attn_q.weight');
      expect(f16Tensor.size).toBe(256 * 256 * 2);
    });

    it('calculates tensor data offset', () => {
      const result = parseGGUF(testBuffer);

      expect(result.tensorDataOffset).toBeGreaterThan(0);
      expect(result.tensorDataOffset % 32).toBe(0);
    });

    it('calculates total tensor size', () => {
      const result = parseGGUF(testBuffer);

      const manualTotal = result.tensors.reduce((sum, t) => sum + t.size, 0);
      expect(result.totalTensorSize).toBe(manualTotal);
    });

    it('extracts model config', () => {
      const result = parseGGUF(testBuffer);

      expect(result.config.architecture).toBe('llama');
      expect(result.config.contextLength).toBe(2048);
      expect(result.config.embeddingLength).toBe(256);
      expect(result.config.blockCount).toBe(4);
    });

    it('detects quantization type', () => {
      const result = parseGGUF(testBuffer);

      expect(result.quantization).toBeDefined();
      expect(typeof result.quantization).toBe('string');
    });

    it('throws on invalid magic number', () => {
      const badBuffer = new ArrayBuffer(100);
      const view = new DataView(badBuffer);
      view.setUint32(0, 0x12345678, true);

      expect(() => parseGGUF(badBuffer)).toThrow(/Invalid GGUF magic/);
    });

    it('throws on unsupported version', () => {
      const badBuffer = new ArrayBuffer(100);
      const view = new DataView(badBuffer);
      view.setUint32(0, 0x46554747, true);
      view.setUint32(4, 1, true);

      expect(() => parseGGUF(badBuffer)).toThrow(/Unsupported GGUF version/);
    });
  });

  describe('parseGGUFHeader', () => {
    it('is an alias for parseGGUF', () => {
      const result = parseGGUFHeader(testBuffer);

      expect(result.version).toBe(3);
      expect(result.architecture).toBe('llama');
    });
  });

  describe('getTensor', () => {
    it('returns tensor by name', () => {
      const parsed = parseGGUF(testBuffer);
      const tensor = getTensor(parsed, 'token_embd.weight');

      expect(tensor).not.toBeNull();
      expect(tensor.name).toBe('token_embd.weight');
    });

    it('returns null for unknown tensor', () => {
      const parsed = parseGGUF(testBuffer);
      const tensor = getTensor(parsed, 'nonexistent');

      expect(tensor).toBeNull();
    });
  });

  describe('getTensors', () => {
    it('returns tensors matching pattern', () => {
      const parsed = parseGGUF(testBuffer);
      const tensors = getTensors(parsed, /blk\.0/);

      expect(tensors.length).toBe(2);
      expect(tensors.every(t => t.name.includes('blk.0'))).toBe(true);
    });

    it('returns empty array for no matches', () => {
      const parsed = parseGGUF(testBuffer);
      const tensors = getTensors(parsed, /blk\.99/);

      expect(tensors.length).toBe(0);
    });
  });

  describe('groupTensorsByLayer', () => {
    it('groups tensors by layer index', () => {
      const parsed = parseGGUF(testBuffer);
      const layers = groupTensorsByLayer(parsed);

      expect(layers.has(0)).toBe(true);
      expect(layers.has(1)).toBe(true);
      expect(layers.get(0).length).toBe(2);
      expect(layers.get(1).length).toBe(1);
    });

    it('excludes non-layer tensors', () => {
      const parsed = parseGGUF(testBuffer);
      const layers = groupTensorsByLayer(parsed);

      const allLayerTensors = Array.from(layers.values()).flat();
      const embedTensor = allLayerTensors.find(t => t.name === 'token_embd.weight');
      expect(embedTensor).toBeUndefined();
    });
  });

  describe('GGMLType constants', () => {
    it('has correct type IDs', () => {
      expect(GGMLType.F32).toBe(0);
      expect(GGMLType.F16).toBe(1);
      expect(GGMLType.Q4_K).toBe(12);
      expect(GGMLType.BF16).toBe(29);
    });

    it('has type names for all types', () => {
      expect(GGMLTypeName[GGMLType.F32]).toBe('F32');
      expect(GGMLTypeName[GGMLType.F16]).toBe('F16');
      expect(GGMLTypeName[GGMLType.Q4_K]).toBe('Q4_K');
    });
  });

  describe('GGML_BLOCK_SIZE', () => {
    it('has correct block sizes', () => {
      expect(GGML_BLOCK_SIZE[GGMLType.Q4_0]).toBe(32);
      expect(GGML_BLOCK_SIZE[GGMLType.Q4_K]).toBe(256);
      expect(GGML_BLOCK_SIZE[GGMLType.Q6_K]).toBe(256);
    });
  });

  describe('GGML_TYPE_SIZE', () => {
    it('has correct type sizes', () => {
      expect(GGML_TYPE_SIZE[GGMLType.F32]).toBe(4);
      expect(GGML_TYPE_SIZE[GGMLType.F16]).toBe(2);
      expect(GGML_TYPE_SIZE[GGMLType.Q4_K]).toBe(144);
      expect(GGML_TYPE_SIZE[GGMLType.BF16]).toBe(2);
    });
  });
});

describe('formats/gguf with fixture file', () => {
  const fixturePath = join(FIXTURES_DIR, 'sample.gguf');

  describe('parseGGUFFile', () => {
    it('parses fixture file correctly', async () => {
      const result = await parseGGUFFile(fixturePath);

      expect(result.version).toBe(3);
      expect(result.architecture).toBe('llama');
      expect(result.modelName).toBe('test-fixture');
    });

    it('extracts all metadata', async () => {
      const result = await parseGGUFFile(fixturePath);

      expect(result.metadata['general.architecture']).toBe('llama');
      expect(result.metadata['llama.context_length']).toBe(512);
      expect(result.metadata['llama.embedding_length']).toBe(64);
      expect(result.metadata['llama.block_count']).toBe(2);
      expect(result.metadata['llama.attention.head_count']).toBe(4);
    });

    it('extracts tokenizer metadata', async () => {
      const result = await parseGGUFFile(fixturePath);

      expect(result.config.tokenizer.model).toBe('llama');
      expect(result.config.tokenizer.bosTokenId).toBe(1);
      expect(result.config.tokenizer.eosTokenId).toBe(2);
    });

    it('parses all tensors', async () => {
      const result = await parseGGUFFile(fixturePath);

      expect(result.tensors.length).toBe(9);

      const tensorNames = result.tensors.map(t => t.name);
      expect(tensorNames).toContain('token_embd.weight');
      expect(tensorNames).toContain('blk.0.attn_q.weight');
      expect(tensorNames).toContain('blk.0.ffn_gate.weight');
      expect(tensorNames).toContain('blk.1.attn_q.weight');
      expect(tensorNames).toContain('output.weight');
    });

    it('stores file metadata', async () => {
      const result = await parseGGUFFile(fixturePath);

      expect(result.filePath).toBe(fixturePath);
      expect(result.fileSize).toBeGreaterThan(0);
    });

    it('groups tensors by layer correctly', async () => {
      const result = await parseGGUFFile(fixturePath);
      const layers = groupTensorsByLayer(result);

      expect(layers.has(0)).toBe(true);
      expect(layers.has(1)).toBe(true);
      expect(layers.get(0).length).toBeGreaterThan(0);
    });

    it('calculates tensor data offset with alignment', async () => {
      const result = await parseGGUFFile(fixturePath);

      expect(result.tensorDataOffset % 32).toBe(0);
    });
  });

  describe('parseGGUF from buffer', () => {
    it('parses fixture loaded as buffer', async () => {
      const fileBuffer = await readFile(fixturePath);
      const arrayBuffer = fileBuffer.buffer.slice(
        fileBuffer.byteOffset,
        fileBuffer.byteOffset + fileBuffer.byteLength
      );

      const result = parseGGUF(arrayBuffer);

      expect(result.version).toBe(3);
      expect(result.architecture).toBe('llama');
      expect(result.tensors.length).toBe(9);
    });
  });
});
