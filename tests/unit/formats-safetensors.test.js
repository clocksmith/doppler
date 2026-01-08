import { describe, expect, it, beforeAll } from 'vitest';
import { readFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

import {
  parseSafetensorsHeader,
  DTYPE_SIZE,
  DTYPE_MAP,
  groupTensorsByLayer,
  calculateTotalSize,
} from '../../src/formats/safetensors/types.js';
import {
  parseSafetensorsFile,
  getTensor,
  getTensors,
} from '../../src/formats/safetensors/parser.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = join(__dirname, '..', 'fixtures');

function createMinimalSafetensorsBuffer() {
  const header = {
    '__metadata__': {
      'format': 'pt',
      'model_type': 'test',
    },
    'model.embed_tokens.weight': {
      'dtype': 'F16',
      'shape': [1000, 256],
      'data_offsets': [0, 512000],
    },
    'model.layers.0.self_attn.q_proj.weight': {
      'dtype': 'F32',
      'shape': [256, 256],
      'data_offsets': [512000, 774144],
    },
    'model.layers.0.self_attn.k_proj.weight': {
      'dtype': 'F32',
      'shape': [256, 256],
      'data_offsets': [774144, 1036288],
    },
    'model.layers.1.mlp.gate_proj.weight': {
      'dtype': 'BF16',
      'shape': [512, 256],
      'data_offsets': [1036288, 1298432],
    },
  };

  const headerJson = JSON.stringify(header);
  const headerBytes = new TextEncoder().encode(headerJson);
  const headerSize = headerBytes.length;

  const dataSize = 1298432;
  const totalSize = 8 + headerSize + dataSize;

  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  view.setUint32(0, headerSize & 0xffffffff, true);
  view.setUint32(4, Math.floor(headerSize / 0x100000000), true);

  bytes.set(headerBytes, 8);

  return buffer;
}

function createSafetensorsBufferWithLayers(numLayers) {
  const tensors = {};
  let offset = 0;

  tensors['model.embed_tokens.weight'] = {
    dtype: 'F16',
    shape: [100, 64],
    data_offsets: [offset, offset + 12800],
  };
  offset += 12800;

  for (let i = 0; i < numLayers; i++) {
    const layerSize = 64 * 64 * 4;

    tensors[`model.layers.${i}.self_attn.q_proj.weight`] = {
      dtype: 'F32',
      shape: [64, 64],
      data_offsets: [offset, offset + layerSize],
    };
    offset += layerSize;

    tensors[`model.layers.${i}.self_attn.k_proj.weight`] = {
      dtype: 'F32',
      shape: [64, 64],
      data_offsets: [offset, offset + layerSize],
    };
    offset += layerSize;

    tensors[`model.layers.${i}.mlp.gate_proj.weight`] = {
      dtype: 'F32',
      shape: [64, 64],
      data_offsets: [offset, offset + layerSize],
    };
    offset += layerSize;
  }

  tensors['lm_head.weight'] = {
    dtype: 'F16',
    shape: [100, 64],
    data_offsets: [offset, offset + 12800],
  };
  offset += 12800;

  const header = { '__metadata__': { format: 'pt' }, ...tensors };
  const headerJson = JSON.stringify(header);
  const headerBytes = new TextEncoder().encode(headerJson);
  const headerSize = headerBytes.length;

  const totalSize = 8 + headerSize + offset;
  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  view.setUint32(0, headerSize & 0xffffffff, true);
  view.setUint32(4, Math.floor(headerSize / 0x100000000), true);
  bytes.set(headerBytes, 8);

  return buffer;
}

describe('formats/safetensors', () => {
  let testBuffer;

  beforeAll(() => {
    testBuffer = createMinimalSafetensorsBuffer();
  });

  describe('parseSafetensorsHeader', () => {
    it('parses header size correctly', () => {
      const result = parseSafetensorsHeader(testBuffer);

      expect(result.headerSize).toBeGreaterThan(0);
      expect(result.dataOffset).toBe(8 + result.headerSize);
    });

    it('extracts metadata', () => {
      const result = parseSafetensorsHeader(testBuffer);

      expect(result.metadata).toBeDefined();
      expect(result.metadata.format).toBe('pt');
      expect(result.metadata.model_type).toBe('test');
    });

    it('parses tensor information', () => {
      const result = parseSafetensorsHeader(testBuffer);

      expect(result.tensors.length).toBe(4);

      const embedTensor = result.tensors.find(t => t.name === 'model.embed_tokens.weight');
      expect(embedTensor).toBeDefined();
      expect(embedTensor.dtype).toBe('F16');
      expect(embedTensor.shape).toEqual([1000, 256]);
    });

    it('calculates tensor offsets correctly', () => {
      const result = parseSafetensorsHeader(testBuffer);

      const embedTensor = result.tensors.find(t => t.name === 'model.embed_tokens.weight');
      expect(embedTensor.offset).toBe(result.dataOffset + 0);
      expect(embedTensor.size).toBe(512000);

      const qTensor = result.tensors.find(t => t.name === 'model.layers.0.self_attn.q_proj.weight');
      expect(qTensor.offset).toBe(result.dataOffset + 512000);
      expect(qTensor.size).toBe(262144);
    });

    it('sets element size based on dtype', () => {
      const result = parseSafetensorsHeader(testBuffer);

      const f16Tensor = result.tensors.find(t => t.dtype === 'F16');
      expect(f16Tensor.elemSize).toBe(2);

      const f32Tensor = result.tensors.find(t => t.dtype === 'F32');
      expect(f32Tensor.elemSize).toBe(4);

      const bf16Tensor = result.tensors.find(t => t.dtype === 'BF16');
      expect(bf16Tensor.elemSize).toBe(2);
    });

    it('sorts tensors by offset', () => {
      const result = parseSafetensorsHeader(testBuffer);

      for (let i = 1; i < result.tensors.length; i++) {
        expect(result.tensors[i].offset).toBeGreaterThanOrEqual(result.tensors[i - 1].offset);
      }
    });

    it('throws on header too large', () => {
      const badBuffer = new ArrayBuffer(16);
      const view = new DataView(badBuffer);
      view.setUint32(0, 200 * 1024 * 1024, true);

      expect(() => parseSafetensorsHeader(badBuffer)).toThrow(/Header too large/);
    });

    it('throws on buffer too small for header', () => {
      const badBuffer = new ArrayBuffer(16);
      const view = new DataView(badBuffer);
      view.setUint32(0, 1000, true);

      expect(() => parseSafetensorsHeader(badBuffer)).toThrow(/does not contain full/);
    });
  });

  describe('DTYPE_SIZE', () => {
    it('has correct sizes for all dtypes', () => {
      expect(DTYPE_SIZE.F64).toBe(8);
      expect(DTYPE_SIZE.F32).toBe(4);
      expect(DTYPE_SIZE.F16).toBe(2);
      expect(DTYPE_SIZE.BF16).toBe(2);
      expect(DTYPE_SIZE.I64).toBe(8);
      expect(DTYPE_SIZE.I32).toBe(4);
      expect(DTYPE_SIZE.I16).toBe(2);
      expect(DTYPE_SIZE.I8).toBe(1);
      expect(DTYPE_SIZE.U8).toBe(1);
      expect(DTYPE_SIZE.BOOL).toBe(1);
    });
  });

  describe('DTYPE_MAP', () => {
    it('maps dtype strings correctly', () => {
      expect(DTYPE_MAP.F32).toBe('F32');
      expect(DTYPE_MAP.F16).toBe('F16');
      expect(DTYPE_MAP.BF16).toBe('BF16');
    });
  });

  describe('groupTensorsByLayer', () => {
    it('groups tensors by layer index', () => {
      const buffer = createSafetensorsBufferWithLayers(3);
      const parsed = parseSafetensorsHeader(buffer);
      const layers = groupTensorsByLayer(parsed);

      expect(layers.has(0)).toBe(true);
      expect(layers.has(1)).toBe(true);
      expect(layers.has(2)).toBe(true);
      expect(layers.get(0).length).toBe(3);
      expect(layers.get(1).length).toBe(3);
      expect(layers.get(2).length).toBe(3);
    });

    it('excludes non-layer tensors', () => {
      const buffer = createSafetensorsBufferWithLayers(2);
      const parsed = parseSafetensorsHeader(buffer);
      const layers = groupTensorsByLayer(parsed);

      const allLayerTensors = Array.from(layers.values()).flat();
      const embedTensor = allLayerTensors.find(t => t.name === 'model.embed_tokens.weight');
      const headTensor = allLayerTensors.find(t => t.name === 'lm_head.weight');

      expect(embedTensor).toBeUndefined();
      expect(headTensor).toBeUndefined();
    });

    it('returns empty map for no layers', () => {
      const header = {
        '__metadata__': {},
        'embed.weight': {
          dtype: 'F16',
          shape: [100, 64],
          data_offsets: [0, 12800],
        },
      };
      const headerJson = JSON.stringify(header);
      const headerBytes = new TextEncoder().encode(headerJson);
      const headerSize = headerBytes.length;

      const buffer = new ArrayBuffer(8 + headerSize + 12800);
      const view = new DataView(buffer);
      const bytes = new Uint8Array(buffer);

      view.setUint32(0, headerSize, true);
      bytes.set(headerBytes, 8);

      const parsed = parseSafetensorsHeader(buffer);
      const layers = groupTensorsByLayer(parsed);

      expect(layers.size).toBe(0);
    });
  });

  describe('calculateTotalSize', () => {
    it('sums all tensor sizes', () => {
      const result = parseSafetensorsHeader(testBuffer);
      const total = calculateTotalSize(result);

      const manualTotal = result.tensors.reduce((sum, t) => sum + t.size, 0);
      expect(total).toBe(manualTotal);
    });

    it('handles empty tensor list', () => {
      const parsed = { tensors: [] };
      const total = calculateTotalSize(parsed);

      expect(total).toBe(0);
    });
  });

  describe('edge cases', () => {
    it('handles empty metadata', () => {
      const header = {
        'weight': {
          dtype: 'F32',
          shape: [10],
          data_offsets: [0, 40],
        },
      };
      const headerJson = JSON.stringify(header);
      const headerBytes = new TextEncoder().encode(headerJson);
      const headerSize = headerBytes.length;

      const buffer = new ArrayBuffer(8 + headerSize + 40);
      const view = new DataView(buffer);
      const bytes = new Uint8Array(buffer);

      view.setUint32(0, headerSize, true);
      bytes.set(headerBytes, 8);

      const result = parseSafetensorsHeader(buffer);

      expect(result.metadata).toEqual({});
      expect(result.tensors.length).toBe(1);
    });

    it('handles single tensor', () => {
      const header = {
        '__metadata__': {},
        'single.weight': {
          dtype: 'F32',
          shape: [10, 10],
          data_offsets: [0, 400],
        },
      };
      const headerJson = JSON.stringify(header);
      const headerBytes = new TextEncoder().encode(headerJson);
      const headerSize = headerBytes.length;

      const buffer = new ArrayBuffer(8 + headerSize + 400);
      const view = new DataView(buffer);
      const bytes = new Uint8Array(buffer);

      view.setUint32(0, headerSize, true);
      bytes.set(headerBytes, 8);

      const result = parseSafetensorsHeader(buffer);

      expect(result.tensors.length).toBe(1);
      expect(result.tensors[0].name).toBe('single.weight');
      expect(result.tensors[0].size).toBe(400);
    });

    it('handles multidimensional tensors', () => {
      const header = {
        '__metadata__': {},
        'conv.weight': {
          dtype: 'F32',
          shape: [64, 32, 3, 3],
          data_offsets: [0, 73728],
        },
      };
      const headerJson = JSON.stringify(header);
      const headerBytes = new TextEncoder().encode(headerJson);
      const headerSize = headerBytes.length;

      const buffer = new ArrayBuffer(8 + headerSize + 73728);
      const view = new DataView(buffer);
      const bytes = new Uint8Array(buffer);

      view.setUint32(0, headerSize, true);
      bytes.set(headerBytes, 8);

      const result = parseSafetensorsHeader(buffer);

      expect(result.tensors[0].shape).toEqual([64, 32, 3, 3]);
      expect(result.tensors[0].size).toBe(73728);
    });
  });
});

describe('formats/safetensors with fixture file', () => {
  const fixturePath = join(FIXTURES_DIR, 'sample.safetensors');

  describe('parseSafetensorsFile', () => {
    it('parses fixture file correctly', async () => {
      const result = await parseSafetensorsFile(fixturePath);

      expect(result.tensors.length).toBe(6);
      expect(result.filePath).toBe(fixturePath);
      expect(result.fileSize).toBeGreaterThan(0);
    });

    it('extracts metadata', async () => {
      const result = await parseSafetensorsFile(fixturePath);

      expect(result.metadata.format).toBe('pt');
      expect(result.metadata.model_type).toBe('test');
      expect(result.metadata.version).toBe('1.0');
    });

    it('parses all tensor shapes and dtypes', async () => {
      const result = await parseSafetensorsFile(fixturePath);

      const embedTensor = result.tensors.find(t => t.name === 'model.embed_tokens.weight');
      expect(embedTensor).toBeDefined();
      expect(embedTensor.dtype).toBe('F16');
      expect(embedTensor.shape).toEqual([100, 64]);

      const qTensor = result.tensors.find(t => t.name === 'model.layers.0.self_attn.q_proj.weight');
      expect(qTensor).toBeDefined();
      expect(qTensor.dtype).toBe('F32');
      expect(qTensor.shape).toEqual([64, 64]);
    });

    it('calculates correct tensor sizes', async () => {
      const result = await parseSafetensorsFile(fixturePath);

      const embedTensor = result.tensors.find(t => t.name === 'model.embed_tokens.weight');
      expect(embedTensor.size).toBe(12800);

      const qTensor = result.tensors.find(t => t.name === 'model.layers.0.self_attn.q_proj.weight');
      expect(qTensor.size).toBe(16384);
    });

    it('stores file path in tensors', async () => {
      const result = await parseSafetensorsFile(fixturePath);

      expect(result.tensors[0].filePath).toBe(fixturePath);
    });
  });

  describe('getTensor', () => {
    it('returns tensor by name', async () => {
      const parsed = await parseSafetensorsFile(fixturePath);
      const tensor = getTensor(parsed, 'model.embed_tokens.weight');

      expect(tensor).not.toBeNull();
      expect(tensor.name).toBe('model.embed_tokens.weight');
    });

    it('returns null for unknown tensor', async () => {
      const parsed = await parseSafetensorsFile(fixturePath);
      const tensor = getTensor(parsed, 'nonexistent');

      expect(tensor).toBeNull();
    });
  });

  describe('getTensors', () => {
    it('returns tensors matching pattern', async () => {
      const parsed = await parseSafetensorsFile(fixturePath);
      const tensors = getTensors(parsed, /layers\.0/);

      expect(tensors.length).toBe(3);
      expect(tensors.every(t => t.name.includes('layers.0'))).toBe(true);
    });

    it('returns empty array for no matches', async () => {
      const parsed = await parseSafetensorsFile(fixturePath);
      const tensors = getTensors(parsed, /layers\.99/);

      expect(tensors.length).toBe(0);
    });
  });

  describe('groupTensorsByLayer', () => {
    it('groups fixture tensors by layer', async () => {
      const parsed = await parseSafetensorsFile(fixturePath);
      const layers = groupTensorsByLayer(parsed);

      expect(layers.has(0)).toBe(true);
      expect(layers.has(1)).toBe(true);
      expect(layers.get(0).length).toBe(3);
      expect(layers.get(1).length).toBe(1);
    });
  });

  describe('parseSafetensorsHeader from buffer', () => {
    it('parses fixture loaded as buffer', async () => {
      const fileBuffer = await readFile(fixturePath);
      const arrayBuffer = fileBuffer.buffer.slice(
        fileBuffer.byteOffset,
        fileBuffer.byteOffset + fileBuffer.byteLength
      );

      const result = parseSafetensorsHeader(arrayBuffer);

      expect(result.tensors.length).toBe(6);
      expect(result.metadata.format).toBe('pt');
    });
  });
});
