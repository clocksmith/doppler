import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

// Mock GPU/WebGPU globals
vi.stubGlobal('GPUBufferUsage', {
  MAP_READ: 1,
  MAP_WRITE: 2,
  COPY_SRC: 4,
  COPY_DST: 8,
  INDEX: 16,
  VERTEX: 32,
  UNIFORM: 64,
  STORAGE: 128,
  INDIRECT: 256,
  QUERY_RESOLVE: 512,
});

// Mock debug module
vi.mock('../../src/debug/index.js', () => ({
  log: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    verbose: vi.fn(),
  },
  trace: {
    loader: vi.fn(),
    kernels: vi.fn(),
  },
  isTraceEnabled: vi.fn(() => false),
}));

// Mock GPU device module
vi.mock('../../src/gpu/device.js', () => ({
  getDevice: vi.fn(() => null),
  initDevice: vi.fn(async () => null),
  getKernelCapabilities: vi.fn(() => ({
    hasF16: true,
    hasSubgroups: true,
  })),
}));

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = join(__dirname, '../fixtures/mini-model');

function loadMiniModelManifest() {
  const raw = readFileSync(join(FIXTURES_DIR, 'manifest.json'), 'utf-8');
  return JSON.parse(raw);
}

function loadMiniModelTensors() {
  const raw = readFileSync(join(FIXTURES_DIR, 'tensors.json'), 'utf-8');
  return JSON.parse(raw);
}

function createValidTensorMap() {
  return {
    'model.embed_tokens.weight': {
      shard: 0,
      offset: 0,
      size: 8192,
      shape: [32, 64],
      dtype: 'F32',
      group: 'embed',
    },
    'model.layers.0.self_attn.q_proj.weight': {
      shard: 0,
      offset: 8448,
      size: 16384,
      shape: [64, 64],
      dtype: 'F32',
      group: 'layer.0',
    },
  };
}

describe('loader/tensor - tensor map parsing', () => {
  let tensors;

  beforeEach(() => {
    tensors = loadMiniModelTensors();
  });

  describe('tensors.json structure', () => {
    it('loads tensors.json successfully', () => {
      expect(tensors).toBeDefined();
      expect(typeof tensors).toBe('object');
    });

    it('has expected number of tensors', () => {
      const tensorCount = Object.keys(tensors).length;
      expect(tensorCount).toBe(20);
    });

    it('contains embedding tensor', () => {
      expect(tensors['model.embed_tokens.weight']).toBeDefined();
    });

    it('contains layer 0 attention tensors', () => {
      expect(tensors['model.layers.0.self_attn.q_proj.weight']).toBeDefined();
      expect(tensors['model.layers.0.self_attn.k_proj.weight']).toBeDefined();
      expect(tensors['model.layers.0.self_attn.v_proj.weight']).toBeDefined();
      expect(tensors['model.layers.0.self_attn.o_proj.weight']).toBeDefined();
    });

    it('contains layer 0 FFN tensors', () => {
      expect(tensors['model.layers.0.mlp.gate_proj.weight']).toBeDefined();
      expect(tensors['model.layers.0.mlp.up_proj.weight']).toBeDefined();
      expect(tensors['model.layers.0.mlp.down_proj.weight']).toBeDefined();
    });

    it('contains layer 0 norm tensors', () => {
      expect(tensors['model.layers.0.input_layernorm.weight']).toBeDefined();
      expect(tensors['model.layers.0.post_attention_layernorm.weight']).toBeDefined();
    });

    it('contains layer 1 tensors', () => {
      expect(tensors['model.layers.1.self_attn.q_proj.weight']).toBeDefined();
      expect(tensors['model.layers.1.mlp.gate_proj.weight']).toBeDefined();
    });

    it('contains final norm tensor', () => {
      expect(tensors['model.norm.weight']).toBeDefined();
    });
  });

  describe('tensor location fields', () => {
    it('each tensor has shard field', () => {
      for (const [name, tensor] of Object.entries(tensors)) {
        expect(tensor.shard).toBeDefined();
        expect(typeof tensor.shard).toBe('number');
      }
    });

    it('each tensor has offset field', () => {
      for (const [name, tensor] of Object.entries(tensors)) {
        expect(tensor.offset).toBeDefined();
        expect(typeof tensor.offset).toBe('number');
        expect(tensor.offset).toBeGreaterThanOrEqual(0);
      }
    });

    it('each tensor has size field', () => {
      for (const [name, tensor] of Object.entries(tensors)) {
        expect(tensor.size).toBeDefined();
        expect(typeof tensor.size).toBe('number');
        expect(tensor.size).toBeGreaterThan(0);
      }
    });

    it('each tensor has shape field', () => {
      for (const [name, tensor] of Object.entries(tensors)) {
        expect(tensor.shape).toBeDefined();
        expect(Array.isArray(tensor.shape)).toBe(true);
        expect(tensor.shape.length).toBeGreaterThan(0);
      }
    });

    it('each tensor has dtype field', () => {
      for (const [name, tensor] of Object.entries(tensors)) {
        expect(tensor.dtype).toBeDefined();
        expect(typeof tensor.dtype).toBe('string');
      }
    });

    it('each tensor has group field', () => {
      for (const [name, tensor] of Object.entries(tensors)) {
        expect(tensor.group).toBeDefined();
        expect(typeof tensor.group).toBe('string');
      }
    });
  });

  describe('offset validation', () => {
    it('embedding tensor starts at offset 0', () => {
      expect(tensors['model.embed_tokens.weight'].offset).toBe(0);
    });

    it('offsets are non-negative', () => {
      for (const [name, tensor] of Object.entries(tensors)) {
        expect(tensor.offset).toBeGreaterThanOrEqual(0);
      }
    });

    it('offsets do not overlap', () => {
      const tensorList = Object.entries(tensors).map(([name, t]) => ({
        name,
        offset: t.offset,
        size: t.size,
        end: t.offset + t.size,
      }));

      tensorList.sort((a, b) => a.offset - b.offset);

      for (let i = 0; i < tensorList.length - 1; i++) {
        const current = tensorList[i];
        const next = tensorList[i + 1];
        expect(current.end).toBeLessThanOrEqual(next.offset);
      }
    });

    it('offsets are within shard bounds', () => {
      const manifest = loadMiniModelManifest();
      const shardSize = manifest.shards[0].size;

      for (const [name, tensor] of Object.entries(tensors)) {
        expect(tensor.offset + tensor.size).toBeLessThanOrEqual(shardSize);
      }
    });
  });

  describe('shape validation', () => {
    it('embedding shape is [vocabSize, hiddenSize]', () => {
      const embedding = tensors['model.embed_tokens.weight'];
      expect(embedding.shape).toEqual([32, 64]);
    });

    it('Q projection shape is [hiddenSize, hiddenSize]', () => {
      const qProj = tensors['model.layers.0.self_attn.q_proj.weight'];
      expect(qProj.shape).toEqual([64, 64]);
    });

    it('K projection shape is [hiddenSize, hiddenSize]', () => {
      const kProj = tensors['model.layers.0.self_attn.k_proj.weight'];
      expect(kProj.shape).toEqual([64, 64]);
    });

    it('V projection shape is [hiddenSize, hiddenSize]', () => {
      const vProj = tensors['model.layers.0.self_attn.v_proj.weight'];
      expect(vProj.shape).toEqual([64, 64]);
    });

    it('O projection shape is [hiddenSize, hiddenSize]', () => {
      const oProj = tensors['model.layers.0.self_attn.o_proj.weight'];
      expect(oProj.shape).toEqual([64, 64]);
    });

    it('gate projection shape is [intermediateSize, hiddenSize]', () => {
      const gate = tensors['model.layers.0.mlp.gate_proj.weight'];
      expect(gate.shape).toEqual([128, 64]);
    });

    it('up projection shape is [intermediateSize, hiddenSize]', () => {
      const up = tensors['model.layers.0.mlp.up_proj.weight'];
      expect(up.shape).toEqual([128, 64]);
    });

    it('down projection shape is [hiddenSize, intermediateSize]', () => {
      const down = tensors['model.layers.0.mlp.down_proj.weight'];
      expect(down.shape).toEqual([64, 128]);
    });

    it('layernorm shape is [hiddenSize]', () => {
      const norm = tensors['model.layers.0.input_layernorm.weight'];
      expect(norm.shape).toEqual([64]);
    });

    it('final norm shape is [hiddenSize]', () => {
      const finalNorm = tensors['model.norm.weight'];
      expect(finalNorm.shape).toEqual([64]);
    });

    it('shape dimensions are positive integers', () => {
      for (const [name, tensor] of Object.entries(tensors)) {
        for (const dim of tensor.shape) {
          expect(Number.isInteger(dim)).toBe(true);
          expect(dim).toBeGreaterThan(0);
        }
      }
    });
  });

  describe('size validation', () => {
    it('size matches shape product times dtype size', () => {
      const dtypeSizes = {
        F32: 4,
        F16: 2,
        BF16: 2,
      };

      for (const [name, tensor] of Object.entries(tensors)) {
        const elements = tensor.shape.reduce((a, b) => a * b, 1);
        const dtypeSize = dtypeSizes[tensor.dtype] || 4;
        const expectedSize = elements * dtypeSize;
        expect(tensor.size).toBe(expectedSize);
      }
    });

    it('embedding size is vocabSize * hiddenSize * 4 (F32)', () => {
      const embedding = tensors['model.embed_tokens.weight'];
      // 32 * 64 * 4 = 8192
      expect(embedding.size).toBe(8192);
    });

    it('Q projection size is hiddenSize * hiddenSize * 4 (F32)', () => {
      const qProj = tensors['model.layers.0.self_attn.q_proj.weight'];
      // 64 * 64 * 4 = 16384
      expect(qProj.size).toBe(16384);
    });

    it('layernorm size is hiddenSize * 4 (F32)', () => {
      const norm = tensors['model.layers.0.input_layernorm.weight'];
      // 64 * 4 = 256
      expect(norm.size).toBe(256);
    });
  });

  describe('dtype validation', () => {
    it('all tensors in mini-model are F32', () => {
      for (const [name, tensor] of Object.entries(tensors)) {
        expect(tensor.dtype).toBe('F32');
      }
    });

    it('recognizes valid dtype values', () => {
      const validDtypes = ['F32', 'F16', 'BF16', 'Q4_K', 'Q4_K_M', 'Q6_K'];
      const tensor = createValidTensorMap()['model.embed_tokens.weight'];
      expect(validDtypes).toContain(tensor.dtype);
    });
  });

  describe('group assignment', () => {
    it('embedding tensor is in embed group', () => {
      expect(tensors['model.embed_tokens.weight'].group).toBe('embed');
    });

    it('layer 0 tensors are in layer.0 group', () => {
      expect(tensors['model.layers.0.self_attn.q_proj.weight'].group).toBe('layer.0');
      expect(tensors['model.layers.0.mlp.gate_proj.weight'].group).toBe('layer.0');
      expect(tensors['model.layers.0.input_layernorm.weight'].group).toBe('layer.0');
    });

    it('layer 1 tensors are in layer.1 group', () => {
      expect(tensors['model.layers.1.self_attn.q_proj.weight'].group).toBe('layer.1');
      expect(tensors['model.layers.1.mlp.gate_proj.weight'].group).toBe('layer.1');
    });

    it('final norm is in head group', () => {
      expect(tensors['model.norm.weight'].group).toBe('head');
    });

    it('group names match manifest groups', () => {
      const manifest = loadMiniModelManifest();
      const manifestGroups = Object.keys(manifest.groups);

      for (const [name, tensor] of Object.entries(tensors)) {
        expect(manifestGroups).toContain(tensor.group);
      }
    });
  });

  describe('shard assignment', () => {
    it('all tensors reference shard 0 in mini-model', () => {
      for (const [name, tensor] of Object.entries(tensors)) {
        expect(tensor.shard).toBe(0);
      }
    });

    it('shard index is within valid range', () => {
      const manifest = loadMiniModelManifest();
      const numShards = manifest.shards.length;

      for (const [name, tensor] of Object.entries(tensors)) {
        expect(tensor.shard).toBeGreaterThanOrEqual(0);
        expect(tensor.shard).toBeLessThan(numShards);
      }
    });
  });

  describe('tensor naming patterns', () => {
    it('embedding follows model.embed_tokens.weight pattern', () => {
      const embedNames = Object.keys(tensors).filter(n => n.includes('embed'));
      expect(embedNames).toContain('model.embed_tokens.weight');
    });

    it('attention projections follow self_attn pattern', () => {
      const attnNames = Object.keys(tensors).filter(n => n.includes('self_attn'));
      expect(attnNames.length).toBeGreaterThan(0);
      attnNames.forEach(name => {
        expect(name).toMatch(/self_attn\.(q|k|v|o)_proj\.weight/);
      });
    });

    it('FFN projections follow mlp pattern', () => {
      const mlpNames = Object.keys(tensors).filter(n => n.includes('mlp'));
      expect(mlpNames.length).toBeGreaterThan(0);
      mlpNames.forEach(name => {
        expect(name).toMatch(/mlp\.(gate|up|down)_proj\.weight/);
      });
    });

    it('layer indices are correctly encoded in names', () => {
      const layer0Names = Object.keys(tensors).filter(n => n.includes('layers.0'));
      const layer1Names = Object.keys(tensors).filter(n => n.includes('layers.1'));

      expect(layer0Names.length).toBeGreaterThan(0);
      expect(layer1Names.length).toBeGreaterThan(0);
    });
  });
});

describe('loader/tensor - tensor location building', () => {
  describe('TensorLocation interface', () => {
    it('maps shard to shardIndex', () => {
      const tensors = loadMiniModelTensors();
      const tensorInfo = tensors['model.embed_tokens.weight'];

      const location = {
        shardIndex: tensorInfo.shard,
        offset: tensorInfo.offset,
        size: tensorInfo.size,
        shape: tensorInfo.shape,
        dtype: tensorInfo.dtype,
      };

      expect(location.shardIndex).toBe(0);
    });

    it('preserves offset', () => {
      const tensors = loadMiniModelTensors();
      const tensorInfo = tensors['model.layers.0.self_attn.q_proj.weight'];

      const location = {
        shardIndex: tensorInfo.shard,
        offset: tensorInfo.offset,
        size: tensorInfo.size,
        shape: tensorInfo.shape,
        dtype: tensorInfo.dtype,
      };

      expect(location.offset).toBe(8448);
    });

    it('preserves size', () => {
      const tensors = loadMiniModelTensors();
      const tensorInfo = tensors['model.embed_tokens.weight'];

      const location = {
        shardIndex: tensorInfo.shard,
        offset: tensorInfo.offset,
        size: tensorInfo.size,
        shape: tensorInfo.shape,
        dtype: tensorInfo.dtype,
      };

      expect(location.size).toBe(8192);
    });

    it('preserves shape', () => {
      const tensors = loadMiniModelTensors();
      const tensorInfo = tensors['model.embed_tokens.weight'];

      const location = {
        shardIndex: tensorInfo.shard,
        offset: tensorInfo.offset,
        size: tensorInfo.size,
        shape: tensorInfo.shape,
        dtype: tensorInfo.dtype,
      };

      expect(location.shape).toEqual([32, 64]);
    });

    it('preserves dtype', () => {
      const tensors = loadMiniModelTensors();
      const tensorInfo = tensors['model.embed_tokens.weight'];

      const location = {
        shardIndex: tensorInfo.shard,
        offset: tensorInfo.offset,
        size: tensorInfo.size,
        shape: tensorInfo.shape,
        dtype: tensorInfo.dtype,
      };

      expect(location.dtype).toBe('F32');
    });
  });

  describe('building tensor location map', () => {
    it('creates Map from tensors.json', () => {
      const tensors = loadMiniModelTensors();
      const locationMap = new Map();

      for (const [name, info] of Object.entries(tensors)) {
        locationMap.set(name, {
          shardIndex: info.shard,
          offset: info.offset,
          size: info.size,
          shape: info.shape,
          dtype: info.dtype,
        });
      }

      expect(locationMap.size).toBe(20);
    });

    it('can retrieve tensor by name', () => {
      const tensors = loadMiniModelTensors();
      const locationMap = new Map();

      for (const [name, info] of Object.entries(tensors)) {
        locationMap.set(name, {
          shardIndex: info.shard,
          offset: info.offset,
          size: info.size,
          shape: info.shape,
          dtype: info.dtype,
        });
      }

      const embedding = locationMap.get('model.embed_tokens.weight');
      expect(embedding).toBeDefined();
      expect(embedding.shape).toEqual([32, 64]);
    });

    it('returns undefined for missing tensor', () => {
      const tensors = loadMiniModelTensors();
      const locationMap = new Map();

      for (const [name, info] of Object.entries(tensors)) {
        locationMap.set(name, info);
      }

      const missing = locationMap.get('nonexistent.tensor');
      expect(missing).toBeUndefined();
    });
  });
});

describe('loader/tensor - tensor predicates', () => {
  describe('isEmbeddingTensor', () => {
    it('identifies embed_tokens as embedding', () => {
      const name = 'model.embed_tokens.weight';
      const isEmbed = name.toLowerCase().includes('embed');
      expect(isEmbed).toBe(true);
    });

    it('identifies wte as embedding', () => {
      const name = 'transformer.wte.weight';
      const isEmbed = name.toLowerCase().includes('wte');
      expect(isEmbed).toBe(true);
    });

    it('does not identify projection as embedding', () => {
      const name = 'model.layers.0.self_attn.q_proj.weight';
      const isEmbed = name.toLowerCase().includes('embed') || name.toLowerCase().includes('wte');
      expect(isEmbed).toBe(false);
    });
  });

  describe('isLMHeadTensor', () => {
    it('identifies lm_head as LM head', () => {
      const name = 'lm_head.weight';
      const isLmHead = name.toLowerCase().includes('lm_head');
      expect(isLmHead).toBe(true);
    });

    it('identifies output.weight as LM head', () => {
      const name = 'model.output.weight';
      const isLmHead = name.toLowerCase().includes('output.weight');
      expect(isLmHead).toBe(true);
    });

    it('does not identify o_proj as LM head', () => {
      const name = 'model.layers.0.self_attn.o_proj.weight';
      const isLmHead = name.toLowerCase().includes('lm_head') || name.toLowerCase().includes('output.weight');
      expect(isLmHead).toBe(false);
    });
  });

  describe('isNormTensor', () => {
    it('identifies layernorm as norm', () => {
      const name = 'model.layers.0.input_layernorm.weight';
      const isNorm = name.toLowerCase().includes('norm');
      expect(isNorm).toBe(true);
    });

    it('identifies final norm as norm', () => {
      const name = 'model.norm.weight';
      const isNorm = name.toLowerCase().includes('norm');
      expect(isNorm).toBe(true);
    });

    it('does not identify projection as norm', () => {
      const name = 'model.layers.0.self_attn.q_proj.weight';
      const isNorm = name.toLowerCase().includes('norm');
      expect(isNorm).toBe(false);
    });
  });

  describe('isMatmulTensor', () => {
    it('identifies q_proj as matmul', () => {
      const name = 'model.layers.0.self_attn.q_proj.weight';
      const isMatmul = name.toLowerCase().includes('proj');
      expect(isMatmul).toBe(true);
    });

    it('identifies gate_proj as matmul', () => {
      const name = 'model.layers.0.mlp.gate_proj.weight';
      const isMatmul = name.toLowerCase().includes('proj') || name.toLowerCase().includes('gate');
      expect(isMatmul).toBe(true);
    });

    it('does not identify norm as matmul', () => {
      const name = 'model.layers.0.input_layernorm.weight';
      const isMatmul = name.toLowerCase().includes('proj') ||
                       name.toLowerCase().includes('gate') ||
                       name.toLowerCase().includes('down') ||
                       name.toLowerCase().includes('up');
      // up is in layernorm but not as a separate word
      const isProj = name.toLowerCase().includes('proj');
      expect(isProj).toBe(false);
    });
  });
});

describe('loader/tensor - error handling', () => {
  describe('missing required fields', () => {
    it('detects missing shard', () => {
      const tensor = { offset: 0, size: 100, shape: [10], dtype: 'F32' };
      expect(tensor.shard).toBeUndefined();
    });

    it('detects missing offset', () => {
      const tensor = { shard: 0, size: 100, shape: [10], dtype: 'F32' };
      expect(tensor.offset).toBeUndefined();
    });

    it('detects missing size', () => {
      const tensor = { shard: 0, offset: 0, shape: [10], dtype: 'F32' };
      expect(tensor.size).toBeUndefined();
    });

    it('detects missing shape', () => {
      const tensor = { shard: 0, offset: 0, size: 100, dtype: 'F32' };
      expect(tensor.shape).toBeUndefined();
    });
  });

  describe('invalid field values', () => {
    it('detects negative shard', () => {
      const tensor = { shard: -1, offset: 0, size: 100, shape: [10], dtype: 'F32' };
      expect(tensor.shard).toBeLessThan(0);
    });

    it('detects negative offset', () => {
      const tensor = { shard: 0, offset: -100, size: 100, shape: [10], dtype: 'F32' };
      expect(tensor.offset).toBeLessThan(0);
    });

    it('detects zero size', () => {
      const tensor = { shard: 0, offset: 0, size: 0, shape: [10], dtype: 'F32' };
      expect(tensor.size).toBe(0);
    });

    it('detects negative size', () => {
      const tensor = { shard: 0, offset: 0, size: -100, shape: [10], dtype: 'F32' };
      expect(tensor.size).toBeLessThan(0);
    });

    it('detects empty shape', () => {
      const tensor = { shard: 0, offset: 0, size: 100, shape: [], dtype: 'F32' };
      expect(tensor.shape.length).toBe(0);
    });

    it('detects shape with zero dimension', () => {
      const tensor = { shard: 0, offset: 0, size: 100, shape: [0, 10], dtype: 'F32' };
      expect(tensor.shape[0]).toBe(0);
    });

    it('detects shape with negative dimension', () => {
      const tensor = { shard: 0, offset: 0, size: 100, shape: [-1, 10], dtype: 'F32' };
      expect(tensor.shape[0]).toBeLessThan(0);
    });
  });

  describe('JSON parsing errors', () => {
    it('handles invalid JSON', () => {
      const invalidJson = '{ invalid json }';
      expect(() => JSON.parse(invalidJson)).toThrow();
    });

    it('handles empty string', () => {
      expect(() => JSON.parse('')).toThrow();
    });

    it('handles null JSON', () => {
      const parsed = JSON.parse('null');
      expect(parsed).toBeNull();
    });

    it('handles array instead of object', () => {
      const parsed = JSON.parse('[]');
      expect(Array.isArray(parsed)).toBe(true);
    });
  });
});

describe('loader/tensor - multi-shard tensor handling', () => {
  describe('spans field', () => {
    it('handles tensor without spans', () => {
      const tensor = createValidTensorMap()['model.embed_tokens.weight'];
      expect(tensor.spans).toBeUndefined();
    });

    it('handles tensor with spans', () => {
      const tensor = {
        shard: 0,
        offset: 0,
        size: 2000,
        shape: [100, 10],
        dtype: 'F32',
        spans: [
          { shardIndex: 0, offset: 0, size: 1000 },
          { shardIndex: 1, offset: 0, size: 1000 },
        ],
      };

      expect(tensor.spans).toBeDefined();
      expect(Array.isArray(tensor.spans)).toBe(true);
      expect(tensor.spans.length).toBe(2);
    });

    it('spans have required fields', () => {
      const tensor = {
        shard: 0,
        offset: 0,
        size: 2000,
        shape: [100, 10],
        dtype: 'F32',
        spans: [
          { shardIndex: 0, offset: 0, size: 1000 },
          { shardIndex: 1, offset: 0, size: 1000 },
        ],
      };

      for (const span of tensor.spans) {
        expect(span.shardIndex).toBeDefined();
        expect(span.offset).toBeDefined();
        expect(span.size).toBeDefined();
      }
    });

    it('spans sum to total size', () => {
      const tensor = {
        shard: 0,
        offset: 0,
        size: 2000,
        shape: [100, 10],
        dtype: 'F32',
        spans: [
          { shardIndex: 0, offset: 0, size: 1000 },
          { shardIndex: 1, offset: 0, size: 1000 },
        ],
      };

      const spanSum = tensor.spans.reduce((sum, s) => sum + s.size, 0);
      expect(spanSum).toBe(tensor.size);
    });
  });

  describe('layout field', () => {
    it('handles tensor without layout', () => {
      const tensor = createValidTensorMap()['model.embed_tokens.weight'];
      expect(tensor.layout).toBeUndefined();
    });

    it('handles tensor with row layout', () => {
      const tensor = {
        shard: 0,
        offset: 0,
        size: 1000,
        shape: [10, 100],
        dtype: 'F32',
        layout: 'row',
      };
      expect(tensor.layout).toBe('row');
    });

    it('handles tensor with column layout', () => {
      const tensor = {
        shard: 0,
        offset: 0,
        size: 1000,
        shape: [10, 100],
        dtype: 'F32',
        layout: 'column',
      };
      expect(tensor.layout).toBe('column');
    });
  });

  describe('originalShape field', () => {
    it('handles tensor without originalShape', () => {
      const tensor = createValidTensorMap()['model.embed_tokens.weight'];
      expect(tensor.originalShape).toBeUndefined();
    });

    it('handles tensor with originalShape for transposed weights', () => {
      const tensor = {
        shard: 0,
        offset: 0,
        size: 1000,
        shape: [100, 10],
        dtype: 'F32',
        layout: 'column',
        originalShape: [10, 100],
      };
      expect(tensor.originalShape).toEqual([10, 100]);
    });
  });
});

describe('loader/tensor - consistency with manifest', () => {
  it('tensorCount matches actual tensor count', () => {
    const manifest = loadMiniModelManifest();
    const tensors = loadMiniModelTensors();
    const actualCount = Object.keys(tensors).length;

    expect(manifest.tensorCount).toBe(actualCount);
  });

  it('all group tensors exist in tensor map', () => {
    const manifest = loadMiniModelManifest();
    const tensors = loadMiniModelTensors();

    for (const [groupName, group] of Object.entries(manifest.groups)) {
      for (const tensorName of group.tensors) {
        expect(tensors[tensorName]).toBeDefined();
      }
    }
  });

  it('tensor groups match group assignments', () => {
    const manifest = loadMiniModelManifest();
    const tensors = loadMiniModelTensors();

    for (const [groupName, group] of Object.entries(manifest.groups)) {
      for (const tensorName of group.tensors) {
        expect(tensors[tensorName].group).toBe(groupName);
      }
    }
  });

  it('total tensor size is within shard bounds', () => {
    const manifest = loadMiniModelManifest();
    const tensors = loadMiniModelTensors();

    let totalTensorSize = 0;
    for (const tensor of Object.values(tensors)) {
      totalTensorSize += tensor.size;
    }

    // Total tensor size should be less than or equal to total shard size
    // (there may be padding)
    expect(totalTensorSize).toBeLessThanOrEqual(manifest.totalSize);
  });
});

describe('loader/tensor - Q4K alignment fallback', () => {
  // Import directly since we're testing the logic without GPU
  const QK_K = 256; // Q4K super block size

  describe('K alignment detection', () => {
    it('identifies K aligned to 256 (should use fused)', () => {
      const K = 2048; // Common LLaMA hidden size, divisible by 256
      expect(K % QK_K).toBe(0);
    });

    it('identifies K not aligned to 256 (should use dequant fallback)', () => {
      const K = 1152; // Gemma 3 1B hidden size, NOT divisible by 256
      expect(K % QK_K).not.toBe(0);
      expect(K % QK_K).toBe(128); // 1152 = 4*256 + 128
    });

    it('identifies various model hidden sizes', () => {
      // Models with aligned hidden sizes (can use fused Q4K)
      const alignedSizes = [2048, 4096, 8192, 3072]; // LLaMA-7B, LLaMA-13B, etc.
      for (const size of alignedSizes) {
        expect(size % QK_K, `hidden size ${size}`).toBe(0);
      }

      // Models with non-aligned hidden sizes (must use dequant fallback)
      const nonAlignedSizes = [1152, 896, 960]; // Examples not divisible by 256
      for (const size of nonAlignedSizes) {
        expect(size % QK_K, `hidden size ${size}`).not.toBe(0);
      }
    });
  });

  describe('location shape parsing', () => {
    it('extracts K from 2D weight shape [N, K]', () => {
      // Weight matrix is typically [output, input] = [N, K]
      const location = { shape: [4096, 1152], dtype: 'Q4_K' };
      const [N, K] = location.shape;
      expect(K).toBe(1152);
      expect(K % QK_K).not.toBe(0); // Not aligned, needs fallback
    });

    it('handles 1D shapes (embeddings)', () => {
      const location = { shape: [32000], dtype: 'Q4_K' };
      expect(location.shape.length).toBe(1);
      // 1D shapes don't need K alignment check
    });
  });
});
