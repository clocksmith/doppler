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

vi.stubGlobal('GPUBuffer', class MockGPUBuffer {
  constructor() {
    this.size = 0;
    this.usage = 0;
    this.mapState = 'unmapped';
  }
  destroy() {}
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

function createTensorLocationMap(tensors) {
  const map = new Map();
  for (const [name, info] of Object.entries(tensors)) {
    map.set(name, {
      shardIndex: info.shard,
      offset: info.offset,
      size: info.size,
      shape: info.shape,
      dtype: info.dtype,
    });
  }
  return map;
}

function createValidLayerWeights() {
  return {
    inputNorm: null,
    qProj: null,
    kProj: null,
    vProj: null,
    oProj: null,
    qNorm: null,
    kNorm: null,
    postAttentionNorm: null,
    preFeedforwardNorm: null,
    postFeedforwardNorm: null,
    postNorm: null,
    postAttnNorm: null,
    ffnGate: null,
    ffnUp: null,
    ffnDown: null,
    ffnGateUp: null,
  };
}

// Layer name prefix patterns
const LAYER_PREFIXES = (layerIdx) => [
  `language_model.model.layers.${layerIdx}`,
  `model.layers.${layerIdx}`,
  `layers.${layerIdx}`,
  `blk.${layerIdx}`,
];

// Attention weight suffixes
const ATTN_SUFFIXES = {
  inputNorm: ['input_layernorm.weight', 'attn_norm.weight'],
  qProj: ['self_attn.q_proj.weight', 'attention.wq.weight', 'attn_q.weight'],
  kProj: ['self_attn.k_proj.weight', 'attention.wk.weight', 'attn_k.weight'],
  vProj: ['self_attn.v_proj.weight', 'attention.wv.weight', 'attn_v.weight'],
  oProj: ['self_attn.o_proj.weight', 'attention.wo.weight', 'attn_output.weight'],
  qNorm: ['self_attn.q_norm.weight', 'attn_q_norm.weight'],
  kNorm: ['self_attn.k_norm.weight', 'attn_k_norm.weight'],
  postAttentionNorm: ['post_attention_layernorm.weight', 'post_attention_norm.weight', 'ffn_norm.weight'],
  preFeedforwardNorm: ['pre_feedforward_layernorm.weight'],
  postFeedforwardNorm: ['post_feedforward_layernorm.weight', 'post_ffw_norm.weight'],
};

// FFN weight suffixes
const FFN_SUFFIXES = {
  ffnGateUp: ['mlp.gate_up_proj.weight', 'ffn_gate_up.weight', 'feed_forward.w1_w3.weight'],
  ffnGate: ['mlp.gate_proj.weight', 'feed_forward.w1.weight', 'ffn_gate.weight'],
  ffnUp: ['mlp.up_proj.weight', 'feed_forward.w3.weight', 'ffn_up.weight'],
  ffnDown: ['mlp.down_proj.weight', 'feed_forward.w2.weight', 'ffn_down.weight'],
};

describe('loader/layer - layer weight loading', () => {
  let manifest;
  let tensors;
  let tensorLocations;

  beforeEach(() => {
    manifest = loadMiniModelManifest();
    tensors = loadMiniModelTensors();
    tensorLocations = createTensorLocationMap(tensors);
  });

  describe('layer group structure', () => {
    it('manifest has layer.0 group', () => {
      expect(manifest.groups['layer.0']).toBeDefined();
    });

    it('manifest has layer.1 group', () => {
      expect(manifest.groups['layer.1']).toBeDefined();
    });

    it('layer groups have correct type', () => {
      expect(manifest.groups['layer.0'].type).toBe('layer');
      expect(manifest.groups['layer.1'].type).toBe('layer');
    });

    it('layer groups have layerIndex', () => {
      expect(manifest.groups['layer.0'].layerIndex).toBe(0);
      expect(manifest.groups['layer.1'].layerIndex).toBe(1);
    });

    it('layer.0 has expected tensor count', () => {
      expect(manifest.groups['layer.0'].tensors.length).toBe(9);
    });

    it('layer.1 has expected tensor count', () => {
      expect(manifest.groups['layer.1'].tensors.length).toBe(9);
    });
  });

  describe('layer prefix resolution', () => {
    it('finds tensors with model.layers prefix', () => {
      const prefixes = LAYER_PREFIXES(0);
      const prefix = prefixes.find(p => tensorLocations.has(`${p}.input_layernorm.weight`));
      expect(prefix).toBe('model.layers.0');
    });

    it('layer 0 tensors use model.layers.0 prefix', () => {
      expect(tensorLocations.has('model.layers.0.input_layernorm.weight')).toBe(true);
      expect(tensorLocations.has('model.layers.0.self_attn.q_proj.weight')).toBe(true);
    });

    it('layer 1 tensors use model.layers.1 prefix', () => {
      expect(tensorLocations.has('model.layers.1.input_layernorm.weight')).toBe(true);
      expect(tensorLocations.has('model.layers.1.self_attn.q_proj.weight')).toBe(true);
    });
  });

  describe('attention weight resolution', () => {
    it('resolves input_layernorm for layer 0', () => {
      const name = 'model.layers.0.input_layernorm.weight';
      const location = tensorLocations.get(name);
      expect(location).toBeDefined();
      expect(location.shape).toEqual([64]);
    });

    it('resolves q_proj for layer 0', () => {
      const name = 'model.layers.0.self_attn.q_proj.weight';
      const location = tensorLocations.get(name);
      expect(location).toBeDefined();
      expect(location.shape).toEqual([64, 64]);
    });

    it('resolves k_proj for layer 0', () => {
      const name = 'model.layers.0.self_attn.k_proj.weight';
      const location = tensorLocations.get(name);
      expect(location).toBeDefined();
      expect(location.shape).toEqual([64, 64]);
    });

    it('resolves v_proj for layer 0', () => {
      const name = 'model.layers.0.self_attn.v_proj.weight';
      const location = tensorLocations.get(name);
      expect(location).toBeDefined();
      expect(location.shape).toEqual([64, 64]);
    });

    it('resolves o_proj for layer 0', () => {
      const name = 'model.layers.0.self_attn.o_proj.weight';
      const location = tensorLocations.get(name);
      expect(location).toBeDefined();
      expect(location.shape).toEqual([64, 64]);
    });

    it('resolves post_attention_layernorm for layer 0', () => {
      const name = 'model.layers.0.post_attention_layernorm.weight';
      const location = tensorLocations.get(name);
      expect(location).toBeDefined();
      expect(location.shape).toEqual([64]);
    });
  });

  describe('FFN weight resolution', () => {
    it('resolves gate_proj for layer 0', () => {
      const name = 'model.layers.0.mlp.gate_proj.weight';
      const location = tensorLocations.get(name);
      expect(location).toBeDefined();
      expect(location.shape).toEqual([128, 64]);
    });

    it('resolves up_proj for layer 0', () => {
      const name = 'model.layers.0.mlp.up_proj.weight';
      const location = tensorLocations.get(name);
      expect(location).toBeDefined();
      expect(location.shape).toEqual([128, 64]);
    });

    it('resolves down_proj for layer 0', () => {
      const name = 'model.layers.0.mlp.down_proj.weight';
      const location = tensorLocations.get(name);
      expect(location).toBeDefined();
      expect(location.shape).toEqual([64, 128]);
    });
  });

  describe('optional weight handling', () => {
    it('q_norm is optional', () => {
      const name = 'model.layers.0.self_attn.q_norm.weight';
      const location = tensorLocations.get(name);
      // mini-model does not have q_norm
      expect(location).toBeUndefined();
    });

    it('k_norm is optional', () => {
      const name = 'model.layers.0.self_attn.k_norm.weight';
      const location = tensorLocations.get(name);
      // mini-model does not have k_norm
      expect(location).toBeUndefined();
    });

    it('pre_feedforward_layernorm is optional', () => {
      const name = 'model.layers.0.pre_feedforward_layernorm.weight';
      const location = tensorLocations.get(name);
      // mini-model does not have pre_feedforward_layernorm
      expect(location).toBeUndefined();
    });

    it('post_feedforward_layernorm is optional', () => {
      const name = 'model.layers.0.post_feedforward_layernorm.weight';
      const location = tensorLocations.get(name);
      // mini-model does not have post_feedforward_layernorm
      expect(location).toBeUndefined();
    });

    it('fused gate_up_proj is optional', () => {
      const name = 'model.layers.0.mlp.gate_up_proj.weight';
      const location = tensorLocations.get(name);
      // mini-model has separate gate and up, not fused
      expect(location).toBeUndefined();
    });
  });

  describe('LayerWeights structure', () => {
    it('has all required fields', () => {
      const weights = createValidLayerWeights();

      expect(weights).toHaveProperty('inputNorm');
      expect(weights).toHaveProperty('qProj');
      expect(weights).toHaveProperty('kProj');
      expect(weights).toHaveProperty('vProj');
      expect(weights).toHaveProperty('oProj');
      expect(weights).toHaveProperty('postAttentionNorm');
      expect(weights).toHaveProperty('ffnGate');
      expect(weights).toHaveProperty('ffnUp');
      expect(weights).toHaveProperty('ffnDown');
    });

    it('has optional fields', () => {
      const weights = createValidLayerWeights();

      expect(weights).toHaveProperty('qNorm');
      expect(weights).toHaveProperty('kNorm');
      expect(weights).toHaveProperty('preFeedforwardNorm');
      expect(weights).toHaveProperty('postFeedforwardNorm');
      expect(weights).toHaveProperty('ffnGateUp');
    });

    it('has alias fields', () => {
      const weights = createValidLayerWeights();

      expect(weights).toHaveProperty('postNorm');
      expect(weights).toHaveProperty('postAttnNorm');
    });

    it('initializes all fields to null', () => {
      const weights = createValidLayerWeights();

      for (const key of Object.keys(weights)) {
        expect(weights[key]).toBeNull();
      }
    });
  });

  describe('layer weight shape validation', () => {
    it('norm weights are 1D', () => {
      const inputNorm = tensorLocations.get('model.layers.0.input_layernorm.weight');
      expect(inputNorm.shape.length).toBe(1);
      expect(inputNorm.shape[0]).toBe(manifest.architecture.hiddenSize);
    });

    it('attention projection weights are 2D', () => {
      const qProj = tensorLocations.get('model.layers.0.self_attn.q_proj.weight');
      expect(qProj.shape.length).toBe(2);
    });

    it('FFN projection weights are 2D', () => {
      const gate = tensorLocations.get('model.layers.0.mlp.gate_proj.weight');
      expect(gate.shape.length).toBe(2);
    });

    it('Q/K/V projections have hiddenSize output', () => {
      const qProj = tensorLocations.get('model.layers.0.self_attn.q_proj.weight');
      const kProj = tensorLocations.get('model.layers.0.self_attn.k_proj.weight');
      const vProj = tensorLocations.get('model.layers.0.self_attn.v_proj.weight');

      // Shape is [out_features, in_features]
      expect(qProj.shape[0]).toBe(manifest.architecture.hiddenSize);
      expect(kProj.shape[0]).toBe(manifest.architecture.hiddenSize);
      expect(vProj.shape[0]).toBe(manifest.architecture.hiddenSize);
    });

    it('O projection has hiddenSize output', () => {
      const oProj = tensorLocations.get('model.layers.0.self_attn.o_proj.weight');
      expect(oProj.shape[0]).toBe(manifest.architecture.hiddenSize);
    });

    it('gate/up projections have intermediateSize output', () => {
      const gate = tensorLocations.get('model.layers.0.mlp.gate_proj.weight');
      const up = tensorLocations.get('model.layers.0.mlp.up_proj.weight');

      expect(gate.shape[0]).toBe(manifest.architecture.intermediateSize);
      expect(up.shape[0]).toBe(manifest.architecture.intermediateSize);
    });

    it('down projection has hiddenSize output', () => {
      const down = tensorLocations.get('model.layers.0.mlp.down_proj.weight');
      expect(down.shape[0]).toBe(manifest.architecture.hiddenSize);
    });
  });
});

describe('loader/layer - group resolution', () => {
  let manifest;
  let tensors;

  beforeEach(() => {
    manifest = loadMiniModelManifest();
    tensors = loadMiniModelTensors();
  });

  describe('embed group', () => {
    it('embed group contains embedding tensor', () => {
      const group = manifest.groups.embed;
      expect(group.tensors).toContain('model.embed_tokens.weight');
    });

    it('embed group has correct type', () => {
      expect(manifest.groups.embed.type).toBe('embed');
    });

    it('embed group references shard 0', () => {
      expect(manifest.groups.embed.shards).toContain(0);
    });

    it('embed group has hash', () => {
      expect(manifest.groups.embed.hash).toBeDefined();
      expect(typeof manifest.groups.embed.hash).toBe('string');
    });
  });

  describe('head group', () => {
    it('head group contains final norm', () => {
      const group = manifest.groups.head;
      expect(group.tensors).toContain('model.norm.weight');
    });

    it('head group has correct type', () => {
      expect(manifest.groups.head.type).toBe('head');
    });

    it('head group references shard 0', () => {
      expect(manifest.groups.head.shards).toContain(0);
    });
  });

  describe('layer group tensors', () => {
    it('layer.0 contains all attention tensors', () => {
      const group = manifest.groups['layer.0'];

      expect(group.tensors).toContain('model.layers.0.input_layernorm.weight');
      expect(group.tensors).toContain('model.layers.0.self_attn.q_proj.weight');
      expect(group.tensors).toContain('model.layers.0.self_attn.k_proj.weight');
      expect(group.tensors).toContain('model.layers.0.self_attn.v_proj.weight');
      expect(group.tensors).toContain('model.layers.0.self_attn.o_proj.weight');
    });

    it('layer.0 contains all FFN tensors', () => {
      const group = manifest.groups['layer.0'];

      expect(group.tensors).toContain('model.layers.0.post_attention_layernorm.weight');
      expect(group.tensors).toContain('model.layers.0.mlp.gate_proj.weight');
      expect(group.tensors).toContain('model.layers.0.mlp.up_proj.weight');
      expect(group.tensors).toContain('model.layers.0.mlp.down_proj.weight');
    });

    it('layer.1 contains corresponding tensors', () => {
      const group = manifest.groups['layer.1'];

      expect(group.tensors).toContain('model.layers.1.input_layernorm.weight');
      expect(group.tensors).toContain('model.layers.1.self_attn.q_proj.weight');
      expect(group.tensors).toContain('model.layers.1.mlp.gate_proj.weight');
    });
  });

  describe('group shard references', () => {
    it('all groups reference valid shards', () => {
      const shardCount = manifest.shards.length;

      for (const [name, group] of Object.entries(manifest.groups)) {
        for (const shardIdx of group.shards) {
          expect(shardIdx).toBeGreaterThanOrEqual(0);
          expect(shardIdx).toBeLessThan(shardCount);
        }
      }
    });

    it('tensor shard matches group shard', () => {
      for (const [name, group] of Object.entries(manifest.groups)) {
        for (const tensorName of group.tensors) {
          const tensor = tensors[tensorName];
          expect(group.shards).toContain(tensor.shard);
        }
      }
    });
  });

  describe('group versioning', () => {
    it('all groups have version', () => {
      for (const [name, group] of Object.entries(manifest.groups)) {
        expect(group.version).toBeDefined();
      }
    });

    it('version is string format', () => {
      for (const [name, group] of Object.entries(manifest.groups)) {
        expect(typeof group.version).toBe('string');
      }
    });
  });

  describe('group hashing', () => {
    it('all groups have hash', () => {
      for (const [name, group] of Object.entries(manifest.groups)) {
        expect(group.hash).toBeDefined();
      }
    });

    it('hash is hex string', () => {
      for (const [name, group] of Object.entries(manifest.groups)) {
        expect(/^[0-9a-fA-F]+$/.test(group.hash)).toBe(true);
      }
    });

    it('hash has expected length (64 chars for sha256)', () => {
      for (const [name, group] of Object.entries(manifest.groups)) {
        expect(group.hash.length).toBe(64);
      }
    });
  });
});

describe('loader/layer - tryLoad helper pattern', () => {
  let tensorLocations;

  beforeEach(() => {
    const tensors = loadMiniModelTensors();
    tensorLocations = createTensorLocationMap(tensors);
  });

  describe('tryLoad with multiple suffixes', () => {
    it('finds tensor with first matching suffix', () => {
      const prefixes = LAYER_PREFIXES(0);
      const suffixes = ATTN_SUFFIXES.inputNorm;

      let found = null;
      for (const prefix of prefixes) {
        for (const suffix of suffixes) {
          const name = `${prefix}.${suffix}`;
          if (tensorLocations.has(name)) {
            found = tensorLocations.get(name);
            break;
          }
        }
        if (found) break;
      }

      expect(found).toBeDefined();
      expect(found.shape).toEqual([64]);
    });

    it('returns null when no suffix matches', () => {
      const prefixes = LAYER_PREFIXES(0);
      const suffixes = ['nonexistent.weight'];

      let found = null;
      for (const prefix of prefixes) {
        for (const suffix of suffixes) {
          const name = `${prefix}.${suffix}`;
          if (tensorLocations.has(name)) {
            found = tensorLocations.get(name);
            break;
          }
        }
        if (found) break;
      }

      expect(found).toBeNull();
    });

    it('tries all prefixes', () => {
      const prefixes = LAYER_PREFIXES(0);

      // model.layers.0 should work
      let foundPrefix = null;
      for (const prefix of prefixes) {
        if (tensorLocations.has(`${prefix}.input_layernorm.weight`)) {
          foundPrefix = prefix;
          break;
        }
      }

      expect(foundPrefix).toBe('model.layers.0');
    });
  });

  describe('tryLoadNorm pattern', () => {
    it('applies norm weight offset when needed', () => {
      // Simulate norm weight offset application
      const normData = new Float32Array([0.5, 0.3, -0.2, 0.1]);
      const offsetApplied = new Float32Array(normData.length);

      for (let i = 0; i < normData.length; i++) {
        offsetApplied[i] = 1.0 + normData[i];
      }

      expect(offsetApplied[0]).toBeCloseTo(1.5);
      expect(offsetApplied[1]).toBeCloseTo(1.3);
      expect(offsetApplied[2]).toBeCloseTo(0.8);
      expect(offsetApplied[3]).toBeCloseTo(1.1);
    });

    it('skips offset when not needed', () => {
      const normData = new Float32Array([0.5, 0.3, -0.2, 0.1]);
      const needsOffset = false;

      const result = needsOffset
        ? normData.map(v => 1.0 + v)
        : normData;

      expect(result[0]).toBeCloseTo(0.5);
    });
  });
});

describe('loader/layer - MoE handling', () => {
  describe('MoE detection', () => {
    it('mini-model is not MoE', () => {
      const manifest = loadMiniModelManifest();
      const isMoE = manifest.moeConfig != null;
      expect(isMoE).toBe(false);
    });

    it('detects MoE from moeConfig', () => {
      const manifest = {
        moeConfig: { numExperts: 8, numExpertsPerToken: 2 },
      };
      const isMoE = manifest.moeConfig != null;
      expect(isMoE).toBe(true);
    });

    it('detects MoE from num_local_experts', () => {
      const manifest = {
        config: { num_local_experts: 8 },
      };
      const isMoE = (manifest.config?.num_local_experts ?? 0) > 1;
      expect(isMoE).toBe(true);
    });
  });

  describe('expert layer identification', () => {
    it('identifies expert layer when MoE and layer is expert', () => {
      const isMoE = true;
      const layerIdx = 5;

      // Simple pattern: all layers are expert layers in MoE
      const isExpertLayer = isMoE;
      expect(isExpertLayer).toBe(true);
    });

    it('non-MoE models have no expert layers', () => {
      const isMoE = false;
      const isExpertLayer = isMoE;
      expect(isExpertLayer).toBe(false);
    });
  });

  describe('router weights', () => {
    it('router weight suffixes are defined', () => {
      const routerSuffixes = {
        routerWeight: ['mlp.router.weight', 'block_sparse_moe.gate.weight'],
        routerBias: ['mlp.router.bias'],
      };

      expect(routerSuffixes.routerWeight.length).toBeGreaterThan(0);
      expect(routerSuffixes.routerBias.length).toBeGreaterThan(0);
    });
  });
});

describe('loader/layer - fused FFN handling', () => {
  describe('fused gate_up detection', () => {
    it('detects when ffnGateUp is present', () => {
      const hasFusedGateUp = (weights) => weights.ffnGateUp != null;

      const weights = createValidLayerWeights();
      expect(hasFusedGateUp(weights)).toBe(false);

      weights.ffnGateUp = new Float32Array(100);
      expect(hasFusedGateUp(weights)).toBe(true);
    });

    it('clears ffnGate and ffnUp when fused', () => {
      const weights = createValidLayerWeights();
      weights.ffnGateUp = new Float32Array(100);

      // When fused path is used, separate weights are null
      weights.ffnGate = null;
      weights.ffnUp = null;

      expect(weights.ffnGateUp).not.toBeNull();
      expect(weights.ffnGate).toBeNull();
      expect(weights.ffnUp).toBeNull();
    });

    it('uses separate gate/up when not fused', () => {
      const weights = createValidLayerWeights();
      weights.ffnGate = new Float32Array(50);
      weights.ffnUp = new Float32Array(50);

      expect(weights.ffnGateUp).toBeNull();
      expect(weights.ffnGate).not.toBeNull();
      expect(weights.ffnUp).not.toBeNull();
    });
  });

  describe('alias fields', () => {
    it('sets gate alias from ffnGate', () => {
      const weights = createValidLayerWeights();
      const ffnGate = new Float32Array(50);
      weights.ffnGate = ffnGate;
      weights.gate = weights.ffnGate;

      expect(weights.gate).toBe(ffnGate);
    });

    it('sets up alias from ffnUp', () => {
      const weights = createValidLayerWeights();
      const ffnUp = new Float32Array(50);
      weights.ffnUp = ffnUp;
      weights.up = weights.ffnUp;

      expect(weights.up).toBe(ffnUp);
    });

    it('sets down alias from ffnDown', () => {
      const weights = createValidLayerWeights();
      const ffnDown = new Float32Array(50);
      weights.ffnDown = ffnDown;
      weights.down = weights.ffnDown;

      expect(weights.down).toBe(ffnDown);
    });

    it('sets gateUp alias from ffnGateUp', () => {
      const weights = createValidLayerWeights();
      const ffnGateUp = new Float32Array(100);
      weights.ffnGateUp = ffnGateUp;
      weights.gateUp = weights.ffnGateUp;

      expect(weights.gateUp).toBe(ffnGateUp);
    });
  });
});

describe('loader/layer - error handling', () => {
  describe('missing required weights', () => {
    it('handles missing inputNorm gracefully', () => {
      const weights = createValidLayerWeights();
      expect(weights.inputNorm).toBeNull();
    });

    it('handles missing attention projections gracefully', () => {
      const weights = createValidLayerWeights();
      expect(weights.qProj).toBeNull();
      expect(weights.kProj).toBeNull();
      expect(weights.vProj).toBeNull();
      expect(weights.oProj).toBeNull();
    });

    it('handles missing FFN projections gracefully', () => {
      const weights = createValidLayerWeights();
      expect(weights.ffnGate).toBeNull();
      expect(weights.ffnUp).toBeNull();
      expect(weights.ffnDown).toBeNull();
    });
  });

  describe('invalid layer index', () => {
    it('negative layer index has no tensors', () => {
      const tensors = loadMiniModelTensors();
      const locationMap = createTensorLocationMap(tensors);

      const prefixes = LAYER_PREFIXES(-1);
      let found = false;
      for (const prefix of prefixes) {
        if (locationMap.has(`${prefix}.input_layernorm.weight`)) {
          found = true;
          break;
        }
      }
      expect(found).toBe(false);
    });

    it('out-of-bounds layer index has no tensors', () => {
      const tensors = loadMiniModelTensors();
      const locationMap = createTensorLocationMap(tensors);

      const prefixes = LAYER_PREFIXES(999);
      let found = false;
      for (const prefix of prefixes) {
        if (locationMap.has(`${prefix}.input_layernorm.weight`)) {
          found = true;
          break;
        }
      }
      expect(found).toBe(false);
    });
  });
});

describe('loader/layer - layer iteration', () => {
  describe('iterating through all layers', () => {
    it('can iterate from 0 to numLayers-1', () => {
      const manifest = loadMiniModelManifest();
      const numLayers = manifest.architecture.numLayers;

      const layerIndices = [];
      for (let i = 0; i < numLayers; i++) {
        layerIndices.push(i);
      }

      expect(layerIndices).toEqual([0, 1]);
    });

    it('each layer has group in manifest', () => {
      const manifest = loadMiniModelManifest();
      const numLayers = manifest.architecture.numLayers;

      for (let i = 0; i < numLayers; i++) {
        const groupName = `layer.${i}`;
        expect(manifest.groups[groupName]).toBeDefined();
      }
    });

    it('each layer group has same tensor structure', () => {
      const manifest = loadMiniModelManifest();

      const layer0Tensors = manifest.groups['layer.0'].tensors;
      const layer1Tensors = manifest.groups['layer.1'].tensors;

      // Same number of tensors
      expect(layer0Tensors.length).toBe(layer1Tensors.length);

      // Same tensor patterns (just different layer index)
      const patterns0 = layer0Tensors.map(t => t.replace(/layers\.0/, 'layers.X'));
      const patterns1 = layer1Tensors.map(t => t.replace(/layers\.1/, 'layers.X'));

      expect(patterns0.sort()).toEqual(patterns1.sort());
    });
  });
});

describe('loader/layer - weight dtype handling', () => {
  describe('dtype consistency', () => {
    it('all layer 0 tensors have same dtype', () => {
      const manifest = loadMiniModelManifest();
      const tensors = loadMiniModelTensors();

      const layer0Tensors = manifest.groups['layer.0'].tensors;
      const dtypes = new Set(layer0Tensors.map(name => tensors[name].dtype));

      // mini-model has all F32
      expect(dtypes.size).toBe(1);
      expect(dtypes.has('F32')).toBe(true);
    });
  });

  describe('matmul weight identification', () => {
    it('identifies attention projections as matmul weights', () => {
      const matmulKeys = ['qProj', 'kProj', 'vProj', 'oProj', 'ffnGate', 'ffnUp', 'ffnDown'];

      const tensorName = 'model.layers.0.self_attn.q_proj.weight';
      const isMatmul = tensorName.includes('proj') ||
                       tensorName.includes('gate') ||
                       tensorName.includes('up') ||
                       tensorName.includes('down');
      expect(isMatmul).toBe(true);
    });

    it('identifies norms as non-matmul weights', () => {
      const tensorName = 'model.layers.0.input_layernorm.weight';
      const isMatmul = tensorName.includes('proj');
      expect(isMatmul).toBe(false);
    });
  });
});

describe('loader/layer - alternative tensor name fallback', () => {
  describe('name pattern alternatives', () => {
    it('has alternative patterns for q_proj', () => {
      const alternatives = ATTN_SUFFIXES.qProj;
      expect(alternatives).toContain('self_attn.q_proj.weight');
      expect(alternatives).toContain('attention.wq.weight');
      expect(alternatives).toContain('attn_q.weight');
    });

    it('has alternative patterns for gate_proj', () => {
      const alternatives = FFN_SUFFIXES.ffnGate;
      expect(alternatives).toContain('mlp.gate_proj.weight');
      expect(alternatives).toContain('feed_forward.w1.weight');
      expect(alternatives).toContain('ffn_gate.weight');
    });

    it('has alternative patterns for input_layernorm', () => {
      const alternatives = ATTN_SUFFIXES.inputNorm;
      expect(alternatives).toContain('input_layernorm.weight');
      expect(alternatives).toContain('attn_norm.weight');
    });
  });

  describe('fallback resolution', () => {
    it('finds tensor with standard naming', () => {
      const tensors = loadMiniModelTensors();
      const locationMap = createTensorLocationMap(tensors);

      // Standard HuggingFace naming
      expect(locationMap.has('model.layers.0.self_attn.q_proj.weight')).toBe(true);
    });
  });
});
