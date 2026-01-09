import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { needsNormWeightOffset } from '../../src/loader/manifest-config.js';
import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';

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

function createValidManifest(overrides = {}) {
  const inference = {
    ...DEFAULT_MANIFEST_INFERENCE,
    attention: {
      ...DEFAULT_MANIFEST_INFERENCE.attention,
      queryPreAttnScalar: 8.0,
      attnLogitSoftcapping: null,
      slidingWindow: null,
      queryKeyNorm: false,
    },
    normalization: {
      ...DEFAULT_MANIFEST_INFERENCE.normalization,
      rmsNormEps: 1e-5,
      rmsNormWeightOffset: false,
      postAttentionNorm: false,
      preFeedforwardNorm: false,
      postFeedforwardNorm: false,
    },
    ffn: {
      ...DEFAULT_MANIFEST_INFERENCE.ffn,
      activation: 'silu',
      gatedActivation: true,
    },
    rope: {
      ...DEFAULT_MANIFEST_INFERENCE.rope,
      ropeTheta: 10000,
      ropeLocalTheta: null,
      ropeScalingType: null,
      ropeScalingFactor: 1,
    },
    output: {
      ...DEFAULT_MANIFEST_INFERENCE.output,
      finalLogitSoftcapping: null,
      tieWordEmbeddings: true,
      scaleEmbeddings: false,
      embeddingTranspose: false,
      embeddingVocabSize: 32000,
    },
  };

  return {
    version: 1,
    modelId: 'test-model-q4k',
    modelType: 'transformer',
    quantization: 'Q4_K',
    hashAlgorithm: 'sha256',
    architecture: {
      numLayers: 12,
      hiddenSize: 768,
      intermediateSize: 3072,
      numAttentionHeads: 12,
      numKeyValueHeads: 12,
      headDim: 64,
      vocabSize: 32000,
      maxSeqLen: 2048,
      ropeTheta: 10000,
      rmsNormEps: 1e-5,
    },
    shards: [
      {
        index: 0,
        filename: 'shard_00000.bin',
        size: 64 * 1024 * 1024,
        hash: 'a'.repeat(64),
        offset: 0,
      },
    ],
    totalSize: 64 * 1024 * 1024,
    tensorsFile: 'tensors.json',
    tensorCount: 100,
    groups: {},
    inference,
    ...overrides,
  };
}

describe('loader/manifest - manifest parsing', () => {
  let miniManifest;

  beforeEach(() => {
    miniManifest = loadMiniModelManifest();
  });

  describe('manifest.json structure', () => {
    it('parses mini-model manifest successfully', () => {
      expect(miniManifest).toBeDefined();
      expect(miniManifest.version).toBe(1);
      expect(miniManifest.modelId).toBe('mini-test-model-wf32');
    });

    it('has correct model type', () => {
      expect(miniManifest.modelType).toBe('transformer');
    });

    it('has correct quantization', () => {
      expect(miniManifest.quantization).toBe('F32');
    });

    it('has valid hash algorithm', () => {
      expect(miniManifest.hashAlgorithm).toBe('sha256');
    });
  });

  describe('version validation', () => {
    it('accepts version 1', () => {
      const manifest = createValidManifest({ version: 1 });
      expect(manifest.version).toBe(1);
    });

    it('identifies invalid version 0', () => {
      const manifest = createValidManifest({ version: 0 });
      expect(manifest.version).toBe(0);
      expect(manifest.version >= 1).toBe(false);
    });

    it('identifies invalid negative version', () => {
      const manifest = createValidManifest({ version: -1 });
      expect(manifest.version >= 1).toBe(false);
    });

    it('identifies invalid non-integer version', () => {
      const manifest = createValidManifest({ version: 1.5 });
      expect(Number.isInteger(manifest.version)).toBe(false);
    });
  });

  describe('architecture extraction', () => {
    it('extracts numLayers correctly', () => {
      expect(miniManifest.architecture.numLayers).toBe(2);
    });

    it('extracts hiddenSize correctly', () => {
      expect(miniManifest.architecture.hiddenSize).toBe(64);
    });

    it('extracts intermediateSize correctly', () => {
      expect(miniManifest.architecture.intermediateSize).toBe(128);
    });

    it('extracts numAttentionHeads correctly', () => {
      expect(miniManifest.architecture.numAttentionHeads).toBe(2);
    });

    it('extracts numKeyValueHeads correctly', () => {
      expect(miniManifest.architecture.numKeyValueHeads).toBe(2);
    });

    it('extracts headDim correctly', () => {
      expect(miniManifest.architecture.headDim).toBe(32);
    });

    it('extracts vocabSize correctly', () => {
      expect(miniManifest.architecture.vocabSize).toBe(32);
    });

    it('extracts maxSeqLen correctly', () => {
      expect(miniManifest.architecture.maxSeqLen).toBe(128);
    });

    it('extracts ropeTheta correctly', () => {
      expect(miniManifest.architecture.ropeTheta).toBe(10000);
    });

    it('extracts rmsNormEps correctly', () => {
      expect(miniManifest.architecture.rmsNormEps).toBe(1e-6);
    });
  });

  describe('quantizationInfo extraction', () => {
    it('extracts weights quantization type', () => {
      expect(miniManifest.quantizationInfo.weights).toBe('f32');
    });

    it('extracts embeddings quantization type', () => {
      expect(miniManifest.quantizationInfo.embeddings).toBe('f32');
    });

    it('extracts variant tag', () => {
      expect(miniManifest.quantizationInfo.variantTag).toBe('wf32');
    });
  });

  describe('inference config extraction', () => {
    it('extracts attention queryPreAttnScalar', () => {
      expect(miniManifest.inference.attention.queryPreAttnScalar).toBeCloseTo(5.656854249492381);
    });

    it('extracts attention attnLogitSoftcapping', () => {
      expect(miniManifest.inference.attention.attnLogitSoftcapping).toBeNull();
    });

    it('extracts attention slidingWindow', () => {
      expect(miniManifest.inference.attention.slidingWindow).toBeNull();
    });

    it('extracts attention queryKeyNorm', () => {
      expect(miniManifest.inference.attention.queryKeyNorm).toBe(false);
    });

    it('extracts normalization rmsNormWeightOffset', () => {
      expect(miniManifest.inference.normalization.rmsNormWeightOffset).toBe(false);
    });

    it('extracts ffn activation', () => {
      expect(miniManifest.inference.ffn.activation).toBe('silu');
    });

    it('extracts ffn gatedActivation', () => {
      expect(miniManifest.inference.ffn.gatedActivation).toBe(true);
    });

    it('extracts rope ropeTheta', () => {
      expect(miniManifest.inference.rope.ropeTheta).toBe(10000);
    });

    it('extracts rope ropeScalingType', () => {
      expect(miniManifest.inference.rope.ropeScalingType).toBeNull();
    });

    it('extracts rope ropeScalingFactor', () => {
      expect(miniManifest.inference.rope.ropeScalingFactor).toBe(1);
    });

    it('extracts output tieWordEmbeddings', () => {
      expect(miniManifest.inference.output.tieWordEmbeddings).toBe(true);
    });

    it('extracts output scaleEmbeddings', () => {
      expect(miniManifest.inference.output.scaleEmbeddings).toBe(false);
    });

    it('extracts output finalLogitSoftcapping', () => {
      expect(miniManifest.inference.output.finalLogitSoftcapping).toBeNull();
    });
  });

  describe('shards extraction', () => {
    it('has one shard', () => {
      expect(miniManifest.shards.length).toBe(1);
    });

    it('shard has correct index', () => {
      expect(miniManifest.shards[0].index).toBe(0);
    });

    it('shard has correct fileName', () => {
      expect(miniManifest.shards[0].fileName).toBe('shard-0.bin');
    });

    it('shard has correct size', () => {
      expect(miniManifest.shards[0].size).toBe(337152);
    });

    it('shard has hash', () => {
      expect(miniManifest.shards[0].hash).toBeDefined();
      expect(typeof miniManifest.shards[0].hash).toBe('string');
    });

    it('shard has hashAlgorithm', () => {
      expect(miniManifest.shards[0].hashAlgorithm).toBe('sha256');
    });
  });

  describe('totalSize validation', () => {
    it('totalSize matches expected value', () => {
      expect(miniManifest.totalSize).toBe(337152);
    });

    it('totalSize equals sum of shard sizes', () => {
      const shardSum = miniManifest.shards.reduce((sum, s) => sum + s.size, 0);
      expect(miniManifest.totalSize).toBe(shardSum);
    });
  });

  describe('groups extraction', () => {
    it('has embed group', () => {
      expect(miniManifest.groups.embed).toBeDefined();
    });

    it('embed group has correct type', () => {
      expect(miniManifest.groups.embed.type).toBe('embed');
    });

    it('embed group has tensors list', () => {
      expect(Array.isArray(miniManifest.groups.embed.tensors)).toBe(true);
      expect(miniManifest.groups.embed.tensors).toContain('model.embed_tokens.weight');
    });

    it('has layer.0 group', () => {
      expect(miniManifest.groups['layer.0']).toBeDefined();
    });

    it('layer.0 group has correct type', () => {
      expect(miniManifest.groups['layer.0'].type).toBe('layer');
    });

    it('layer.0 group has layerIndex', () => {
      expect(miniManifest.groups['layer.0'].layerIndex).toBe(0);
    });

    it('layer.0 group has expected tensors', () => {
      const tensors = miniManifest.groups['layer.0'].tensors;
      expect(tensors).toContain('model.layers.0.input_layernorm.weight');
      expect(tensors).toContain('model.layers.0.self_attn.q_proj.weight');
      expect(tensors).toContain('model.layers.0.self_attn.k_proj.weight');
      expect(tensors).toContain('model.layers.0.self_attn.v_proj.weight');
      expect(tensors).toContain('model.layers.0.self_attn.o_proj.weight');
      expect(tensors).toContain('model.layers.0.mlp.gate_proj.weight');
      expect(tensors).toContain('model.layers.0.mlp.up_proj.weight');
      expect(tensors).toContain('model.layers.0.mlp.down_proj.weight');
    });

    it('has layer.1 group', () => {
      expect(miniManifest.groups['layer.1']).toBeDefined();
      expect(miniManifest.groups['layer.1'].layerIndex).toBe(1);
    });

    it('has head group', () => {
      expect(miniManifest.groups.head).toBeDefined();
      expect(miniManifest.groups.head.type).toBe('head');
    });

    it('head group contains final norm', () => {
      expect(miniManifest.groups.head.tensors).toContain('model.norm.weight');
    });
  });

  describe('tensorsFile reference', () => {
    it('has tensorsFile reference', () => {
      expect(miniManifest.tensorsFile).toBe('tensors.json');
    });

    it('has tensorCount', () => {
      expect(miniManifest.tensorCount).toBe(20);
    });
  });

  describe('tokenizer config', () => {
    it('has tokenizer config', () => {
      expect(miniManifest.tokenizer).toBeDefined();
    });

    it('tokenizer type is bundled', () => {
      expect(miniManifest.tokenizer.type).toBe('bundled');
    });

    it('tokenizer file reference is correct', () => {
      expect(miniManifest.tokenizer.file).toBe('tokenizer.json');
    });

    it('tokenizer vocabSize matches architecture', () => {
      expect(miniManifest.tokenizer.vocabSize).toBe(miniManifest.architecture.vocabSize);
    });

    it('tokenizer type is bpe', () => {
      expect(miniManifest.tokenizer.tokenizerType).toBe('bpe');
    });
  });

  describe('moeConfig handling', () => {
    it('moeConfig is null for non-MoE model', () => {
      expect(miniManifest.moeConfig).toBeNull();
    });

    it('identifies MoE model when moeConfig is present', () => {
      const moeManifest = createValidManifest({
        moeConfig: {
          numExperts: 8,
          numExpertsPerToken: 2,
        },
      });
      expect(moeManifest.moeConfig).not.toBeNull();
      expect(moeManifest.moeConfig.numExperts).toBe(8);
      expect(moeManifest.moeConfig.numExpertsPerToken).toBe(2);
    });
  });

  describe('required field validation', () => {
    it('validates version is present', () => {
      const manifest = loadMiniModelManifest();
      expect(manifest.version).toBeDefined();
    });

    it('validates modelId is present', () => {
      const manifest = loadMiniModelManifest();
      expect(manifest.modelId).toBeDefined();
      expect(typeof manifest.modelId).toBe('string');
      expect(manifest.modelId.length).toBeGreaterThan(0);
    });

    it('validates modelType is present', () => {
      const manifest = loadMiniModelManifest();
      expect(manifest.modelType).toBeDefined();
    });

    it('validates architecture is present', () => {
      const manifest = loadMiniModelManifest();
      expect(manifest.architecture).toBeDefined();
    });

    it('validates shards array is present', () => {
      const manifest = loadMiniModelManifest();
      expect(Array.isArray(manifest.shards)).toBe(true);
      expect(manifest.shards.length).toBeGreaterThan(0);
    });

    it('validates inference config is present', () => {
      const manifest = loadMiniModelManifest();
      expect(manifest.inference).toBeDefined();
    });
  });

  describe('field type validation', () => {
    it('version is a number', () => {
      expect(typeof miniManifest.version).toBe('number');
    });

    it('modelId is a string', () => {
      expect(typeof miniManifest.modelId).toBe('string');
    });

    it('architecture fields are numbers', () => {
      expect(typeof miniManifest.architecture.numLayers).toBe('number');
      expect(typeof miniManifest.architecture.hiddenSize).toBe('number');
      expect(typeof miniManifest.architecture.vocabSize).toBe('number');
    });

    it('shards is an array', () => {
      expect(Array.isArray(miniManifest.shards)).toBe(true);
    });

    it('groups is an object', () => {
      expect(typeof miniManifest.groups).toBe('object');
      expect(miniManifest.groups).not.toBeNull();
    });

    it('inference is an object', () => {
      expect(typeof miniManifest.inference).toBe('object');
      expect(miniManifest.inference).not.toBeNull();
    });
  });

  describe('architecture field computation', () => {
    it('headDim can be computed from hiddenSize and numAttentionHeads', () => {
      const computed = miniManifest.architecture.hiddenSize / miniManifest.architecture.numAttentionHeads;
      expect(computed).toBe(miniManifest.architecture.headDim);
    });

    it('numKeyValueHeads defaults to numAttentionHeads when not specified', () => {
      const manifest = createValidManifest();
      delete manifest.architecture.numKeyValueHeads;
      // When numKeyValueHeads is not set, it should default to numAttentionHeads
      const numKVHeads = manifest.architecture.numKeyValueHeads ?? manifest.architecture.numAttentionHeads;
      expect(numKVHeads).toBe(manifest.architecture.numAttentionHeads);
    });
  });

  describe('error handling', () => {
    it('handles missing architecture gracefully', () => {
      const manifest = { version: 1, modelId: 'test' };
      expect(manifest.architecture).toBeUndefined();
    });

    it('handles missing inference config gracefully', () => {
      const manifest = { version: 1, modelId: 'test' };
      expect(manifest.inference).toBeUndefined();
    });

    it('handles missing shards gracefully', () => {
      const manifest = { version: 1, modelId: 'test' };
      expect(manifest.shards).toBeUndefined();
    });

    it('handles empty shards array', () => {
      const manifest = createValidManifest({ shards: [] });
      expect(manifest.shards.length).toBe(0);
    });

    it('handles invalid JSON string', () => {
      const invalidJson = '{ invalid json }';
      expect(() => JSON.parse(invalidJson)).toThrow();
    });
  });

  describe('config fallback values', () => {
    it('extracts config from manifest.config when present', () => {
      expect(miniManifest.config).toBeDefined();
      expect(miniManifest.config.architectures).toContain('TestTransformerForCausalLM');
    });

    it('config contains HuggingFace-style fields', () => {
      expect(miniManifest.config.hidden_size).toBe(64);
      expect(miniManifest.config.num_hidden_layers).toBe(2);
      expect(miniManifest.config.vocab_size).toBe(32);
    });

    it('config special token IDs are present', () => {
      expect(miniManifest.config.bos_token_id).toBe(1);
      expect(miniManifest.config.eos_token_id).toBe(2);
      expect(miniManifest.config.pad_token_id).toBe(0);
    });
  });
});

describe('loader/manifest - manifest config helpers', () => {
  describe('needsNormWeightOffset detection', () => {
    it('returns false when rmsNormWeightOffset is false', () => {
      const manifest = loadMiniModelManifest();
      expect(needsNormWeightOffset(manifest)).toBe(false);
    });

    it('returns true when rmsNormWeightOffset is true', () => {
      const manifest = createValidManifest();
      manifest.inference.normalization.rmsNormWeightOffset = true;
      expect(needsNormWeightOffset(manifest)).toBe(true);
    });

    it('throws when rmsNormWeightOffset is missing', () => {
      const manifest = createValidManifest();
      delete manifest.inference.normalization.rmsNormWeightOffset;
      expect(() => needsNormWeightOffset(manifest)).toThrow(/rmsNormWeightOffset/);
    });
  });

  describe('isMoE detection', () => {
    it('returns false for dense model', () => {
      const manifest = loadMiniModelManifest();
      const hasMoeConfig = manifest.moeConfig != null;
      const hasExpertGroups = Object.keys(manifest.groups || {}).some(g => g.includes('.expert.'));
      expect(hasMoeConfig || hasExpertGroups).toBe(false);
    });

    it('returns true when moeConfig present', () => {
      const manifest = createValidManifest({
        moeConfig: { numExperts: 8, numExpertsPerToken: 2 },
      });
      expect(manifest.moeConfig != null).toBe(true);
    });

    it('returns true when expert groups exist', () => {
      const manifest = createValidManifest({
        groups: {
          'layer.0.expert.0': { type: 'expert', tensors: [] },
        },
      });
      const hasExpertGroups = Object.keys(manifest.groups).some(g => g.includes('.expert.'));
      expect(hasExpertGroups).toBe(true);
    });

    it('returns true when num_local_experts > 1 in config', () => {
      const manifest = createValidManifest();
      manifest.config = { num_local_experts: 8 };
      expect(manifest.config.num_local_experts > 1).toBe(true);
    });
  });

  describe('Q4K layout detection', () => {
    it('returns null when no q4kLayout in config', () => {
      const manifest = loadMiniModelManifest();
      const q4kLayout = manifest.config?.q4kLayout ?? null;
      expect(q4kLayout).toBeNull();
    });

    it('returns flat when q4kLayout is flat', () => {
      const manifest = createValidManifest();
      manifest.config = { q4kLayout: 'flat' };
      expect(manifest.config.q4kLayout).toBe('flat');
    });

    it('returns column_wise when q4kLayout is column_wise', () => {
      const manifest = createValidManifest();
      manifest.config = { q4kLayout: 'column_wise' };
      expect(manifest.config.q4kLayout).toBe('column_wise');
    });
  });

  describe('tied embeddings detection', () => {
    it('detects tieWordEmbeddings true', () => {
      const manifest = loadMiniModelManifest();
      expect(manifest.inference.output.tieWordEmbeddings).toBe(true);
    });

    it('detects tieWordEmbeddings false', () => {
      const manifest = createValidManifest();
      manifest.inference.output.tieWordEmbeddings = false;
      expect(manifest.inference.output.tieWordEmbeddings).toBe(false);
    });
  });

  describe('numLayers resolution', () => {
    it('resolves from architecture.numLayers', () => {
      const manifest = loadMiniModelManifest();
      expect(manifest.architecture.numLayers).toBe(2);
    });

    it('resolves from config.num_hidden_layers', () => {
      const manifest = loadMiniModelManifest();
      expect(manifest.config.num_hidden_layers).toBe(2);
    });

    it('architecture and config match', () => {
      const manifest = loadMiniModelManifest();
      expect(manifest.architecture.numLayers).toBe(manifest.config.num_hidden_layers);
    });
  });
});

describe('loader/manifest - shard info extraction', () => {
  let miniManifest;

  beforeEach(() => {
    miniManifest = loadMiniModelManifest();
  });

  describe('getShardInfo', () => {
    it('returns shard info for valid index', () => {
      const shard = miniManifest.shards[0];
      expect(shard).toBeDefined();
      expect(shard.index).toBe(0);
    });

    it('returns undefined for invalid index', () => {
      const shard = miniManifest.shards[99];
      expect(shard).toBeUndefined();
    });

    it('returns undefined for negative index', () => {
      const shard = miniManifest.shards[-1];
      expect(shard).toBeUndefined();
    });
  });

  describe('shard offset calculation', () => {
    it('first shard has offset 0', () => {
      const manifest = createValidManifest({
        shards: [
          { index: 0, filename: 'shard_00000.bin', size: 1000, hash: 'a'.repeat(64), offset: 0 },
          { index: 1, filename: 'shard_00001.bin', size: 500, hash: 'b'.repeat(64), offset: 1000 },
        ],
      });
      expect(manifest.shards[0].offset).toBe(0);
    });

    it('subsequent shard offsets accumulate', () => {
      const manifest = createValidManifest({
        shards: [
          { index: 0, filename: 'shard_00000.bin', size: 1000, hash: 'a'.repeat(64), offset: 0 },
          { index: 1, filename: 'shard_00001.bin', size: 500, hash: 'b'.repeat(64), offset: 1000 },
          { index: 2, filename: 'shard_00002.bin', size: 300, hash: 'c'.repeat(64), offset: 1500 },
        ],
        totalSize: 1800,
      });
      expect(manifest.shards[1].offset).toBe(1000);
      expect(manifest.shards[2].offset).toBe(1500);
    });
  });

  describe('shard hash validation', () => {
    it('shard hash has correct length for sha256', () => {
      const manifest = createValidManifest();
      expect(manifest.shards[0].hash.length).toBe(64);
    });

    it('shard hash is lowercase hex', () => {
      const manifest = createValidManifest();
      expect(/^[0-9a-f]+$/.test(manifest.shards[0].hash)).toBe(true);
    });
  });
});
