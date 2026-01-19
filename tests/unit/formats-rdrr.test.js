import { describe, expect, it, beforeEach, afterEach } from 'vitest';

import {
  generateShardFilename,
  calculateShardCount,
  createShardLayout,
  createManifest,
  serializeTensorMap,
  serializeManifest,
  getShardUrl,
  getManifestUrl,
} from '../../src/formats/rdrr/manifest.js';

import {
  parseManifest,
  parseTensorMap,
  getManifest,
  setManifest,
  clearManifest,
  getShardInfo,
  getShardCount,
  isMoE,
} from '../../src/formats/rdrr/parsing.js';

import {
  validateManifest,
} from '../../src/formats/rdrr/validation.js';

import {
  RDRR_VERSION,
  SHARD_SIZE,
  MANIFEST_FILENAME,
  TENSORS_FILENAME,
} from '../../src/formats/rdrr/types.js';
import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';

function createValidManifest(overrides = {}) {
  const inference = {
    ...DEFAULT_MANIFEST_INFERENCE,
    attention: {
      ...DEFAULT_MANIFEST_INFERENCE.attention,
      queryPreAttnScalar: 8.0,
    },
    normalization: {
      ...DEFAULT_MANIFEST_INFERENCE.normalization,
      rmsNormEps: 1e-5,
    },
  };

  return {
    version: RDRR_VERSION,
    modelId: 'test-model-q4k',
    modelType: 'transformer',
    quantization: 'Q4_K',
    hashAlgorithm: 'sha256',
    eos_token_id: 1,
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
      {
        index: 1,
        filename: 'shard_00001.bin',
        size: 32 * 1024 * 1024,
        hash: 'b'.repeat(64),
        offset: 64 * 1024 * 1024,
      },
    ],
    totalSize: 96 * 1024 * 1024,
    tensorsFile: 'tensors.json',
    tensorCount: 100,
    inference,
    ...overrides,
  };
}

function createValidTensorMap() {
  return {
    'model.embed_tokens.weight': {
      shard: 0,
      offset: 0,
      size: 1000 * 256 * 2,
      shape: [1000, 256],
      dtype: 'F16',
      role: 'embedding',
    },
    'model.layers.0.self_attn.q_proj.weight': {
      shard: 0,
      offset: 512000,
      size: 256 * 256 * 2,
      shape: [256, 256],
      dtype: 'F16',
      role: 'q_proj',
    },
    'model.layers.0.self_attn.k_proj.weight': {
      shard: 0,
      offset: 643072,
      size: 256 * 256 * 2,
      shape: [256, 256],
      dtype: 'F16',
      role: 'k_proj',
    },
  };
}

describe('formats/rdrr/manifest', () => {
  describe('generateShardFilename', () => {
    it('generates padded filenames', () => {
      expect(generateShardFilename(0)).toBe('shard_00000.bin');
      expect(generateShardFilename(1)).toBe('shard_00001.bin');
      expect(generateShardFilename(99)).toBe('shard_00099.bin');
      expect(generateShardFilename(12345)).toBe('shard_12345.bin');
    });
  });

  describe('calculateShardCount', () => {
    it('calculates correct shard count', () => {
      expect(calculateShardCount(100 * 1024 * 1024)).toBe(2);
      expect(calculateShardCount(64 * 1024 * 1024)).toBe(1);
      expect(calculateShardCount(64 * 1024 * 1024 + 1)).toBe(2);
    });

    it('uses custom shard size', () => {
      expect(calculateShardCount(100, 50)).toBe(2);
      expect(calculateShardCount(150, 50)).toBe(3);
      expect(calculateShardCount(151, 50)).toBe(4);
    });
  });

  describe('createShardLayout', () => {
    it('creates shard layout with correct offsets', () => {
      const hashes = ['a'.repeat(64), 'b'.repeat(64)];
      const totalSize = 100 * 1024 * 1024;
      const layout = createShardLayout(totalSize, hashes);

      expect(layout.length).toBe(2);

      expect(layout[0].index).toBe(0);
      expect(layout[0].filename).toBe('shard_00000.bin');
      expect(layout[0].size).toBe(SHARD_SIZE);
      expect(layout[0].offset).toBe(0);
      expect(layout[0].hash).toBe('a'.repeat(64));

      expect(layout[1].index).toBe(1);
      expect(layout[1].filename).toBe('shard_00001.bin');
      expect(layout[1].size).toBe(totalSize - SHARD_SIZE);
      expect(layout[1].offset).toBe(SHARD_SIZE);
    });

    it('throws on hash count mismatch', () => {
      const hashes = ['a'.repeat(64)];
      const totalSize = 100 * 1024 * 1024;

      expect(() => createShardLayout(totalSize, hashes)).toThrow(/Hash count mismatch/);
    });
  });

  describe('createManifest', () => {
    it('creates valid manifest', () => {
      const options = {
        modelId: 'test-model',
        modelType: 'transformer',
        quantization: 'F16',
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
            size: 1000,
            hash: 'a'.repeat(64),
            offset: 0,
          },
        ],
        totalSize: 1000,
        tensorsFile: 'tensors.json',
        tensorCount: 10,
        inference: {
          ...DEFAULT_MANIFEST_INFERENCE,
          attention: {
            ...DEFAULT_MANIFEST_INFERENCE.attention,
            queryPreAttnScalar: 8.0,
          },
          normalization: {
            ...DEFAULT_MANIFEST_INFERENCE.normalization,
            rmsNormEps: 1e-5,
          },
        },
      };

      const manifest = createManifest(options);

      expect(manifest.version).toBe(RDRR_VERSION);
      expect(manifest.modelId).toBe('test-model');
      expect(manifest.tensorsFile).toBe('tensors.json');
    });

    it('throws on invalid options', () => {
      const options = {
        modelId: 'test',
        modelType: 'transformer',
        quantization: 'F16',
        hashAlgorithm: 'sha256',
        architecture: {},
        shards: [],
        totalSize: 0,
        inference: {},
      };

      expect(() => createManifest(options)).toThrow();
    });
  });

  describe('serializeTensorMap', () => {
    it('serializes tensor map to JSON', () => {
      const tensorMap = createValidTensorMap();
      const json = serializeTensorMap(tensorMap);
      const parsed = JSON.parse(json);

      expect(parsed['model.embed_tokens.weight']).toBeDefined();
      expect(parsed['model.embed_tokens.weight'].shard).toBe(0);
    });
  });

  describe('serializeManifest', () => {
    it('serializes manifest to JSON', () => {
      const manifest = createValidManifest();
      const json = serializeManifest(manifest);
      const parsed = JSON.parse(json);

      expect(parsed.version).toBe(RDRR_VERSION);
      expect(parsed.modelId).toBe('test-model-q4k');
    });
  });

  describe('getShardUrl', () => {
    beforeEach(() => {
      const manifest = createValidManifest();
      setManifest(manifest);
    });

    afterEach(() => {
      clearManifest();
    });

    it('generates correct shard URL', () => {
      const url = getShardUrl('https://example.com/models/test', 0);
      expect(url).toBe('https://example.com/models/test/shard_00000.bin');
    });

    it('strips trailing slash from base URL', () => {
      const url = getShardUrl('https://example.com/models/test/', 0);
      expect(url).toBe('https://example.com/models/test/shard_00000.bin');
    });

    it('throws on invalid shard index', () => {
      expect(() => getShardUrl('https://example.com', 99)).toThrow(/Invalid shard index/);
    });
  });

  describe('getManifestUrl', () => {
    it('generates correct manifest URL', () => {
      const url = getManifestUrl('https://example.com/models/test');
      expect(url).toBe('https://example.com/models/test/manifest.json');
    });

    it('strips trailing slash', () => {
      const url = getManifestUrl('https://example.com/models/test/');
      expect(url).toBe('https://example.com/models/test/manifest.json');
    });
  });
});

describe('formats/rdrr/parsing', () => {
  afterEach(() => {
    clearManifest();
  });

  describe('parseManifest', () => {
    it('parses valid manifest JSON', () => {
      const manifest = createValidManifest();
      const json = JSON.stringify(manifest);
      const parsed = parseManifest(json);

      expect(parsed.version).toBe(RDRR_VERSION);
      expect(parsed.modelId).toBe('test-model-q4k');
      expect(parsed.architecture.numLayers).toBe(12);
    });

    it('normalizes shard offsets', () => {
      const manifest = createValidManifest();
      delete manifest.shards[0].offset;
      delete manifest.shards[1].offset;

      const json = JSON.stringify(manifest);
      const parsed = parseManifest(json);

      expect(parsed.shards[0].offset).toBe(0);
      expect(parsed.shards[1].offset).toBe(64 * 1024 * 1024);
    });

    it('normalizes shard index', () => {
      const manifest = createValidManifest();
      delete manifest.shards[0].index;
      delete manifest.shards[1].index;

      const json = JSON.stringify(manifest);
      const parsed = parseManifest(json);

      expect(parsed.shards[0].index).toBe(0);
      expect(parsed.shards[1].index).toBe(1);
    });

    it('normalizes fileName to filename', () => {
      const manifest = createValidManifest();
      manifest.shards[0].fileName = 'shard_00000.bin';
      delete manifest.shards[0].filename;

      const json = JSON.stringify(manifest);
      const parsed = parseManifest(json);

      expect(parsed.shards[0].filename).toBe('shard_00000.bin');
    });

    it('sets current manifest', () => {
      const manifest = createValidManifest();
      const json = JSON.stringify(manifest);
      parseManifest(json);

      expect(getManifest()).not.toBeNull();
      expect(getManifest().modelId).toBe('test-model-q4k');
    });

    it('throws on invalid JSON', () => {
      expect(() => parseManifest('not json')).toThrow(/Failed to parse manifest JSON/);
    });

    it('throws on invalid manifest', () => {
      const json = JSON.stringify({ version: 1 });
      expect(() => parseManifest(json)).toThrow(/Invalid manifest/);
    });

    it('throws on missing eos_token_id', () => {
      const manifest = createValidManifest();
      delete manifest.eos_token_id;
      const json = JSON.stringify(manifest);

      expect(() => parseManifest(json)).toThrow(/Missing eos_token_id/);
    });
  });

  describe('parseTensorMap', () => {
    it('parses valid tensor map', () => {
      const tensorMap = createValidTensorMap();
      const json = JSON.stringify(tensorMap);
      const parsed = parseTensorMap(json);

      expect(parsed['model.embed_tokens.weight']).toBeDefined();
      expect(parsed['model.embed_tokens.weight'].shard).toBe(0);
      expect(parsed['model.embed_tokens.weight'].offset).toBe(0);
    });

    it('does not infer role from name without group', () => {
      const tensorMap = {
        'model.embed_tokens.weight': { shard: 0, offset: 0, size: 100, shape: [10] },
      };
      const json = JSON.stringify(tensorMap);

      expect(() => parseTensorMap(json)).toThrow(/missing role/);
    });

    it('throws on missing shard index', () => {
      const tensorMap = {
        'weight': { offset: 0, size: 100, shape: [10] },
      };
      const json = JSON.stringify(tensorMap);

      expect(() => parseTensorMap(json)).toThrow(/missing shard index/);
    });

    it('throws on missing offset', () => {
      const tensorMap = {
        'weight': { shard: 0, size: 100, shape: [10] },
      };
      const json = JSON.stringify(tensorMap);

      expect(() => parseTensorMap(json)).toThrow(/missing offset/);
    });

    it('throws on missing size', () => {
      const tensorMap = {
        'weight': { shard: 0, offset: 0, shape: [10] },
      };
      const json = JSON.stringify(tensorMap);

      expect(() => parseTensorMap(json)).toThrow(/missing size/);
    });

    it('throws on missing shape', () => {
      const tensorMap = {
        'weight': { shard: 0, offset: 0, size: 100 },
      };
      const json = JSON.stringify(tensorMap);

      expect(() => parseTensorMap(json)).toThrow(/missing shape/);
    });

    it('throws on invalid JSON', () => {
      expect(() => parseTensorMap('not json')).toThrow(/Failed to parse tensors.json/);
    });
  });

  describe('getManifest/setManifest/clearManifest', () => {
    it('returns null when no manifest set', () => {
      expect(getManifest()).toBeNull();
    });

    it('sets and gets manifest', () => {
      const manifest = createValidManifest();
      setManifest(manifest);

      expect(getManifest()).toBe(manifest);
    });

    it('clears manifest', () => {
      const manifest = createValidManifest();
      setManifest(manifest);
      clearManifest();

      expect(getManifest()).toBeNull();
    });
  });

  describe('getShardInfo', () => {
    it('returns shard info by index', () => {
      const manifest = createValidManifest();
      setManifest(manifest);

      const shard = getShardInfo(0);
      expect(shard).not.toBeNull();
      expect(shard.index).toBe(0);
      expect(shard.filename).toBe('shard_00000.bin');
    });

    it('returns null for invalid index', () => {
      const manifest = createValidManifest();
      setManifest(manifest);

      expect(getShardInfo(-1)).toBeNull();
      expect(getShardInfo(99)).toBeNull();
    });

    it('returns null when no manifest', () => {
      expect(getShardInfo(0)).toBeNull();
    });
  });

  describe('getShardCount', () => {
    it('returns shard count', () => {
      const manifest = createValidManifest();
      setManifest(manifest);

      expect(getShardCount()).toBe(2);
    });

    it('returns 0 when no manifest', () => {
      expect(getShardCount()).toBe(0);
    });
  });

  describe('isMoE', () => {
    it('returns false for non-MoE model', () => {
      const manifest = createValidManifest();
      setManifest(manifest);

      expect(isMoE()).toBe(false);
    });

    it('returns true when moeConfig present', () => {
      const manifest = createValidManifest({
        moeConfig: { numExperts: 8, numExpertsPerToken: 2, expertFormat: 'mixtral' },
      });
      setManifest(manifest);

      expect(isMoE()).toBe(true);
    });

    it('returns true when expert groups present', () => {
      const manifest = createValidManifest({
        groups: {
          'layer.0.expert.0': { type: 'expert', version: '1', shards: [0], tensors: [], hash: 'a'.repeat(64) },
        },
      });
      setManifest(manifest);

      expect(isMoE()).toBe(true);
    });
  });
});

describe('formats/rdrr/validation', () => {
  describe('validateManifest', () => {
    it('passes for valid manifest', () => {
      const manifest = createValidManifest();
      const result = validateManifest(manifest);

      expect(result.valid).toBe(true);
      expect(result.errors.length).toBe(0);
    });

    it('fails on invalid version', () => {
      const manifest = createValidManifest({ version: 0 });
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('version'))).toBe(true);
    });

    it('fails on missing modelId', () => {
      const manifest = createValidManifest();
      delete manifest.modelId;
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('modelId'))).toBe(true);
    });

    it('fails on missing modelType', () => {
      const manifest = createValidManifest();
      delete manifest.modelType;
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('modelType'))).toBe(true);
    });

    it('fails on missing quantization', () => {
      const manifest = createValidManifest();
      delete manifest.quantization;
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('quantization'))).toBe(true);
    });

    it('fails on invalid hashAlgorithm', () => {
      const manifest = createValidManifest({ hashAlgorithm: 'md5' });
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('hashAlgorithm'))).toBe(true);
    });

    it('fails on missing inference', () => {
      const manifest = createValidManifest();
      delete manifest.inference;
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('inference'))).toBe(true);
    });

    it('fails on invalid architecture fields', () => {
      const manifest = createValidManifest({
        architecture: {
          numLayers: 0,
          hiddenSize: 768,
          intermediateSize: 3072,
          numAttentionHeads: 12,
          vocabSize: 32000,
          maxSeqLen: 2048,
        },
      });
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('numLayers'))).toBe(true);
    });

    it('fails on empty shards', () => {
      const manifest = createValidManifest({ shards: [] });
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('shards'))).toBe(true);
    });

    it('fails on incorrect shard index', () => {
      const manifest = createValidManifest();
      manifest.shards[0].index = 1;
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('incorrect index'))).toBe(true);
    });

    it('fails on invalid shard hash', () => {
      const manifest = createValidManifest();
      manifest.shards[0].hash = 'short';
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('invalid hash'))).toBe(true);
    });

    it('fails on incorrect shard offset', () => {
      const manifest = createValidManifest();
      manifest.shards[1].offset = 0;
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('incorrect offset'))).toBe(true);
    });

    it('fails on totalSize mismatch', () => {
      const manifest = createValidManifest({ totalSize: 1000 });
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('totalSize mismatch'))).toBe(true);
    });

    it('validates MoE config', () => {
      const manifest = createValidManifest({
        moeConfig: { numExperts: 0, numExpertsPerToken: 2, expertFormat: 'mixtral' },
      });
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('numExperts'))).toBe(true);
    });

    it('fails when numExpertsPerToken exceeds numExperts', () => {
      const manifest = createValidManifest({
        moeConfig: { numExperts: 2, numExpertsPerToken: 4, expertFormat: 'mixtral' },
      });
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('cannot exceed'))).toBe(true);
    });

    it('validates group structure', () => {
      const manifest = createValidManifest({
        groups: {
          'layer.0': {
            type: 'layer',
          },
        },
      });
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('version'))).toBe(true);
    });

    it('validates group shard references', () => {
      const manifest = createValidManifest({
        groups: {
          'layer.0': {
            type: 'layer',
            version: '1',
            shards: [99],
            tensors: [],
            hash: 'a'.repeat(64),
          },
        },
      });
      const result = validateManifest(manifest);

      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.includes('invalid shard index'))).toBe(true);
    });

    it('passes for LoRA adapter without architecture', () => {
      const manifest = createValidManifest({
        adapterType: 'lora',
        architecture: undefined,
      });
      delete manifest.architecture;
      const result = validateManifest(manifest);

      expect(result.errors.filter(e => e.includes('architecture'))).toEqual([]);
    });
  });
});

describe('formats/rdrr/types', () => {
  describe('constants', () => {
    it('has correct RDRR version', () => {
      expect(RDRR_VERSION).toBeGreaterThanOrEqual(1);
    });

    it('has correct shard size', () => {
      expect(SHARD_SIZE).toBe(64 * 1024 * 1024);
    });

    it('has correct manifest filename', () => {
      expect(MANIFEST_FILENAME).toBe('manifest.json');
    });

    it('has correct tensors filename', () => {
      expect(TENSORS_FILENAME).toBe('tensors.json');
    });
  });
});
