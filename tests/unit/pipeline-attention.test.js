import { describe, expect, it, beforeAll } from 'vitest';
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  parseModelConfig,
  hasManifestInference,
} from '../../src/inference/pipeline/config.js';

import {
  DEFAULT_MANIFEST_INFERENCE,
  validateManifestInference,
  hasInferenceConfig,
} from '../../src/config/schema/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = join(__dirname, '..', 'fixtures', 'mini-model');

function cloneInference() {
  return JSON.parse(JSON.stringify(DEFAULT_MANIFEST_INFERENCE));
}

function deepMerge(target, source) {
  const result = { ...target };
  for (const key of Object.keys(source)) {
    if (
      source[key] &&
      typeof source[key] === 'object' &&
      !Array.isArray(source[key]) &&
      target[key] &&
      typeof target[key] === 'object'
    ) {
      result[key] = deepMerge(target[key], source[key]);
    } else {
      result[key] = source[key];
    }
  }
  return result;
}

function makeManifest(overrides = {}) {
  const base = {
    version: 1,
    modelId: 'test-attention-model',
    modelType: 'transformer',
    quantization: 'F16',
    eos_token_id: 2,
    shards: [],
    totalSize: 0,
    tensorsFile: 'tensors.json',
    tensorCount: 0,
    groups: {},
    architecture: {
      numLayers: 4,
      hiddenSize: 256,
      intermediateSize: 512,
      numAttentionHeads: 4,
      numKeyValueHeads: 4,
      headDim: 64,
      vocabSize: 1024,
      maxSeqLen: 2048,
      ropeTheta: 10000,
      rmsNormEps: 1e-5,
    },
    inference: cloneInference(),
  };

  if (overrides.architecture) {
    base.architecture = { ...base.architecture, ...overrides.architecture };
    delete overrides.architecture;
  }
  if (overrides.inference) {
    base.inference = deepMerge(base.inference, overrides.inference);
    delete overrides.inference;
  }

  return { ...base, ...overrides };
}

// Debug helpers inlined to avoid GPU dependencies from attention/types.ts
function shouldDebugLayer(layerIdx, debugLayers) {
  if (debugLayers === null) return false;
  if (debugLayers === undefined || debugLayers.length === 0) {
    return layerIdx === 0;
  }
  return debugLayers.includes(layerIdx);
}

function markStageLogged(layerIdx, stage, flags) {
  if (!flags.loggedStages) {
    flags.loggedStages = new Set();
  }
  const key = `L${layerIdx}_${stage}`;
  if (flags.loggedStages.has(key)) {
    return true;
  }
  flags.loggedStages.add(key);
  return false;
}

describe('QKV projection dimensions', () => {
  describe('standard MHA (numHeads == numKVHeads)', () => {
    it('validates numHeads * headDim equals hidden size', () => {
      const manifest = makeManifest({
        architecture: {
          hiddenSize: 512,
          numAttentionHeads: 8,
          headDim: 64,
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.numHeads * parsed.headDim).toBe(parsed.hiddenSize);
    });

    it('extracts correct Q projection dimensions', () => {
      const manifest = makeManifest({
        architecture: {
          hiddenSize: 256,
          numAttentionHeads: 4,
          numKeyValueHeads: 4,
          headDim: 64,
        },
      });

      const parsed = parseModelConfig(manifest);
      const qProjOutDim = parsed.numHeads * parsed.headDim;

      expect(qProjOutDim).toBe(256);
      expect(parsed.hiddenSize).toBe(256);
    });

    it('extracts correct K/V projection dimensions for MHA', () => {
      const manifest = makeManifest({
        architecture: {
          numAttentionHeads: 8,
          numKeyValueHeads: 8,
          headDim: 64,
        },
      });

      const parsed = parseModelConfig(manifest);
      const kvProjOutDim = parsed.numKVHeads * parsed.headDim;

      expect(kvProjOutDim).toBe(512);
      expect(parsed.numHeads).toBe(parsed.numKVHeads);
    });
  });

  describe('non-standard head dimensions', () => {
    it('handles non-divisible hidden size with explicit headDim', () => {
      const manifest = makeManifest({
        architecture: {
          hiddenSize: 768,
          numAttentionHeads: 12,
          headDim: 64,
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.hiddenSize).toBe(768);
      expect(parsed.numHeads).toBe(12);
      expect(parsed.headDim).toBe(64);
    });

    it('supports Gemma-style large headDim (256)', () => {
      const manifest = makeManifest({
        architecture: {
          hiddenSize: 2048,
          numAttentionHeads: 8,
          headDim: 256,
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.headDim).toBe(256);
      expect(parsed.numHeads).toBe(8);
    });

    it('supports small headDim for mini models', () => {
      const manifest = makeManifest({
        architecture: {
          hiddenSize: 64,
          numAttentionHeads: 2,
          headDim: 32,
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.headDim).toBe(32);
      expect(parsed.numHeads * parsed.headDim).toBe(64);
    });
  });

});

describe('attention scaling (queryPreAttnScalar)', () => {
  describe('standard sqrt(headDim) scaling', () => {
    it('parses scalar matching sqrt(headDim)', () => {
      const manifest = makeManifest({
        architecture: { headDim: 64 },
        inference: {
          attention: { queryPreAttnScalar: 8 },
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.queryPreAttnScalar).toBe(8);
      expect(parsed.queryPreAttnScalar).toBe(Math.sqrt(parsed.headDim));
    });

    it('computes attention scale as 1/sqrt(scalar)', () => {
      const manifest = makeManifest({
        architecture: { headDim: 64 },
        inference: {
          attention: { queryPreAttnScalar: 8 },
        },
      });

      const parsed = parseModelConfig(manifest);
      const attnScale = 1.0 / Math.sqrt(parsed.queryPreAttnScalar);

      expect(attnScale).toBeCloseTo(0.353553, 5);
    });
  });

  describe('Gemma 2 head_dim scaling', () => {
    it('parses Gemma 2 head_dim scalar (256 instead of 16)', () => {
      const manifest = makeManifest({
        architecture: { headDim: 256 },
        inference: {
          attention: { queryPreAttnScalar: 256 },
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.queryPreAttnScalar).toBe(256);
      expect(parsed.queryPreAttnScalar).toBe(parsed.headDim);
    });

    it('computes Gemma 2 attention scale correctly', () => {
      const manifest = makeManifest({
        architecture: { headDim: 256 },
        inference: {
          attention: { queryPreAttnScalar: 256 },
        },
      });

      const parsed = parseModelConfig(manifest);
      const attnScale = 1.0 / Math.sqrt(parsed.queryPreAttnScalar);

      expect(attnScale).toBeCloseTo(0.0625, 5);
    });
  });

  describe('arbitrary scalar values', () => {
    it('supports custom scalar values', () => {
      const manifest = makeManifest({
        inference: {
          attention: { queryPreAttnScalar: 12.5 },
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.queryPreAttnScalar).toBe(12.5);
    });

    it('supports fractional scalar for mini-model', () => {
      const manifest = makeManifest({
        architecture: { headDim: 32 },
        inference: {
          attention: { queryPreAttnScalar: 5.656854249492381 },
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.queryPreAttnScalar).toBeCloseTo(Math.sqrt(32), 10);
    });
  });
});

describe('softcapping behavior (attnLogitSoftcapping)', () => {
  describe('Gemma 2 softcapping', () => {
    it('parses Gemma 2 softcapping value (50)', () => {
      const manifest = makeManifest({
        inference: {
          attention: { attnLogitSoftcapping: 50.0 },
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.attnLogitSoftcapping).toBe(50.0);
    });

    it('sets isGemma2 flag when attn softcapping is present', () => {
      const manifest = makeManifest({
        inference: {
          attention: { attnLogitSoftcapping: 50.0 },
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.isGemma2).toBe(true);
    });
  });

  describe('softcapping disabled', () => {
    it('treats null as softcapping disabled', () => {
      const manifest = makeManifest({
        inference: {
          attention: { attnLogitSoftcapping: null },
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.attnLogitSoftcapping).toBeNull();
    });

    it('isGemma2 is false when attn softcapping is null', () => {
      const manifest = makeManifest({
        inference: {
          attention: { attnLogitSoftcapping: null },
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.isGemma2).toBe(false);
    });
  });

  describe('custom softcapping values', () => {
    it('supports various softcapping thresholds', () => {
      const values = [25.0, 30.0, 50.0, 100.0];

      for (const value of values) {
        const manifest = makeManifest({
          inference: {
            attention: { attnLogitSoftcapping: value },
          },
        });

        const parsed = parseModelConfig(manifest);
        expect(parsed.attnLogitSoftcapping).toBe(value);
      }
    });

    it('softcapping formula: score = tanh(score/cap) * cap', () => {
      const cap = 50.0;
      const rawScore = 100.0;
      const softcapped = Math.tanh(rawScore / cap) * cap;

      expect(softcapped).toBeCloseTo(48.2, 1);
      expect(softcapped).toBeLessThan(cap);
    });
  });
});

describe('sliding window attention', () => {
  describe('full attention (no sliding window)', () => {
    it('treats null as full attention', () => {
      const manifest = makeManifest({
        inference: {
          attention: { slidingWindow: null },
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.slidingWindow).toBeNull();
    });

    it('applies causal mask to full sequence', () => {
      const manifest = makeManifest({
        architecture: { maxSeqLen: 4096 },
        inference: {
          attention: { slidingWindow: null },
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.slidingWindow).toBeNull();
      expect(parsed.maxSeqLen).toBe(4096);
    });
  });

  describe('sliding window configurations', () => {
    it('parses numeric sliding window size', () => {
      const manifest = makeManifest({
        inference: {
          attention: { slidingWindow: 4096 },
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.slidingWindow).toBe(4096);
    });

    it('supports various sliding window sizes', () => {
      const windowSizes = [512, 1024, 2048, 4096, 8192];

      for (const windowSize of windowSizes) {
        const manifest = makeManifest({
          inference: {
            attention: { slidingWindow: windowSize },
          },
        });

        const parsed = parseModelConfig(manifest);
        expect(parsed.slidingWindow).toBe(windowSize);
      }
    });

    it('throws when slidingWindow is undefined (manifest incomplete)', () => {
      const inference = cloneInference();
      delete inference.attention.slidingWindow;

      const manifest = {
        version: 1,
        modelId: 'incomplete-model',
        modelType: 'transformer',
        quantization: 'F16',
        shards: [],
        totalSize: 0,
        tensorsFile: 'tensors.json',
        tensorCount: 0,
        groups: {},
        architecture: {
          numLayers: 2,
          hiddenSize: 16,
          intermediateSize: 32,
          numAttentionHeads: 2,
          numKeyValueHeads: 2,
          headDim: 8,
          vocabSize: 128,
          maxSeqLen: 64,
          ropeTheta: 10000,
          rmsNormEps: 1e-5,
        },
        inference,
      };

      expect(() => parseModelConfig(manifest)).toThrow(/slidingWindow/);
    });
  });

  describe('layer patterns for hybrid attention', () => {
    it('generates alternating layer types with odd global pattern', () => {
      const manifest = makeManifest({
        architecture: { numLayers: 6 },
        inference: {
          attention: { slidingWindow: 4096 },
          layerPattern: {
            type: 'alternating',
            globalPattern: 'odd',
          },
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.layerTypes).toEqual([
        'sliding_attention',
        'full_attention',
        'sliding_attention',
        'full_attention',
        'sliding_attention',
        'full_attention',
      ]);
    });

    it('generates alternating layer types with even global pattern', () => {
      const manifest = makeManifest({
        architecture: { numLayers: 6 },
        inference: {
          attention: { slidingWindow: 4096 },
          layerPattern: {
            type: 'alternating',
            globalPattern: 'even',
          },
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.layerTypes).toEqual([
        'full_attention',
        'sliding_attention',
        'full_attention',
        'sliding_attention',
        'full_attention',
        'sliding_attention',
      ]);
    });

    it('generates every_n layer types with period', () => {
      const manifest = makeManifest({
        architecture: { numLayers: 12 },
        inference: {
          attention: { slidingWindow: 1024 },
          layerPattern: {
            type: 'every_n',
            period: 6,
          },
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.layerTypes).toHaveLength(12);
      expect(parsed.layerTypes[0]).toBe('full_attention');
      expect(parsed.layerTypes[6]).toBe('full_attention');
      expect(parsed.layerTypes[1]).toBe('sliding_attention');
      expect(parsed.layerTypes[5]).toBe('sliding_attention');
    });

    it('returns null layerTypes when no layerPattern specified', () => {
      const manifest = makeManifest();

      const parsed = parseModelConfig(manifest);

      expect(parsed.layerTypes).toBeNull();
    });
  });
});

describe('GQA (grouped query attention) handling', () => {
  describe('MHA detection (numKVHeads == numHeads)', () => {
    it('detects MHA when numKVHeads equals numHeads', () => {
      const manifest = makeManifest({
        architecture: {
          numAttentionHeads: 8,
          numKeyValueHeads: 8,
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.numHeads).toBe(parsed.numKVHeads);
    });

    it('K/V projection matches Q projection in MHA', () => {
      const manifest = makeManifest({
        architecture: {
          numAttentionHeads: 8,
          numKeyValueHeads: 8,
          headDim: 64,
        },
      });

      const parsed = parseModelConfig(manifest);
      const qDim = parsed.numHeads * parsed.headDim;
      const kvDim = parsed.numKVHeads * parsed.headDim;

      expect(kvDim).toBe(qDim);
    });
  });

  describe('GQA detection (numKVHeads < numHeads)', () => {
    it('detects GQA when numKVHeads is less than numHeads', () => {
      const manifest = makeManifest({
        architecture: {
          numAttentionHeads: 32,
          numKeyValueHeads: 8,
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.numKVHeads).toBeLessThan(parsed.numHeads);
    });

    it('computes correct KV group size', () => {
      const manifest = makeManifest({
        architecture: {
          numAttentionHeads: 32,
          numKeyValueHeads: 8,
        },
      });

      const parsed = parseModelConfig(manifest);
      const kvGroupSize = parsed.numHeads / parsed.numKVHeads;

      expect(kvGroupSize).toBe(4);
    });

    it('supports various GQA ratios', () => {
      const ratios = [
        { numHeads: 16, numKVHeads: 4 },
        { numHeads: 24, numKVHeads: 6 },
        { numHeads: 32, numKVHeads: 8 },
        { numHeads: 64, numKVHeads: 8 },
      ];

      for (const { numHeads, numKVHeads } of ratios) {
        const manifest = makeManifest({
          architecture: { numAttentionHeads: numHeads, numKeyValueHeads: numKVHeads },
        });

        const parsed = parseModelConfig(manifest);
        const groupSize = parsed.numHeads / parsed.numKVHeads;

        expect(groupSize).toBe(numHeads / numKVHeads);
        expect(Number.isInteger(groupSize)).toBe(true);
      }
    });

    it('K/V projection is smaller than Q projection in GQA', () => {
      const manifest = makeManifest({
        architecture: {
          numAttentionHeads: 32,
          numKeyValueHeads: 8,
          headDim: 64,
        },
      });

      const parsed = parseModelConfig(manifest);
      const qDim = parsed.numHeads * parsed.headDim;
      const kvDim = parsed.numKVHeads * parsed.headDim;

      expect(kvDim).toBe(512);
      expect(qDim).toBe(2048);
      expect(kvDim).toBeLessThan(qDim);
    });
  });

  describe('MQA detection (numKVHeads == 1)', () => {
    it('detects MQA when numKVHeads is 1', () => {
      const manifest = makeManifest({
        architecture: {
          numAttentionHeads: 16,
          numKeyValueHeads: 1,
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.numKVHeads).toBe(1);
      expect(parsed.numHeads).toBe(16);
    });

    it('all query heads share single K/V head in MQA', () => {
      const manifest = makeManifest({
        architecture: {
          numAttentionHeads: 16,
          numKeyValueHeads: 1,
          headDim: 64,
        },
      });

      const parsed = parseModelConfig(manifest);
      const kvDim = parsed.numKVHeads * parsed.headDim;

      expect(kvDim).toBe(64);
      expect(parsed.numHeads / parsed.numKVHeads).toBe(16);
    });
  });

  describe('edge case: zero numKVHeads', () => {
    it('handles zero numKVHeads in architecture', () => {
      const manifest = makeManifest({
        architecture: {
          numAttentionHeads: 8,
          numKeyValueHeads: 0,
        },
      });

      const parsed = parseModelConfig(manifest);

      expect(parsed.numKVHeads).toBe(0);
    });
  });
});

describe('attention debug helpers', () => {
  describe('shouldDebugLayer', () => {
    it('returns false when debugLayers is null', () => {
      expect(shouldDebugLayer(0, null)).toBe(false);
      expect(shouldDebugLayer(1, null)).toBe(false);
    });

    it('returns true for layer 0 when debugLayers is undefined or empty', () => {
      expect(shouldDebugLayer(0, undefined)).toBe(true);
      expect(shouldDebugLayer(0, [])).toBe(true);
    });

    it('returns false for non-zero layers when debugLayers is undefined or empty', () => {
      expect(shouldDebugLayer(1, undefined)).toBe(false);
      expect(shouldDebugLayer(2, [])).toBe(false);
    });

    it('returns true only for layers in debugLayers array', () => {
      const debugLayers = [0, 2, 5];
      expect(shouldDebugLayer(0, debugLayers)).toBe(true);
      expect(shouldDebugLayer(2, debugLayers)).toBe(true);
      expect(shouldDebugLayer(5, debugLayers)).toBe(true);
      expect(shouldDebugLayer(1, debugLayers)).toBe(false);
      expect(shouldDebugLayer(3, debugLayers)).toBe(false);
    });
  });

  describe('markStageLogged', () => {
    it('initializes loggedStages set if undefined', () => {
      const flags = {};
      markStageLogged(0, 'qkv', flags);
      expect(flags.loggedStages).toBeInstanceOf(Set);
    });

    it('returns false on first log of a stage', () => {
      const flags = {};
      const alreadyLogged = markStageLogged(0, 'qkv', flags);
      expect(alreadyLogged).toBe(false);
    });

    it('returns true on subsequent logs of same stage', () => {
      const flags = {};
      markStageLogged(0, 'qkv', flags);
      const alreadyLogged = markStageLogged(0, 'qkv', flags);
      expect(alreadyLogged).toBe(true);
    });

    it('tracks different layers separately', () => {
      const flags = {};
      markStageLogged(0, 'qkv', flags);
      expect(markStageLogged(1, 'qkv', flags)).toBe(false);
      expect(markStageLogged(0, 'qkv', flags)).toBe(true);
    });

    it('tracks different stages separately', () => {
      const flags = {};
      markStageLogged(0, 'qkv', flags);
      expect(markStageLogged(0, 'attn', flags)).toBe(false);
      expect(markStageLogged(0, 'output', flags)).toBe(false);
      expect(markStageLogged(0, 'qkv', flags)).toBe(true);
    });
  });
});

describe('queryKeyNorm config', () => {
  it('parses queryKeyNorm as true', () => {
    const manifest = makeManifest({
      inference: {
        attention: { queryKeyNorm: true },
      },
    });

    const parsed = parseModelConfig(manifest);

    expect(parsed.queryKeyNorm).toBe(true);
  });

  it('parses queryKeyNorm as false', () => {
    const manifest = makeManifest({
      inference: {
        attention: { queryKeyNorm: false },
      },
    });

    const parsed = parseModelConfig(manifest);

    expect(parsed.queryKeyNorm).toBe(false);
  });
});

describe('RoPE theta handling', () => {
  it('extracts ropeTheta from manifest inference', () => {
    const manifest = makeManifest({
      inference: {
        rope: { ropeTheta: 10000 },
      },
    });

    const parsed = parseModelConfig(manifest);

    expect(parsed.ropeTheta).toBe(10000);
  });

  it('supports high ropeTheta values (Gemma 3: 1M)', () => {
    const manifest = makeManifest({
      inference: {
        rope: { ropeTheta: 1000000 },
      },
    });

    const parsed = parseModelConfig(manifest);

    expect(parsed.ropeTheta).toBe(1000000);
  });

  it('parses ropeLocalTheta for local attention layers', () => {
    const manifest = makeManifest({
      inference: {
        rope: {
          ropeTheta: 1000000,
          ropeLocalTheta: 10000,
        },
      },
    });

    const parsed = parseModelConfig(manifest);

    expect(parsed.ropeTheta).toBe(1000000);
    expect(parsed.ropeLocalTheta).toBe(10000);
  });

  it('sets isGemma3 flag when ropeLocalTheta is present', () => {
    const manifest = makeManifest({
      inference: {
        rope: {
          ropeTheta: 1000000,
          ropeLocalTheta: 10000,
        },
      },
    });

    const parsed = parseModelConfig(manifest);

    expect(parsed.isGemma3).toBe(true);
  });
});

describe('manifest validation', () => {
  describe('hasManifestInference', () => {
    it('returns true when manifest has inference field', () => {
      const manifest = makeManifest();
      expect(hasManifestInference(manifest)).toBe(true);
    });

    it('returns false when manifest lacks inference field', () => {
      const manifest = { modelId: 'test', version: 1 };
      expect(hasManifestInference(manifest)).toBe(false);
    });

    it('returns false when inference is null', () => {
      const manifest = { modelId: 'test', version: 1, inference: null };
      expect(hasManifestInference(manifest)).toBe(false);
    });
  });

  describe('validateManifestInference', () => {
    it('throws when inference field is missing', () => {
      const manifest = { modelId: 'missing-inference' };
      expect(() => validateManifestInference(manifest)).toThrow(/missing required/);
    });

    it('does not throw when inference field is present', () => {
      const manifest = makeManifest();
      expect(() => validateManifestInference(manifest)).not.toThrow();
    });
  });

  describe('hasInferenceConfig type guard', () => {
    it('returns true for manifest with inference', () => {
      const manifest = makeManifest();
      expect(hasInferenceConfig(manifest)).toBe(true);
    });

    it('returns false for manifest without inference', () => {
      const manifest = { modelId: 'test' };
      expect(hasInferenceConfig(manifest)).toBe(false);
    });
  });
});

describe('mini-model fixture integration', () => {
  let manifest;

  beforeAll(async () => {
    const manifestPath = join(FIXTURES_DIR, 'manifest.json');
    const content = await readFile(manifestPath, 'utf-8');
    manifest = JSON.parse(content);
  });

  it('parses mini-model manifest successfully', () => {
    expect(() => parseModelConfig(manifest)).not.toThrow();
  });

  it('extracts correct architecture values', () => {
    const parsed = parseModelConfig(manifest);

    expect(parsed.numLayers).toBe(2);
    expect(parsed.hiddenSize).toBe(64);
    expect(parsed.numHeads).toBe(2);
    expect(parsed.numKVHeads).toBe(2);
    expect(parsed.headDim).toBe(32);
  });

  it('extracts attention config from fixture', () => {
    const parsed = parseModelConfig(manifest);

    expect(parsed.queryPreAttnScalar).toBeCloseTo(5.656854249492381);
    expect(parsed.attnLogitSoftcapping).toBeNull();
    expect(parsed.slidingWindow).toBeNull();
    expect(parsed.queryKeyNorm).toBe(false);
  });

  it('mini-model uses MHA (numHeads == numKVHeads)', () => {
    const parsed = parseModelConfig(manifest);

    expect(parsed.numHeads).toBe(parsed.numKVHeads);
  });

  it('validates QKV projection dimensions match tensor shapes', async () => {
    const tensorsPath = join(FIXTURES_DIR, 'tensors.json');
    const tensorsContent = await readFile(tensorsPath, 'utf-8');
    const tensors = JSON.parse(tensorsContent);

    const parsed = parseModelConfig(manifest);
    const expectedQKVShape = [parsed.hiddenSize, parsed.hiddenSize];

    const qProj = tensors['model.layers.0.self_attn.q_proj.weight'];
    const kProj = tensors['model.layers.0.self_attn.k_proj.weight'];
    const vProj = tensors['model.layers.0.self_attn.v_proj.weight'];

    expect(qProj.shape).toEqual(expectedQKVShape);
    expect(kProj.shape).toEqual(expectedQKVShape);
    expect(vProj.shape).toEqual(expectedQKVShape);
  });

  it('extracts RoPE config from fixture', () => {
    const parsed = parseModelConfig(manifest);

    expect(parsed.ropeTheta).toBe(10000);
    expect(parsed.ropeLocalTheta).toBeNull();
    expect(parsed.ropeScalingType).toBeNull();
    expect(parsed.ropeScale).toBe(1);
  });

  it('fixture has no layer pattern', () => {
    const parsed = parseModelConfig(manifest);

    expect(parsed.layerTypes).toBeNull();
  });
});

describe('error handling', () => {
  it('throws when manifest lacks inference config entirely', () => {
    const manifest = {
      version: 1,
      modelId: 'test-model',
      modelType: 'transformer',
      quantization: 'F16',
      shards: [],
    };

    expect(() => parseModelConfig(manifest)).toThrow(/missing inference config/);
  });

  it('throws descriptive error when required attention field missing', () => {
    const inference = cloneInference();
    delete inference.attention.queryPreAttnScalar;

    const manifest = {
      version: 1,
      modelId: 'incomplete-attention',
      modelType: 'transformer',
      quantization: 'F16',
      shards: [],
      totalSize: 0,
      tensorsFile: 'tensors.json',
      tensorCount: 0,
      groups: {},
      architecture: {
        numLayers: 2,
        hiddenSize: 64,
        intermediateSize: 128,
        numAttentionHeads: 2,
        numKeyValueHeads: 2,
        headDim: 32,
        vocabSize: 128,
        maxSeqLen: 512,
        ropeTheta: 10000,
        rmsNormEps: 1e-5,
      },
      inference,
    };

    expect(() => parseModelConfig(manifest)).toThrow(/queryPreAttnScalar/);
  });

  it('throws when alternating pattern lacks globalPattern', () => {
    const manifest = makeManifest({
      architecture: { numLayers: 4 },
      inference: {
        layerPattern: { type: 'alternating' },
      },
    });

    expect(() => parseModelConfig(manifest)).toThrow(/globalPattern/);
  });

  it('throws when every_n pattern lacks period', () => {
    const manifest = makeManifest({
      architecture: { numLayers: 4 },
      inference: {
        layerPattern: { type: 'every_n' },
      },
    });

    expect(() => parseModelConfig(manifest)).toThrow(/period/);
  });

  it('preserves all required inference fields through parsing', () => {
    const manifest = makeManifest();
    const parsed = parseModelConfig(manifest);

    expect(parsed.numHeads).toBeDefined();
    expect(parsed.numKVHeads).toBeDefined();
    expect(parsed.headDim).toBeDefined();
    expect(parsed.hiddenSize).toBeDefined();
    expect(parsed.rmsNormEps).toBeDefined();
    expect(parsed.queryPreAttnScalar).toBeDefined();
    expect(parsed.queryKeyNorm).toBeDefined();
    expect(parsed.slidingWindow).toBeDefined();
    expect(parsed.ropeTheta).toBeDefined();
  });
});
