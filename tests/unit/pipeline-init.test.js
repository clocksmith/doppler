import { describe, expect, it, beforeAll, afterAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { parseModelConfig, hasManifestInference } from '../../src/inference/pipeline/config.js';
import { PipelineState } from '../../src/inference/pipeline/state.js';
import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';

const FIXTURES_DIR = join(import.meta.dirname, '../fixtures/mini-model');

function loadMiniModelManifest() {
  const raw = readFileSync(join(FIXTURES_DIR, 'manifest.json'), 'utf-8');
  return JSON.parse(raw);
}

function cloneInference() {
  return JSON.parse(JSON.stringify(DEFAULT_MANIFEST_INFERENCE));
}

function makeManifest(overrides = {}) {
  return {
    version: 1,
    modelId: 'test-model',
    modelType: 'transformer',
    quantization: 'F16',
    shards: [],
    totalSize: 0,
    tensorsFile: 'tensors.json',
    tensorCount: 0,
    groups: {},
    architecture: {
      numLayers: 4,
      hiddenSize: 256,
      intermediateSize: 512,
      numAttentionHeads: 8,
      numKeyValueHeads: 8,
      headDim: 32,
      vocabSize: 1024,
      maxSeqLen: 512,
      ropeTheta: 10000,
      rmsNormEps: 1e-5,
    },
    inference: cloneInference(),
    ...overrides,
  };
}

describe('pipeline initialization', () => {
  describe('parseModelConfig with valid manifest', () => {
    let miniManifest;

    beforeAll(() => {
      miniManifest = loadMiniModelManifest();
    });

    it('parses mini-model manifest successfully', () => {
      const config = parseModelConfig(miniManifest);

      expect(config.numLayers).toBe(2);
      expect(config.hiddenSize).toBe(64);
      expect(config.intermediateSize).toBe(128);
      expect(config.numHeads).toBe(2);
      expect(config.numKVHeads).toBe(2);
      expect(config.headDim).toBe(32);
      expect(config.vocabSize).toBe(32);
      expect(config.maxSeqLen).toBe(128);
    });

    it('extracts RoPE configuration from manifest', () => {
      const config = parseModelConfig(miniManifest);

      expect(config.ropeTheta).toBe(10000);
      expect(config.ropeLocalTheta).toBeNull();
      expect(config.ropeScalingType).toBeNull();
      expect(config.ropeScale).toBe(1);
    });

    it('extracts attention configuration from manifest', () => {
      const config = parseModelConfig(miniManifest);

      expect(config.queryPreAttnScalar).toBeCloseTo(5.656854249492381);
      expect(config.attnLogitSoftcapping).toBeNull();
      expect(config.slidingWindow).toBeNull();
      expect(config.queryKeyNorm).toBe(false);
    });

    it('extracts normalization configuration from manifest', () => {
      const config = parseModelConfig(miniManifest);

      expect(config.rmsNormWeightOffset).toBe(false);
    });

    it('extracts output configuration from manifest', () => {
      const config = parseModelConfig(miniManifest);

      expect(config.finalLogitSoftcapping).toBeNull();
      expect(config.scaleEmbeddings).toBe(false);
    });

    it('extracts FFN activation from manifest', () => {
      const config = parseModelConfig(miniManifest);

      expect(config.hiddenActivation).toBe('silu');
    });
  });

  describe('config parsing from manifest architecture', () => {
    it('extracts architecture dimensions correctly', () => {
      const manifest = makeManifest();
      const config = parseModelConfig(manifest);

      expect(config.numLayers).toBe(4);
      expect(config.hiddenSize).toBe(256);
      expect(config.intermediateSize).toBe(512);
      expect(config.numHeads).toBe(8);
      expect(config.numKVHeads).toBe(8);
      expect(config.headDim).toBe(32);
      expect(config.vocabSize).toBe(1024);
      expect(config.maxSeqLen).toBe(512);
    });

    it('uses inference.rope.ropeTheta as source of truth', () => {
      const manifest = makeManifest();
      manifest.inference.rope.ropeTheta = 500000;
      manifest.architecture.ropeTheta = 10000;

      const config = parseModelConfig(manifest);

      expect(config.ropeTheta).toBe(500000);
    });

    it('parses gelu activation correctly', () => {
      const manifest = makeManifest();
      manifest.inference.ffn.activation = 'gelu';

      const config = parseModelConfig(manifest);

      expect(config.hiddenActivation).toBe('gelu');
    });

    it('parses geglu activation as gelu', () => {
      const manifest = makeManifest();
      manifest.inference.ffn.activation = 'geglu';

      const config = parseModelConfig(manifest);

      expect(config.hiddenActivation).toBe('gelu');
    });

    it('parses swiglu activation as silu', () => {
      const manifest = makeManifest();
      manifest.inference.ffn.activation = 'swiglu';

      const config = parseModelConfig(manifest);

      expect(config.hiddenActivation).toBe('silu');
    });
  });

  describe('state initialization', () => {
    it('creates PipelineState with default values', () => {
      const state = new PipelineState();

      expect(state.currentSeqLen).toBe(0);
      expect(state.isLoaded).toBe(false);
      expect(state.isGenerating).toBe(false);
      expect(state.debug).toBe(false);
    });

    it('initializes with null components', () => {
      const state = new PipelineState();

      expect(state.tokenizer).toBeNull();
      expect(state.kvCache).toBeNull();
      expect(state.moeRouter).toBeNull();
      expect(state.speculativeDecoder).toBeNull();
      expect(state.manifest).toBeNull();
      expect(state.modelConfig).toBeNull();
    });

    it('initializes empty KV cache state', () => {
      const state = new PipelineState();

      expect(state.kvCache).toBeNull();
      expect(state.ropeFreqsCos).toBeNull();
      expect(state.ropeFreqsSin).toBeNull();
      expect(state.ropeLocalCos).toBeNull();
      expect(state.ropeLocalSin).toBeNull();
    });

    it('initializes empty weights map', () => {
      const state = new PipelineState();

      expect(state.weights).toBeInstanceOf(Map);
      expect(state.weights.size).toBe(0);
      expect(state.expertWeights).toBeInstanceOf(Map);
      expect(state.expertWeights.size).toBe(0);
    });

    it('initializes stats with zero values', () => {
      const state = new PipelineState();

      expect(state.stats.prefillTimeMs).toBe(0);
      expect(state.stats.decodeTimeMs).toBe(0);
      expect(state.stats.prefillTokens).toBe(0);
      expect(state.stats.decodeTokens).toBe(0);
    });

    it('initializes batching stats with zero values', () => {
      const state = new PipelineState();

      expect(state.batchingStats.batchedForwardCalls).toBe(0);
      expect(state.batchingStats.unbatchedForwardCalls).toBe(0);
      expect(state.batchingStats.totalBatchedTimeMs).toBe(0);
      expect(state.batchingStats.totalUnbatchedTimeMs).toBe(0);
      expect(state.batchingStats.gpuSubmissions).toBe(0);
    });

    it('initializes tied embeddings as false', () => {
      const state = new PipelineState();

      expect(state.useTiedEmbeddings).toBe(false);
      expect(state.embeddingVocabSize).toBeNull();
      expect(state.embeddingTranspose).toBe(false);
    });
  });

  describe('error handling for missing required fields', () => {
    it('throws when manifest has no inference config', () => {
      const manifest = {
        version: 1,
        modelId: 'no-inference-model',
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
          maxSeqLen: 64,
        },
      };

      expect(() => parseModelConfig(manifest)).toThrow(/missing inference config/i);
    });

    it('throws when attention.queryPreAttnScalar is missing', () => {
      const manifest = makeManifest();
      delete manifest.inference.attention.queryPreAttnScalar;

      expect(() => parseModelConfig(manifest)).toThrow(/queryPreAttnScalar/);
    });

    it('throws when attention.queryKeyNorm is missing', () => {
      const manifest = makeManifest();
      delete manifest.inference.attention.queryKeyNorm;

      expect(() => parseModelConfig(manifest)).toThrow(/queryKeyNorm/);
    });

    it('throws when attention.slidingWindow is undefined', () => {
      const manifest = makeManifest();
      delete manifest.inference.attention.slidingWindow;

      expect(() => parseModelConfig(manifest)).toThrow(/slidingWindow/);
    });

    it('throws when attention.attnLogitSoftcapping is undefined', () => {
      const manifest = makeManifest();
      delete manifest.inference.attention.attnLogitSoftcapping;

      expect(() => parseModelConfig(manifest)).toThrow(/attnLogitSoftcapping/);
    });

    it('throws when normalization.rmsNormWeightOffset is missing', () => {
      const manifest = makeManifest();
      delete manifest.inference.normalization.rmsNormWeightOffset;

      expect(() => parseModelConfig(manifest)).toThrow(/rmsNormWeightOffset/);
    });

    it('throws when normalization.postAttentionNorm is missing', () => {
      const manifest = makeManifest();
      delete manifest.inference.normalization.postAttentionNorm;

      expect(() => parseModelConfig(manifest)).toThrow(/postAttentionNorm/);
    });

    it('throws when normalization.preFeedforwardNorm is missing', () => {
      const manifest = makeManifest();
      delete manifest.inference.normalization.preFeedforwardNorm;

      expect(() => parseModelConfig(manifest)).toThrow(/preFeedforwardNorm/);
    });

    it('throws when normalization.postFeedforwardNorm is missing', () => {
      const manifest = makeManifest();
      delete manifest.inference.normalization.postFeedforwardNorm;

      expect(() => parseModelConfig(manifest)).toThrow(/postFeedforwardNorm/);
    });

    it('throws when ffn.activation is missing', () => {
      const manifest = makeManifest();
      delete manifest.inference.ffn.activation;

      expect(() => parseModelConfig(manifest)).toThrow(/ffn\.activation/);
    });

    it('throws when ffn.gatedActivation is missing', () => {
      const manifest = makeManifest();
      delete manifest.inference.ffn.gatedActivation;

      expect(() => parseModelConfig(manifest)).toThrow(/gatedActivation/);
    });

    it('throws when rope.ropeTheta is missing', () => {
      const manifest = makeManifest();
      delete manifest.inference.rope.ropeTheta;

      expect(() => parseModelConfig(manifest)).toThrow(/ropeTheta/);
    });

    it('throws when rope.ropeScalingFactor is missing', () => {
      const manifest = makeManifest();
      delete manifest.inference.rope.ropeScalingFactor;

      expect(() => parseModelConfig(manifest)).toThrow(/ropeScalingFactor/);
    });

    it('throws when rope.ropeScalingType is undefined', () => {
      const manifest = makeManifest();
      delete manifest.inference.rope.ropeScalingType;

      expect(() => parseModelConfig(manifest)).toThrow(/ropeScalingType/);
    });

    it('throws when rope.ropeLocalTheta is undefined', () => {
      const manifest = makeManifest();
      delete manifest.inference.rope.ropeLocalTheta;

      expect(() => parseModelConfig(manifest)).toThrow(/ropeLocalTheta/);
    });

    it('throws when output.tieWordEmbeddings is missing', () => {
      const manifest = makeManifest();
      delete manifest.inference.output.tieWordEmbeddings;

      expect(() => parseModelConfig(manifest)).toThrow(/tieWordEmbeddings/);
    });

    it('throws when output.scaleEmbeddings is missing', () => {
      const manifest = makeManifest();
      delete manifest.inference.output.scaleEmbeddings;

      expect(() => parseModelConfig(manifest)).toThrow(/scaleEmbeddings/);
    });

    it('throws when output.embeddingTranspose is missing', () => {
      const manifest = makeManifest();
      delete manifest.inference.output.embeddingTranspose;

      expect(() => parseModelConfig(manifest)).toThrow(/embeddingTranspose/);
    });

    it('throws when output.embeddingVocabSize is undefined', () => {
      const manifest = makeManifest();
      delete manifest.inference.output.embeddingVocabSize;

      expect(() => parseModelConfig(manifest)).toThrow(/embeddingVocabSize/);
    });

    it('throws when output.finalLogitSoftcapping is undefined', () => {
      const manifest = makeManifest();
      delete manifest.inference.output.finalLogitSoftcapping;

      expect(() => parseModelConfig(manifest)).toThrow(/finalLogitSoftcapping/);
    });
  });

  describe('error handling for invalid manifest version', () => {
    it('accepts valid version 1 manifest', () => {
      const manifest = makeManifest({ version: 1 });

      expect(() => parseModelConfig(manifest)).not.toThrow();
    });
  });

  describe('config validation', () => {
    it('parses config when hiddenSize is divisible by numHeads', () => {
      const manifest = makeManifest();
      manifest.architecture.hiddenSize = 256;
      manifest.architecture.numAttentionHeads = 8;
      manifest.architecture.headDim = 32;

      const config = parseModelConfig(manifest);

      expect(config.hiddenSize).toBe(256);
      expect(config.numHeads).toBe(8);
      expect(config.headDim).toBe(32);
    });

    it('uses architecture headDim directly', () => {
      const manifest = makeManifest();
      manifest.architecture.hiddenSize = 256;
      manifest.architecture.numAttentionHeads = 4;
      manifest.architecture.headDim = 64;

      const config = parseModelConfig(manifest);

      expect(config.headDim).toBe(64);
    });

    it('accepts null for nullable inference fields', () => {
      const manifest = makeManifest();
      manifest.inference.attention.slidingWindow = null;
      manifest.inference.attention.attnLogitSoftcapping = null;
      manifest.inference.output.finalLogitSoftcapping = null;
      manifest.inference.rope.ropeScalingType = null;
      manifest.inference.rope.ropeLocalTheta = null;

      expect(() => parseModelConfig(manifest)).not.toThrow();

      const config = parseModelConfig(manifest);
      expect(config.slidingWindow).toBeNull();
      expect(config.attnLogitSoftcapping).toBeNull();
      expect(config.finalLogitSoftcapping).toBeNull();
      expect(config.ropeScalingType).toBeNull();
      expect(config.ropeLocalTheta).toBeNull();
    });

    it('parses sliding window when set', () => {
      const manifest = makeManifest();
      manifest.inference.attention.slidingWindow = 4096;

      const config = parseModelConfig(manifest);

      expect(config.slidingWindow).toBe(4096);
    });

    it('parses attention softcapping when set', () => {
      const manifest = makeManifest();
      manifest.inference.attention.attnLogitSoftcapping = 50.0;

      const config = parseModelConfig(manifest);

      expect(config.attnLogitSoftcapping).toBe(50.0);
    });

    it('parses final logit softcapping when set', () => {
      const manifest = makeManifest();
      manifest.inference.output.finalLogitSoftcapping = 30.0;

      const config = parseModelConfig(manifest);

      expect(config.finalLogitSoftcapping).toBe(30.0);
    });
  });

  describe('hasManifestInference', () => {
    it('returns true when manifest has inference config', () => {
      const manifest = makeManifest();

      expect(hasManifestInference(manifest)).toBe(true);
    });

    it('returns false when manifest has no inference config', () => {
      const manifest = {
        version: 1,
        modelId: 'test',
        modelType: 'transformer',
      };

      expect(hasManifestInference(manifest)).toBe(false);
    });

    it('returns false when inference is null', () => {
      const manifest = {
        version: 1,
        modelId: 'test',
        modelType: 'transformer',
        inference: null,
      };

      expect(hasManifestInference(manifest)).toBe(false);
    });

    it('returns false when inference is undefined', () => {
      const manifest = {
        version: 1,
        modelId: 'test',
        modelType: 'transformer',
        inference: undefined,
      };

      expect(hasManifestInference(manifest)).toBe(false);
    });
  });

  describe('layer pattern parsing', () => {
    it('throws when alternating pattern lacks globalPattern', () => {
      const manifest = makeManifest();
      manifest.inference.layerPattern = { type: 'alternating' };

      expect(() => parseModelConfig(manifest)).toThrow(/globalPattern/);
    });

    it('throws when every_n pattern lacks period', () => {
      const manifest = makeManifest();
      manifest.inference.layerPattern = { type: 'every_n' };

      expect(() => parseModelConfig(manifest)).toThrow(/period/);
    });

    it('parses alternating even pattern correctly', () => {
      const manifest = makeManifest();
      manifest.architecture.numLayers = 4;
      manifest.inference.layerPattern = { type: 'alternating', globalPattern: 'even' };

      const config = parseModelConfig(manifest);

      expect(config.layerTypes).toEqual([
        'full_attention',
        'sliding_attention',
        'full_attention',
        'sliding_attention',
      ]);
    });

    it('parses alternating odd pattern correctly', () => {
      const manifest = makeManifest();
      manifest.architecture.numLayers = 4;
      manifest.inference.layerPattern = { type: 'alternating', globalPattern: 'odd' };

      const config = parseModelConfig(manifest);

      expect(config.layerTypes).toEqual([
        'sliding_attention',
        'full_attention',
        'sliding_attention',
        'full_attention',
      ]);
    });

    it('parses every_n pattern correctly', () => {
      const manifest = makeManifest();
      manifest.architecture.numLayers = 6;
      manifest.inference.layerPattern = { type: 'every_n', period: 3 };

      const config = parseModelConfig(manifest);

      expect(config.layerTypes).toEqual([
        'full_attention',
        'sliding_attention',
        'sliding_attention',
        'full_attention',
        'sliding_attention',
        'sliding_attention',
      ]);
    });
  });

  describe('YARN rope scaling', () => {
    it('propagates YARN params into parsed config', () => {
      const manifest = makeManifest();
      manifest.inference.rope.ropeScalingType = 'yarn';
      manifest.inference.rope.ropeScalingFactor = 2.0;
      manifest.inference.rope.yarnBetaFast = 32;
      manifest.inference.rope.yarnBetaSlow = 1;
      manifest.inference.rope.yarnOriginalMaxPos = 4096;

      const config = parseModelConfig(manifest);

      expect(config.ropeScalingType).toBe('yarn');
      expect(config.ropeScale).toBe(2.0);
      expect(config.ropeScaling).toEqual({
        type: 'yarn',
        factor: 2.0,
        beta_fast: 32,
        beta_slow: 1,
        original_max_position_embeddings: 4096,
      });
    });

    it('sets ropeScaling to null when no scaling type', () => {
      const manifest = makeManifest();
      manifest.inference.rope.ropeScalingType = null;

      const config = parseModelConfig(manifest);

      expect(config.ropeScaling).toBeNull();
    });

    it('sets linear scaling config when type is linear', () => {
      const manifest = makeManifest();
      manifest.inference.rope.ropeScalingType = 'linear';
      manifest.inference.rope.ropeScalingFactor = 2.0;

      const config = parseModelConfig(manifest);

      expect(config.ropeScalingType).toBe('linear');
      expect(config.ropeScale).toBe(2.0);
      expect(config.ropeScaling).toEqual({
        type: 'linear',
        factor: 2.0,
      });
    });
  });

  describe('model family detection from inference config', () => {
    it('detects isGemma3 when ropeLocalTheta is set', () => {
      const manifest = makeManifest();
      manifest.inference.rope.ropeLocalTheta = 10000;

      const config = parseModelConfig(manifest);

      expect(config.isGemma3).toBe(true);
    });

    it('detects isGemma3 as false when ropeLocalTheta is null', () => {
      const manifest = makeManifest();
      manifest.inference.rope.ropeLocalTheta = null;

      const config = parseModelConfig(manifest);

      expect(config.isGemma3).toBe(false);
    });

    it('detects isGemma2 when attnLogitSoftcapping is set', () => {
      const manifest = makeManifest();
      manifest.inference.attention.attnLogitSoftcapping = 50.0;

      const config = parseModelConfig(manifest);

      expect(config.isGemma2).toBe(true);
    });

    it('detects isGemma2 as false when attnLogitSoftcapping is null', () => {
      const manifest = makeManifest();
      manifest.inference.attention.attnLogitSoftcapping = null;

      const config = parseModelConfig(manifest);

      expect(config.isGemma2).toBe(false);
    });
  });

  describe('runtime overrides', () => {
    it('applies runtime attention overrides', () => {
      const manifest = makeManifest();
      manifest.inference.attention.queryKeyNorm = false;

      const config = parseModelConfig(manifest, {
        attention: { queryKeyNorm: true },
      });

      expect(config.queryKeyNorm).toBe(true);
    });

    it('applies runtime rope overrides', () => {
      const manifest = makeManifest();
      manifest.inference.rope.ropeTheta = 10000;

      const config = parseModelConfig(manifest, {
        rope: { ropeTheta: 500000 },
      });

      expect(config.ropeTheta).toBe(500000);
    });

    it('applies runtime output overrides', () => {
      const manifest = makeManifest();
      manifest.inference.output.finalLogitSoftcapping = null;

      const config = parseModelConfig(manifest, {
        output: { finalLogitSoftcapping: 30.0 },
      });

      expect(config.finalLogitSoftcapping).toBe(30.0);
    });
  });
});
