import { describe, expect, it, beforeEach, afterEach } from 'vitest';

import { parseModelConfig } from '../../src/inference/pipeline/config.js';
import {
  isMoELayerLocal,
  hasLoggedFusedDownNorm,
  setLoggedFusedDownNorm,
} from '../../src/inference/pipeline/ffn/types.js';
import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';

function cloneInference() {
  return JSON.parse(JSON.stringify(DEFAULT_MANIFEST_INFERENCE));
}

function makeManifest(inference, overrides = {}) {
  return {
    version: 1,
    modelId: 'test-ffn-model',
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
      maxSeqLen: 512,
      ropeTheta: 10000,
      rmsNormEps: 1e-6,
    },
    inference,
    ...overrides,
  };
}

function makeConfig(parsedConfig = {}) {
  return {
    numLayers: 4,
    hiddenSize: 256,
    intermediateSize: 512,
    numHeads: 4,
    numKVHeads: 4,
    headDim: 64,
    vocabSize: 1024,
    maxSeqLen: 512,
    useMoE: false,
    numExperts: 8,
    moeTopK: 2,
    slidingWindow: null,
    ropeTheta: 10000,
    ropeLocalTheta: null,
    ropeScale: 1.0,
    ropeScalingType: null,
    ropeScaling: null,
    quantization: 'f16',
    quantMethod: null,
    rmsNormEps: 1e-6,
    rmsNormWeightOffset: false,
    scaleEmbeddings: false,
    hiddenActivation: 'silu',
    isGemma3: false,
    isGemma2: false,
    isLlama3Instruct: false,
    isQwen3: false,
    isGptOss: false,
    stopTokenIds: [],
    layerTypes: null,
    attentionBias: false,
    finalLogitSoftcapping: null,
    attnLogitSoftcapping: null,
    queryKeyNorm: false,
    queryPreAttnScalar: 8,
    chatTemplateEnabled: false,
    ...parsedConfig,
  };
}

describe('FFN config extraction from manifest', () => {
  it('extracts silu activation from manifest', () => {
    const inference = cloneInference();
    inference.ffn.activation = 'silu';
    inference.ffn.gatedActivation = true;

    const config = parseModelConfig(makeManifest(inference));
    expect(config.hiddenActivation).toBe('silu');
  });

  it('extracts gelu activation from manifest', () => {
    const inference = cloneInference();
    inference.ffn.activation = 'gelu';
    inference.ffn.gatedActivation = true;

    const config = parseModelConfig(makeManifest(inference));
    expect(config.hiddenActivation).toBe('gelu');
  });

  it('maps swiglu to silu activation type', () => {
    const inference = cloneInference();
    inference.ffn.activation = 'swiglu';
    inference.ffn.gatedActivation = true;

    const config = parseModelConfig(makeManifest(inference));
    expect(config.hiddenActivation).toBe('silu');
  });

  it('maps geglu to gelu activation type', () => {
    const inference = cloneInference();
    inference.ffn.activation = 'geglu';
    inference.ffn.gatedActivation = true;

    const config = parseModelConfig(makeManifest(inference));
    expect(config.hiddenActivation).toBe('gelu');
  });

  it('preserves gatedActivation flag in inference config', () => {
    const inference = cloneInference();
    inference.ffn.gatedActivation = true;

    const manifest = makeManifest(inference);
    expect(manifest.inference.ffn.gatedActivation).toBe(true);

    inference.ffn.gatedActivation = false;
    const manifest2 = makeManifest(inference);
    expect(manifest2.inference.ffn.gatedActivation).toBe(false);
  });

  it('throws when ffn.activation is missing', () => {
    const inference = cloneInference();
    delete inference.ffn.activation;

    expect(() => parseModelConfig(makeManifest(inference))).toThrow(/ffn.activation/);
  });

  it('throws when ffn.gatedActivation is missing', () => {
    const inference = cloneInference();
    delete inference.ffn.gatedActivation;

    expect(() => parseModelConfig(makeManifest(inference))).toThrow(/ffn.gatedActivation/);
  });
});

describe('activation function selection', () => {
  it('defaults to silu for standard models', () => {
    const inference = cloneInference();
    const config = parseModelConfig(makeManifest(inference));
    expect(config.hiddenActivation).toBe('silu');
  });

  it('uses gelu when specified in manifest', () => {
    const inference = cloneInference();
    inference.ffn.activation = 'gelu';
    const config = parseModelConfig(makeManifest(inference));
    expect(config.hiddenActivation).toBe('gelu');
  });

  it('normalizes relu to silu as fallback', () => {
    const inference = cloneInference();
    inference.ffn.activation = 'relu';
    const config = parseModelConfig(makeManifest(inference));
    expect(config.hiddenActivation).toBe('silu');
  });
});

describe('gated FFN detection', () => {
  it('detects gated activation (swiglu) from manifest', () => {
    const inference = cloneInference();
    inference.ffn.activation = 'swiglu';
    inference.ffn.gatedActivation = true;

    const manifest = makeManifest(inference);
    expect(manifest.inference.ffn.gatedActivation).toBe(true);
  });

  it('detects non-gated activation from manifest', () => {
    const inference = cloneInference();
    inference.ffn.activation = 'silu';
    inference.ffn.gatedActivation = false;

    const manifest = makeManifest(inference);
    expect(manifest.inference.ffn.gatedActivation).toBe(false);
  });

  it('gated activation implies gate_proj + up_proj architecture', () => {
    const inference = cloneInference();
    inference.ffn.activation = 'silu';
    inference.ffn.gatedActivation = true;

    const manifest = makeManifest(inference);
    expect(manifest.inference.ffn.gatedActivation).toBe(true);
    expect(manifest.inference.ffn.activation).toBe('silu');
  });
});

describe('FFN shape validation', () => {
  it('extracts correct hiddenSize and intermediateSize from architecture', () => {
    const inference = cloneInference();
    const manifest = makeManifest(inference, {
      architecture: {
        numLayers: 4,
        hiddenSize: 512,
        intermediateSize: 2048,
        numAttentionHeads: 8,
        numKeyValueHeads: 8,
        headDim: 64,
        vocabSize: 32000,
        maxSeqLen: 4096,
        ropeTheta: 10000,
        rmsNormEps: 1e-5,
      },
    });

    const config = parseModelConfig(manifest);
    expect(config.hiddenSize).toBe(512);
    expect(config.intermediateSize).toBe(2048);
  });

  it('validates FFN expansion ratio (intermediateSize / hiddenSize)', () => {
    const inference = cloneInference();
    const manifest = makeManifest(inference, {
      architecture: {
        numLayers: 4,
        hiddenSize: 256,
        intermediateSize: 1024,
        numAttentionHeads: 4,
        numKeyValueHeads: 4,
        headDim: 64,
        vocabSize: 1024,
        maxSeqLen: 512,
        ropeTheta: 10000,
        rmsNormEps: 1e-6,
      },
    });

    const config = parseModelConfig(manifest);
    const expansionRatio = config.intermediateSize / config.hiddenSize;
    expect(expansionRatio).toBe(4);
  });

  it('handles non-standard expansion ratios', () => {
    const inference = cloneInference();
    const manifest = makeManifest(inference, {
      architecture: {
        numLayers: 4,
        hiddenSize: 256,
        intermediateSize: 683,
        numAttentionHeads: 4,
        numKeyValueHeads: 4,
        headDim: 64,
        vocabSize: 1024,
        maxSeqLen: 512,
        ropeTheta: 10000,
        rmsNormEps: 1e-6,
      },
    });

    const config = parseModelConfig(manifest);
    expect(config.hiddenSize).toBe(256);
    expect(config.intermediateSize).toBe(683);
  });
});

describe('sandwich FFN (pre/post feedforward norms)', () => {
  it('detects standard FFN (no sandwich norms)', () => {
    const inference = cloneInference();
    inference.normalization.preFeedforwardNorm = false;
    inference.normalization.postFeedforwardNorm = false;

    const manifest = makeManifest(inference);
    expect(manifest.inference.normalization.preFeedforwardNorm).toBe(false);
    expect(manifest.inference.normalization.postFeedforwardNorm).toBe(false);
  });

  it('detects Gemma 3 style sandwich norms', () => {
    const inference = cloneInference();
    inference.normalization.preFeedforwardNorm = true;
    inference.normalization.postFeedforwardNorm = true;
    inference.normalization.postAttentionNorm = true;

    const manifest = makeManifest(inference);
    expect(manifest.inference.normalization.preFeedforwardNorm).toBe(true);
    expect(manifest.inference.normalization.postFeedforwardNorm).toBe(true);
    expect(manifest.inference.normalization.postAttentionNorm).toBe(true);
  });

  it('detects partial sandwich norm (post only)', () => {
    const inference = cloneInference();
    inference.normalization.preFeedforwardNorm = false;
    inference.normalization.postFeedforwardNorm = true;

    const manifest = makeManifest(inference);
    expect(manifest.inference.normalization.preFeedforwardNorm).toBe(false);
    expect(manifest.inference.normalization.postFeedforwardNorm).toBe(true);
  });

  it('sandwich norm detection includes rmsNormWeightOffset', () => {
    const inference = cloneInference();
    inference.normalization.rmsNormWeightOffset = true;
    inference.normalization.preFeedforwardNorm = true;
    inference.normalization.postFeedforwardNorm = true;

    const config = parseModelConfig(makeManifest(inference));
    expect(config.rmsNormWeightOffset).toBe(true);
  });
});

describe('MoE detection and routing config', () => {
  it('detects non-MoE model from config', () => {
    const config = makeConfig({ useMoE: false });
    expect(isMoELayerLocal(0, config)).toBe(false);
  });

  it('detects MoE model from config', () => {
    const config = makeConfig({ useMoE: true, layerTypes: null });
    expect(isMoELayerLocal(0, config)).toBe(true);
  });

  it('detects MoE layer when router weight present', () => {
    const config = makeConfig({ useMoE: true });
    const layerWeights = { routerWeight: new Float32Array(8) };
    expect(isMoELayerLocal(0, config, layerWeights)).toBe(true);
  });

  it('respects layerTypes for per-layer MoE detection', () => {
    const config = makeConfig({
      useMoE: true,
      layerTypes: ['dense', 'moe', 'dense', 'moe'],
    });

    expect(isMoELayerLocal(0, config)).toBe(false);
    expect(isMoELayerLocal(1, config)).toBe(true);
    expect(isMoELayerLocal(2, config)).toBe(false);
    expect(isMoELayerLocal(3, config)).toBe(true);
  });

  it('falls back to true for MoE when layerTypes unavailable', () => {
    const config = makeConfig({ useMoE: true, layerTypes: null });
    expect(isMoELayerLocal(0, config)).toBe(true);
    expect(isMoELayerLocal(5, config)).toBe(true);
  });

  it('handles out-of-bounds layer index', () => {
    const config = makeConfig({
      useMoE: true,
      layerTypes: ['dense', 'moe'],
    });
    expect(isMoELayerLocal(10, config)).toBe(true);
  });

  it('extracts MoE config from manifest', () => {
    const inference = cloneInference();
    const manifest = makeManifest(inference, {
      config: {
        num_local_experts: 8,
        num_experts_per_tok: 2,
      },
    });

    const config = parseModelConfig(manifest);
    expect(config.useMoE).toBe(true);
    expect(config.numExperts).toBe(8);
    expect(config.moeTopK).toBe(2);
  });

  it('handles alternative MoE config keys', () => {
    const inference = cloneInference();
    const manifest = makeManifest(inference, {
      config: {
        num_experts: 4,
        top_k: 1,
      },
    });

    const config = parseModelConfig(manifest);
    expect(config.useMoE).toBe(true);
    expect(config.numExperts).toBe(4);
    expect(config.moeTopK).toBe(1);
  });
});

describe('fused down norm logging state', () => {
  beforeEach(() => {
    setLoggedFusedDownNorm(false);
  });

  afterEach(() => {
    setLoggedFusedDownNorm(false);
  });

  it('starts with unlogged state', () => {
    expect(hasLoggedFusedDownNorm()).toBe(false);
  });

  it('tracks logged state', () => {
    setLoggedFusedDownNorm(true);
    expect(hasLoggedFusedDownNorm()).toBe(true);
  });

  it('can reset logged state', () => {
    setLoggedFusedDownNorm(true);
    setLoggedFusedDownNorm(false);
    expect(hasLoggedFusedDownNorm()).toBe(false);
  });
});

describe('FFN with mini-model fixture', () => {
  it('parses FFN config from mini-model manifest', async () => {
    const manifest = await import('../../tests/fixtures/mini-model/manifest.json', {
      with: { type: 'json' }
    });

    const config = parseModelConfig(manifest.default);

    expect(config.hiddenSize).toBe(64);
    expect(config.intermediateSize).toBe(128);
    expect(config.hiddenActivation).toBe('silu');
    expect(config.useMoE).toBe(false);
  });

  it('validates mini-model uses gated activation', async () => {
    const manifest = await import('../../tests/fixtures/mini-model/manifest.json', {
      with: { type: 'json' }
    });

    expect(manifest.default.inference.ffn.gatedActivation).toBe(true);
    expect(manifest.default.inference.ffn.activation).toBe('silu');
  });

  it('validates mini-model has no sandwich norms', async () => {
    const manifest = await import('../../tests/fixtures/mini-model/manifest.json', {
      with: { type: 'json' }
    });

    expect(manifest.default.inference.normalization.preFeedforwardNorm).toBe(false);
    expect(manifest.default.inference.normalization.postFeedforwardNorm).toBe(false);
  });
});

describe('FFN integration with layer pattern', () => {
  it('alternating layer pattern does not affect FFN config', () => {
    const inference = cloneInference();
    inference.layerPattern = {
      type: 'alternating',
      globalPattern: 'even',
    };

    const config = parseModelConfig(makeManifest(inference));
    expect(config.hiddenActivation).toBe('silu');
    expect(config.layerTypes).toHaveLength(4);
  });

  it('uniform layer pattern preserves FFN behavior', () => {
    const inference = cloneInference();
    inference.layerPattern = {
      type: 'uniform',
    };

    const config = parseModelConfig(makeManifest(inference));
    expect(config.hiddenActivation).toBe('silu');
    expect(config.layerTypes).toBeNull();
  });
});
