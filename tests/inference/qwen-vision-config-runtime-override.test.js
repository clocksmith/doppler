import assert from 'node:assert/strict';

import { parseModelConfigFromManifest } from '../../src/inference/pipelines/text/config.js';

const manifest = {
  modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
  modelType: 'qwen3',
  quantization: 'Q4_K_M',
  eos_token_id: 248044,
  image_token_id: 248056,
  architecture: {
    numLayers: 24,
    hiddenSize: 1024,
    intermediateSize: 4096,
    numAttentionHeads: 8,
    numKeyValueHeads: 8,
    headDim: 128,
    vocabSize: 248320,
    maxSeqLen: 8192,
  },
  inference: {
    schema: 'doppler.execution/v1',
    attention: {
      queryPreAttnScalar: 1,
      attnLogitSoftcapping: null,
      slidingWindow: null,
      queryKeyNorm: false,
      valueNorm: false,
      causal: true,
      attentionBias: false,
      attentionOutputGate: false,
    },
    normalization: {
      rmsNormEps: 1e-6,
      rmsNormWeightOffset: true,
      postAttentionNorm: true,
      preFeedforwardNorm: true,
      postFeedforwardNorm: false,
    },
    ffn: {
      activation: 'silu',
      gatedActivation: true,
      useDoubleWideMlp: false,
      swigluLimit: null,
    },
    rope: {
      ropeTheta: 1000000,
      ropeLocalTheta: null,
      ropeInterleaved: false,
      mropeInterleaved: false,
      mropeSection: null,
      partialRotaryFactor: 1,
      ropeLocalPartialRotaryFactor: null,
      ropeFrequencyBaseDim: 128,
      ropeLocalFrequencyBaseDim: null,
      ropeScalingType: null,
      ropeScalingFactor: 1,
      ropeLocalScalingType: null,
      ropeLocalScalingFactor: 1,
      yarnBetaFast: null,
      yarnBetaSlow: null,
      yarnOriginalMaxPos: null,
      ropeLocalYarnBetaFast: null,
      ropeLocalYarnBetaSlow: null,
      ropeLocalYarnOriginalMaxPos: null,
    },
    output: {
      finalLogitSoftcapping: null,
      tieWordEmbeddings: false,
      scaleEmbeddings: false,
      embeddingTranspose: false,
      embeddingVocabSize: 248320,
      embeddingPostprocessor: null,
    },
    layerPattern: {
      type: 'global',
      globalPattern: 'attention',
      period: null,
      offset: null,
      layerTypes: null,
    },
    chatTemplate: {
      type: 'qwen',
      enabled: true,
    },
    pipeline: null,
  },
};

const modelConfig = parseModelConfigFromManifest(manifest, {
  vision_config: {
    vision_architecture: 'qwen3vl',
    depth: 12,
    hidden_size: 768,
    intermediate_size: 3072,
    num_heads: 12,
    num_key_value_heads: 12,
    head_dim: 64,
    out_hidden_size: 1024,
    patch_size: 16,
    pooling_kernel_size: 2,
    spatial_merge_size: 2,
    temporal_patch_size: 2,
    eps: 1e-6,
    hidden_activation: 'gelu_pytorch_tanh',
    min_pixels: 65536,
    max_pixels: 16777216,
    normalization: {
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
    },
  },
});

assert.deepEqual(modelConfig.visionConfig.normalization, {
  mean: [0.5, 0.5, 0.5],
  std: [0.5, 0.5, 0.5],
});
assert.equal(modelConfig.visionConfig.spatialMergeSize, 2);
assert.equal(modelConfig.visionConfig.minPixels, 65536);
assert.equal(modelConfig.visionConfig.maxPixels, 16777216);
assert.equal(modelConfig.visionConfig.visionArchitecture, 'qwen3vl');

console.log('qwen-vision-config-runtime-override.test: ok');
