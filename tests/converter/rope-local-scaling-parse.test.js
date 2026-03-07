import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { parseModelConfig } = await import('../../src/inference/pipelines/text/config.js');

const manifest = {
  modelId: 'rope-local-scaling-parse-test',
  eos_token_id: 1,
  quantization: 'F16',
  architecture: {
    numLayers: 2,
    hiddenSize: 64,
    intermediateSize: 256,
    numAttentionHeads: 2,
    numKeyValueHeads: 1,
    headDim: 32,
    vocabSize: 1024,
    maxSeqLen: 256,
    ropeTheta: 10000,
    rmsNormEps: 1e-6,
  },
  inference: {
    attention: {
      queryPreAttnScalar: 8,
      attnLogitSoftcapping: null,
      slidingWindow: 128,
      queryKeyNorm: true,
      attentionBias: false,
      causal: true,
    },
    normalization: {
      rmsNormEps: 1e-6,
      rmsNormWeightOffset: true,
      postAttentionNorm: true,
      preFeedforwardNorm: true,
      postFeedforwardNorm: true,
    },
    ffn: {
      activation: 'gelu',
      gatedActivation: true,
      swigluLimit: null,
    },
    rope: {
      ropeTheta: 10000,
      ropeLocalTheta: 10000,
      ropeScalingType: 'linear',
      ropeScalingFactor: 8.0,
      mropeInterleaved: false,
      mropeSection: null,
      partialRotaryFactor: null,
      ropeLocalScalingType: 'linear',
      ropeLocalScalingFactor: 4.0,
      yarnBetaFast: null,
      yarnBetaSlow: null,
      yarnOriginalMaxPos: null,
      ropeLocalYarnBetaFast: null,
      ropeLocalYarnBetaSlow: null,
      ropeLocalYarnOriginalMaxPos: null,
    },
    output: {
      finalLogitSoftcapping: null,
      tieWordEmbeddings: true,
      scaleEmbeddings: true,
      embeddingTranspose: false,
      embeddingVocabSize: 1024,
    },
    layerPattern: {
      type: 'uniform',
      globalPattern: null,
      period: null,
      offset: null,
    },
    chatTemplate: {
      type: 'gemma',
      enabled: true,
    },
    pipeline: null,
    defaultKernelPath: null,
  },
  config: {
    model_type: 'gemma3_text',
  },
};

const parsed = parseModelConfig(manifest, null);

assert.equal(parsed.ropeScale, 8.0);
assert.equal(parsed.ropeScalingType, 'linear');
assert.equal(parsed.ropeLocalScale, 4.0);
assert.equal(parsed.ropeLocalScalingType, 'linear');
assert.equal(parsed.ropeScaling?.factor, 8.0);
assert.equal(parsed.ropeLocalScaling?.factor, 4.0);

console.log('rope-local-scaling-parse.test: ok');
