import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { parseModelConfig } = await import('../../src/inference/pipelines/text/config.js');

const manifest = {
  modelId: 'gemma3-layer-pattern-test',
  eos_token_id: 1,
  quantization: 'F16',
  architecture: {
    numLayers: 18,
    hiddenSize: 640,
    intermediateSize: 2048,
    numAttentionHeads: 4,
    numKeyValueHeads: 1,
    headDim: 256,
    vocabSize: 262144,
    maxSeqLen: 32768,
    ropeTheta: 1000000,
    rmsNormEps: 1e-6,
  },
  inference: {
    attention: {
      queryPreAttnScalar: 256,
      attnLogitSoftcapping: null,
      slidingWindow: 512,
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
      ropeTheta: 1000000,
      ropeLocalTheta: 10000,
      ropeScalingType: null,
      ropeScalingFactor: 1.0,
      yarnBetaFast: null,
      yarnBetaSlow: null,
      yarnOriginalMaxPos: null,
    },
    output: {
      finalLogitSoftcapping: null,
      tieWordEmbeddings: true,
      scaleEmbeddings: true,
      embeddingTranspose: false,
      embeddingVocabSize: 262144,
    },
    layerPattern: {
      type: 'every_n',
      globalPattern: null,
      period: 6,
      offset: 0,
    },
    chatTemplate: {
      type: 'gemma',
      enabled: true,
    },
    defaultKernelPath: 'gemma3-f16-f16a',
  },
  config: {
    model_type: 'gemma3_text',
  },
};

const parsed = parseModelConfig(manifest, null);
const firstTwelve = parsed.layerTypes.slice(0, 12);

assert.deepEqual(firstTwelve, [
  'full_attention',
  'sliding_attention',
  'sliding_attention',
  'sliding_attention',
  'sliding_attention',
  'sliding_attention',
  'full_attention',
  'sliding_attention',
  'sliding_attention',
  'sliding_attention',
  'sliding_attention',
  'sliding_attention',
]);

const offsetManifest = {
  ...manifest,
  modelId: 'gemma3-layer-pattern-offset-test',
  inference: {
    ...manifest.inference,
    layerPattern: {
      ...manifest.inference.layerPattern,
      offset: 5,
    },
  },
};

const parsedOffset = parseModelConfig(offsetManifest, null);
const firstTwelveOffset = parsedOffset.layerTypes.slice(0, 12);

assert.deepEqual(firstTwelveOffset, [
  'sliding_attention',
  'sliding_attention',
  'sliding_attention',
  'sliding_attention',
  'sliding_attention',
  'full_attention',
  'sliding_attention',
  'sliding_attention',
  'sliding_attention',
  'sliding_attention',
  'sliding_attention',
  'full_attention',
]);

console.log('layer-pattern-every-n.test: ok');
