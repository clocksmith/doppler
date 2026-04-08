import assert from 'node:assert/strict';
import { createExecutionV1Session } from '../helpers/execution-v1-fixtures.js';

const {
  parseModelConfigFromManifest,
  resolveLayerIntermediateSize,
} = await import('../../src/inference/pipelines/text/config.js');

function createGemma4E2BManifest(useDoubleWideMlp) {
  return {
    modelId: `gemma4-double-wide-${useDoubleWideMlp ? 'enabled' : 'disabled'}`,
    modelType: 'text',
    quantization: 'Q4_K_M',
    architecture: {
      hiddenSize: 1536,
      numLayers: 35,
      numAttentionHeads: 8,
      numKeyValueHeads: 1,
      headDim: 256,
      globalHeadDim: 512,
      intermediateSize: 6144,
      vocabSize: 262144,
      maxSeqLen: 131072,
      ropeTheta: 1000000,
      hiddenSizePerLayerInput: 256,
      vocabSizePerLayerInput: 262144,
      numKvSharedLayers: 20,
    },
    eos_token_id: 1,
    inference: {
      attention: {
        queryPreAttnScalar: 1,
        queryKeyNorm: true,
        valueNorm: true,
        attentionBias: false,
        causal: true,
        slidingWindow: 512,
        attnLogitSoftcapping: null,
        attentionOutputGate: false,
      },
      normalization: {
        rmsNormWeightOffset: false,
        rmsNormEps: 1e-6,
        postAttentionNorm: true,
        preFeedforwardNorm: true,
        postFeedforwardNorm: true,
      },
      ffn: {
        activation: 'gelu',
        gatedActivation: true,
        useDoubleWideMlp,
        swigluLimit: null,
      },
      rope: {
        ropeTheta: 1000000,
        ropeLocalTheta: 10000,
        ropeInterleaved: false,
        mropeInterleaved: false,
        mropeSection: null,
        partialRotaryFactor: 0.25,
        ropeLocalPartialRotaryFactor: null,
        ropeFrequencyBaseDim: 512,
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
        tieWordEmbeddings: true,
        scaleEmbeddings: true,
        embeddingTranspose: false,
        finalLogitSoftcapping: 30,
        embeddingVocabSize: null,
        embeddingPostprocessor: null,
      },
      layerPattern: {
        type: 'every_n',
        globalPattern: null,
        period: 5,
        offset: 4,
        layerTypes: null,
      },
      chatTemplate: {
        type: 'gemma4',
        enabled: true,
      },
      pipeline: null,
      session: createExecutionV1Session(),
      execution: null,
    },
    tensors: {
      'model.language_model.layers.0.mlp.gate_proj.weight': { shape: [6144, 1536] },
      'model.language_model.layers.0.mlp.up_proj.weight': { shape: [6144, 1536] },
      'model.language_model.layers.0.mlp.down_proj.weight': { shape: [1536, 6144] },
      'model.language_model.layers.15.mlp.gate_proj.weight': { shape: [12288, 1536] },
      'model.language_model.layers.15.mlp.up_proj.weight': { shape: [12288, 1536] },
      'model.language_model.layers.15.mlp.down_proj.weight': { shape: [1536, 12288] },
    },
  };
}

{
  const parsed = parseModelConfigFromManifest(createGemma4E2BManifest(true));

  assert.equal(parsed.useDoubleWideMlp, true);
  assert.equal(parsed.maxIntermediateSize, 12288);
  assert.equal(resolveLayerIntermediateSize(parsed, 0), 6144);
  assert.equal(resolveLayerIntermediateSize(parsed, 14), 6144);
  assert.equal(resolveLayerIntermediateSize(parsed, 15), 12288);
  assert.equal(resolveLayerIntermediateSize(parsed, 34), 12288);
}

{
  assert.throws(
    () => parseModelConfigFromManifest(createGemma4E2BManifest(false)),
    /layer 15 gate weight shape \[12288, 1536\] does not match the resolved FFN contract \[6144, 1536\]/i
  );
}

console.log('gemma4-double-wide-mlp-runtime.test: ok');
