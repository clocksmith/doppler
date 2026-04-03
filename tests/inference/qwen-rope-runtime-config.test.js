import assert from 'node:assert/strict';

import { initRoPEFrequencies } from '../../src/inference/pipelines/text/init.js';
import { parseModelConfigFromManifest } from '../../src/inference/pipelines/text/config.js';

{
  const freqs = await initRoPEFrequencies({
    headDim: 256,
    rotaryDim: 64,
    ropeFrequencyBaseDim: 64,
    ropeLocalFrequencyBaseDim: 256,
    maxSeqLen: 8,
    ropeTheta: 10000000,
    ropeLocalTheta: null,
    mropeInterleaved: true,
    mropeSection: [11, 11, 10],
    partialRotaryFactor: 0.25,
    ropeScale: 1,
    ropeLocalScale: 1,
    ropeScalingType: null,
    ropeLocalScalingType: null,
    ropeScaling: null,
    ropeLocalScaling: null,
  }, false);

  assert.equal(freqs.cos.length, 8 * 32);
  assert.equal(freqs.sin.length, 8 * 32);
}

{
  await assert.rejects(
    initRoPEFrequencies({
      headDim: 256,
      rotaryDim: 64,
      ropeFrequencyBaseDim: 64,
      ropeLocalFrequencyBaseDim: 256,
      maxSeqLen: 8,
      ropeTheta: 10000000,
      ropeLocalTheta: null,
      mropeInterleaved: true,
      mropeSection: [10, 10, 10],
      partialRotaryFactor: 0.25,
      ropeScale: 1,
      ropeLocalScale: 1,
      ropeScalingType: null,
      ropeLocalScalingType: null,
      ropeScaling: null,
      ropeLocalScaling: null,
    }, false),
    /mropeSection expands to 60 dims, but rotaryDim is 64/
  );
}

// Standard RoPE (no MRoPE): full rotary dim (headDim=256)
{
  const freqs = await initRoPEFrequencies({
    headDim: 256,
    rotaryDim: undefined,
    ropeFrequencyBaseDim: 256,
    ropeLocalFrequencyBaseDim: 256,
    maxSeqLen: 8,
    ropeTheta: 10000000,
    ropeLocalTheta: null,
    mropeInterleaved: false,
    mropeSection: null,
    partialRotaryFactor: null,
    ropeScale: 1,
    ropeLocalScale: 1,
    ropeScalingType: null,
    ropeLocalScalingType: null,
    ropeScaling: null,
    ropeLocalScaling: null,
  }, false);

  // Full headDim=256 → halfDim=128 → cos/sin length = 8 * 128
  assert.equal(freqs.cos.length, 8 * 128);
  assert.equal(freqs.sin.length, 8 * 128);
}

{
  const freqs = await initRoPEFrequencies({
    headDim: 8,
    rotaryDim: undefined,
    ropeFrequencyBaseDim: 8,
    ropeLocalFrequencyBaseDim: 8,
    maxSeqLen: 4,
    ropeTheta: 10000,
    ropeLocalTheta: 10000,
    mropeInterleaved: false,
    mropeSection: null,
    partialRotaryFactor: null,
    ropeScale: 4,
    ropeLocalScale: 1,
    ropeScalingType: 'yarn',
    ropeLocalScalingType: null,
    ropeScaling: {
      factor: 4,
      beta_fast: 32,
      beta_slow: 1,
      original_max_position_embeddings: 1,
    },
    ropeLocalScaling: null,
  }, false);

  assert.ok(freqs.localCos instanceof Float32Array);
  assert.ok(freqs.localSin instanceof Float32Array);
  assert.ok(Math.abs(freqs.localCos[4] - Math.cos(1)) < 1e-6);
  assert.ok(Math.abs(freqs.localSin[4] - Math.sin(1)) < 1e-6);
}

{
  const parsed = parseModelConfigFromManifest({
    modelId: 'qwen-mrope-contract-fixture',
    modelType: 'transformer',
    eos_token_id: [1],
    architecture: {
      numLayers: 1,
      hiddenSize: 1024,
      intermediateSize: 3584,
      numAttentionHeads: 8,
      numKeyValueHeads: 2,
      headDim: 256,
      vocabSize: 248320,
      maxSeqLen: 262144,
      linearNumKeyHeads: 16,
      linearNumValueHeads: 16,
      linearKeyHeadDim: 128,
      linearValueHeadDim: 128,
      linearConvKernelDim: 4,
      linearNormMode: 'shared',
    },
    inference: {
      schema: null,
      attention: {
        queryPreAttnScalar: 256,
        attnLogitSoftcapping: null,
        slidingWindow: null,
        queryKeyNorm: true,
        attentionOutputGate: true,
        causal: true,
        attentionBias: false,
      },
      normalization: {
        rmsNormEps: 1e-6,
        rmsNormWeightOffset: false,
        postAttentionNorm: true,
        preFeedforwardNorm: false,
        postFeedforwardNorm: false,
      },
      ffn: {
        activation: 'silu',
        gatedActivation: false,
        swigluLimit: null,
      },
      rope: {
        ropeTheta: 10000000,
        ropeLocalTheta: null,
        ropeScalingType: null,
        ropeScalingFactor: 1,
        ropeFrequencyBaseDim: null,
        ropeLocalFrequencyBaseDim: null,
        yarnBetaFast: null,
        yarnBetaSlow: null,
        yarnOriginalMaxPos: null,
        ropeLocalScalingType: null,
        ropeLocalScalingFactor: 1,
        ropeLocalYarnBetaFast: null,
        ropeLocalYarnBetaSlow: null,
        ropeLocalYarnOriginalMaxPos: null,
        mropeInterleaved: true,
        mropeSection: [11, 11, 10],
        partialRotaryFactor: 0.25,
        ropeLocalPartialRotaryFactor: null,
      },
      output: {
        finalLogitSoftcapping: null,
        tieWordEmbeddings: true,
        scaleEmbeddings: false,
        embeddingTranspose: false,
        embeddingVocabSize: 248320,
        embeddingPostprocessor: null,
      },
      layerPattern: {
        type: 'custom',
        globalPattern: null,
        period: null,
        offset: null,
        layerTypes: ['full_attention'],
      },
      chatTemplate: {
        type: 'qwen',
        enabled: true,
      },
      session: {
        decodeLoop: {
          batchSize: 4,
          disableCommandBatching: true,
        },
      },
      execution: null,
      defaultKernelPath: null,
      pipeline: null,
    },
    quantization: 'f16',
  });

  assert.equal(parsed.mropeInterleaved, true);
  assert.equal(parsed.ropeFrequencyBaseDim, 64);
  assert.equal(parsed.ropeLocalFrequencyBaseDim, 256);
  assert.equal(
    parsed.ropeInterleaved,
    false,
    'Qwen mRoPE interleaving must not force adjacent-pair RoPE rotation.'
  );
}

console.log('qwen-rope-runtime-config.test: ok');
