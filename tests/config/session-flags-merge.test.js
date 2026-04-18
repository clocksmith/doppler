import assert from 'node:assert/strict';
import { mergeConfig } from '../../src/config/merge.js';

// Minimal manifest with just enough inference structure to pass merge.
function buildManifest(sessionOverrides = {}, largeWeightsOverrides = undefined) {
  return {
    modelId: 'session-flags-witness',
    inference: {
      attention: {
        queryPreAttnScalar: 1,
        attentionBias: false,
        attnLogitSoftcapping: null,
        slidingWindow: 4096,
        queryKeyNorm: false,
        valueNorm: false,
        attentionOutputGate: false,
        causal: true,
      },
      normalization: {
        rmsNormEps: 1e-6,
        rmsNormWeightOffset: 0,
        postAttentionNorm: true,
        preFeedforwardNorm: true,
        postFeedforwardNorm: false,
      },
      ffn: { activation: 'gelu', gatedActivation: false, swigluLimit: null },
      rope: {
        ropeTheta: 1000000, ropeLocalTheta: null, ropeFrequencyBaseDim: null,
        ropeLocalFrequencyBaseDim: null, ropeScalingType: null, ropeScalingFactor: null,
        ropeLocalScalingType: null, ropeLocalScalingFactor: null, yarnBetaFast: null,
        yarnBetaSlow: null, yarnOriginalMaxPos: null, ropeLocalYarnBetaFast: null,
        ropeLocalYarnBetaSlow: null, ropeLocalYarnOriginalMaxPos: null,
      },
      output: {
        finalLogitSoftcapping: null, tieWordEmbeddings: false, scaleEmbeddings: false,
        embeddingTranspose: false, embeddingVocabSize: 0, embeddingPostprocessor: null,
      },
      session: sessionOverrides,
      ...(largeWeightsOverrides !== undefined ? { largeWeights: largeWeightsOverrides } : {}),
      pipeline: null,
      layerPattern: null,
      chatTemplate: { type: 'gemma', enabled: true },
    },
    architecture: { headDim: 64, maxSeqLen: 2048 },
  };
}

// Case 1: manifest sets useFlashPrefillAttention=true, no runtime override → merged value is true, source=manifest.
{
  const merged = mergeConfig(
    buildManifest({ useFlashPrefillAttention: true }),
    {}
  );
  assert.equal(merged.inference.session.useFlashPrefillAttention, true);
  assert.equal(merged._sources.get('inference.session.useFlashPrefillAttention'), 'manifest');
}

// Case 2: manifest sets useFlashPrefillAttention=true, runtime overrides to false → runtime wins, source=runtime.
{
  const merged = mergeConfig(
    buildManifest({ useFlashPrefillAttention: true }),
    { session: { useFlashPrefillAttention: false } }
  );
  assert.equal(merged.inference.session.useFlashPrefillAttention, false);
  assert.equal(merged._sources.get('inference.session.useFlashPrefillAttention'), 'runtime');
}

// Case 3: manifest does not set prefillChunkSubmitMode, runtime sets 'async' → merged='async', source=runtime.
{
  const merged = mergeConfig(
    buildManifest({}),
    { session: { prefillChunkSubmitMode: 'async' } }
  );
  assert.equal(merged.inference.session.prefillChunkSubmitMode, 'async');
  assert.equal(merged._sources.get('inference.session.prefillChunkSubmitMode'), 'runtime');
}

// Case 4: per-field overlay — manifest sets decodeLoop, runtime sets different session flag, both survive.
{
  const merged = mergeConfig(
    buildManifest({
      decodeLoop: { batchSize: 8, readbackInterval: 8 },
      useFlashPrefillAttention: true,
    }),
    { session: { retainQ4KMaterialization: true } }
  );
  assert.equal(merged.inference.session.decodeLoop.batchSize, 8, 'manifest decodeLoop preserved');
  assert.equal(merged.inference.session.useFlashPrefillAttention, true, 'manifest flag preserved');
  assert.equal(merged.inference.session.retainQ4KMaterialization, true, 'runtime flag applied');
  assert.equal(merged._sources.get('inference.session.decodeLoop'), 'manifest');
  assert.equal(merged._sources.get('inference.session.useFlashPrefillAttention'), 'manifest');
  assert.equal(merged._sources.get('inference.session.retainQ4KMaterialization'), 'runtime');
}

// Case 5: inference.largeWeights is merged as a sibling of session (not nested under session).
{
  const merged = mergeConfig(
    buildManifest({}, { gpuResidentOverrides: ['a.b.weight'] }),
    {}
  );
  assert.deepEqual(merged.inference.largeWeights.gpuResidentOverrides, ['a.b.weight']);
  assert.equal(merged._sources.get('inference.largeWeights.gpuResidentOverrides'), 'manifest');
}

// Case 6: runtime largeWeights override wins over manifest.
{
  const merged = mergeConfig(
    buildManifest({}, { gpuResidentOverrides: ['manifest.weight'] }),
    { largeWeights: { gpuResidentOverrides: ['runtime.weight'] } }
  );
  assert.deepEqual(merged.inference.largeWeights.gpuResidentOverrides, ['runtime.weight']);
  assert.equal(merged._sources.get('inference.largeWeights.gpuResidentOverrides'), 'runtime');
}

console.log('session-flags-merge.test: ok');
