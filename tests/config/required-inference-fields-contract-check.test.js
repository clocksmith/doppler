import assert from 'node:assert/strict';

import {
  buildManifestRequiredInferenceFieldsArtifact,
  buildRequiredInferenceFieldsContractArtifact,
} from '../../src/config/required-inference-fields-contract-check.js';

const artifact = buildRequiredInferenceFieldsContractArtifact();

assert.equal(artifact.ok, true);
assert.ok(artifact.checks.every((entry) => entry.ok));
assert.ok(artifact.stats.fieldCases > 20);
assert.ok(artifact.stats.nullableCases > 5);

const manifestArtifact = buildManifestRequiredInferenceFieldsArtifact({
  attention: {
    queryPreAttnScalar: 256,
    queryKeyNorm: true,
    attentionBias: false,
    causal: true,
    slidingWindow: null,
    attnLogitSoftcapping: null,
  },
  normalization: {
    rmsNormWeightOffset: true,
    rmsNormEps: 1e-6,
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
    ropeScalingFactor: 1.0,
    ropeScalingType: null,
    ropeLocalTheta: null,
    mropeInterleaved: false,
    mropeSection: null,
    partialRotaryFactor: null,
    yarnBetaFast: null,
    yarnBetaSlow: null,
    yarnOriginalMaxPos: null,
  },
  output: {
    tieWordEmbeddings: true,
    scaleEmbeddings: true,
    embeddingTranspose: false,
    finalLogitSoftcapping: null,
    embeddingVocabSize: null,
  },
  layerPattern: {
    type: 'every_n',
    globalPattern: null,
    period: 6,
    offset: 0,
  },
  chatTemplate: {
    type: null,
    enabled: true,
  },
  defaultKernelPath: 'unit-test',
}, 'fixture.inference');

assert.equal(manifestArtifact.scope, 'manifest');
assert.equal(manifestArtifact.label, 'fixture.inference');
assert.equal(manifestArtifact.ok, true);

console.log('required-inference-fields-contract-check.test: ok');
