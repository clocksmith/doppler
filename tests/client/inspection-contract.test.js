import assert from 'node:assert/strict';

import {
  aggregateWordPerplexity,
  assertComparableFingerprints,
  buildComparisonFingerprint,
  resolveObservationPolicy,
} from '../../src/client/inspection.js';

const alwaysOn = resolveObservationPolicy('demo/always-on');
assert.equal(alwaysOn.modifiesExecution, false);
assert.equal(alwaysOn.performanceRepresentative, true);
assert.equal(alwaysOn.gpuTimestampQueries, false);

const guided = resolveObservationPolicy('demo/guided-quality');
assert.equal(guided.modifiesExecution, true);
assert.equal(guided.performanceRepresentative, false);
assert.equal(guided.perplexity.wordSegmentation, 'doppler.word-segmentation/unicode-whitespace-v1');

function fingerprint(tokenizer, policyId = 'demo/always-on') {
  return buildComparisonFingerprint({
    artifact: {
      modelId: 'model-a',
      manifestHash: `sha256:${'a'.repeat(64)}`,
    },
    tokenizer,
    promptTokenIds: [1, 2, 3],
    sampling: { temperature: 0, topK: 1 },
    observationPolicyId: policyId,
    execution: {
      backend: 'webgpu',
      executionPlanId: 'plan-a',
      kernelPathId: 'kernel-a',
    },
    browser: {
      userAgent: 'browser-a',
      platform: 'platform-a',
      language: 'en',
    },
    adapter: {
      vendor: 'vendor-a',
      architecture: 'arch-a',
      device: 'device-a',
      description: 'adapter-a',
    },
  });
}

const first = fingerprint({ type: 'sentencepiece', digest: 'tokenizer-a' });
const second = fingerprint({ type: 'sentencepiece', digest: 'tokenizer-a' });
const rawDigest = buildComparisonFingerprint({
  ...first.identity,
  artifact: {
    modelId: 'model-a',
    manifestHash: 'a'.repeat(64),
  },
  tokenizer: { type: 'sentencepiece', digest: 'tokenizer-a' },
  observationPolicyId: 'demo/always-on',
});
assert.equal(rawDigest.identity.artifact.manifestHash, `sha256:${'a'.repeat(64)}`);
assert.equal(assertComparableFingerprints('quality', first, second), true);
assert.equal(assertComparableFingerprints('performance', first, second), true);
assert.throws(
  () => assertComparableFingerprints(
    'quality',
    first,
    fingerprint({ type: 'bpe', digest: 'tokenizer-b' })
  ),
  /tokenizer identities differ/
);
assert.throws(
  () => assertComparableFingerprints(
    'performance',
    fingerprint({ type: 'sentencepiece' }, 'demo/guided-quality'),
    fingerprint({ type: 'sentencepiece' }, 'demo/guided-quality')
  ),
  /not representative/
);

const quality = aggregateWordPerplexity([
  { index: 0, text: 'hello', surprisal: 0.2 },
  { index: 1, text: ' world', surprisal: 0.5 },
  { index: 2, text: '!', surprisal: 0.1 },
], {
  windowUnit: 'words',
  windowSize: 2,
});
assert.equal(quality.words.length, 2);
assert.equal(quality.words[0].text, 'hello');
assert.equal(quality.words[1].text, 'world!');
assert.equal(quality.words[1].summedSurprisal, 0.6);
assert.equal(quality.words[1].rollingWindow.size, 2);

console.log('inspection-contract.test: ok');
