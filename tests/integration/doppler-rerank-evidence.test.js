import assert from 'node:assert/strict';

import { createModelHandle } from '../../src/client/runtime/model-session.js';
import { computeCanonicalSha256 } from '../../src/utils/canonical-hash.js';

const scoring = {
  format: 'qwen3_yes_no_logit',
  instruction: 'Find relevant documents',
  inputTemplate: 'Instruction: {instruction}\nQuery: {query}\nDocument: {document}',
  prefix: '<start>',
  suffix: '<answer>',
  trueToken: 'yes',
  trueTokenId: 7,
  falseToken: 'no',
  falseTokenId: 9,
  score: 'logit_difference',
  probability: 'sigmoid',
};

const pipeline = {
  isLoaded: true,
  manifest: {
    modelId: 'rerank-resolved',
    inference: {
      supportsRerank: true,
      rerank: scoring,
    },
  },
  resolvedRuntimeSession: {
    id: `sha256:${'b'.repeat(64)}`,
  },
  reset() {},
  async prefillWithTokenLogits(prompt, tokenIds, options) {
    assert.deepEqual(tokenIds, [7, 9]);
    assert.equal(options.useChatTemplate, false);
    const relevant = prompt.includes('WebGPU runs compute shaders');
    return {
      tokens: [1, 2, 3],
      logits: Float32Array.from(relevant ? [5, -1] : [-2, 3]),
    };
  },
  getKernelCapabilities() {
    return {
      adapterInfo: { vendor: 'fixture-vendor', device: 'fixture-device' },
      hasF16: true,
      hasSubgroups: false,
      maxBufferSize: 4096,
      deviceEpoch: 2,
    };
  },
  getStats() {
    return { kernelPathId: 'rerank-fixture-path' };
  },
  unload() {},
};

const handle = createModelHandle(pipeline, {
  logicalModelId: 'search-reranker',
  modelId: 'rerank-resolved',
  manifestHash: 'a'.repeat(64),
});

assert.equal(handle.supportsRerank, true);
const evidence = await handle.rerankWithEvidence('What runs compute shaders?', [
  'A recipe for sourdough bread.',
  'WebGPU runs compute shaders in browsers.',
]);

assert.equal(evidence.schema, 'doppler_rerank_evidence/v1');
assert.equal(evidence.scores.length, 2);
assert.equal(evidence.ranking[0].index, 1);
assert.equal(evidence.ranking[0].rank, 1);
assert.ok(evidence.ranking[0].score > evidence.ranking[1].score);
assert.equal(evidence.resolution.logicalModelId, 'search-reranker');
assert.equal(evidence.resolution.resolvedArtifactVariantId, `sha256:${'a'.repeat(64)}`);
assert.equal(
  evidence.resolution.resolvedExecutionId,
  computeCanonicalSha256(evidence.executionIdentity)
);
assert.equal(evidence.inputHash, computeCanonicalSha256({
  query: evidence.query,
  documents: evidence.documents,
}));
assert.equal(evidence.outputHash, computeCanonicalSha256({
  scores: evidence.scores,
  ranking: evidence.ranking,
}));

console.log('doppler-rerank-evidence.test: ok');
