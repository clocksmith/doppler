import assert from 'node:assert/strict';

import { InferencePipeline } from '../../src/inference/pipelines/text.js';

const pipeline = new InferencePipeline();
let observedPrompt = null;
let observedOptions = null;

pipeline.prefillWithEmbedding = async (prompt, options = {}) => {
  observedPrompt = prompt;
  observedOptions = options;
  return {
    embedding: new Float32Array([1, 0]),
    tokens: [1, 2, 3],
    seqLen: 3,
    embeddingMode: 'last',
  };
};

const result = await pipeline.embed('small-model embedding prompt', {
  signal: null,
  customFlag: true,
});

assert.equal(observedPrompt, 'small-model embedding prompt');
assert.equal(observedOptions.customFlag, true);
assert.equal(observedOptions.__skipStateSnapshot, true);
assert.deepEqual(Array.from(result.embedding), [1, 0]);
assert.deepEqual(result.tokens, [1, 2, 3]);
assert.equal(result.seqLen, 3);
assert.equal(result.embeddingMode, 'last');

console.log('pipeline-embedding-fast-path.test: ok');
