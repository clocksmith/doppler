import assert from 'node:assert/strict';

import {
  assertMultimodalSequenceCapacity,
} from '../../src/inference/pipelines/text/modality-token-contract.js';

assert.equal(
  assertMultimodalSequenceCapacity({
    inputTokenCount: 4366,
    maxTokens: 32,
    maxSeqLen: 4608,
  }),
  4398
);

assert.throws(
  () => assertMultimodalSequenceCapacity({
    inputTokenCount: 4366,
    maxTokens: 1,
    maxSeqLen: 4096,
  }),
  /requires 4367 sequence slots \(4366 input \+ 1 output\).*maxSeqLen is 4096/
);

assert.throws(
  () => assertMultimodalSequenceCapacity({
    inputTokenCount: 4366,
    maxTokens: 1,
  }),
  /active KV cache maxSeqLen must be a positive integer/
);

console.log('multimodal-sequence-capacity-contract.test: ok');
