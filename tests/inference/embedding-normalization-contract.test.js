import assert from 'node:assert/strict';

import { resolveEmbeddingNormalization } from '../../src/inference/pipelines/text/embedding-contract.js';

assert.equal(resolveEmbeddingNormalization(null), null);
assert.deepEqual(resolveEmbeddingNormalization({
  type: 'rmsnorm',
  withScale: false,
  eps: 1e-5,
  position: 'after-scale',
}), {
  type: 'rmsnorm',
  withScale: false,
  eps: 1e-5,
  position: 'after-scale',
});
assert.throws(
  () => resolveEmbeddingNormalization({
    type: 'rmsnorm', withScale: true, eps: 1e-5, position: 'after-scale',
  }),
  /supports only weightless rmsnorm/
);
assert.throws(
  () => resolveEmbeddingNormalization({
    type: 'rmsnorm', withScale: false, eps: 1e-5, position: 'before-scale',
  }),
  /position must be "after-scale"/
);

console.log('embedding-normalization-contract.test: ok');
