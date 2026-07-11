import assert from 'node:assert/strict';

import {
  computeAttentionBackwardData,
  computeAttentionSoftmaxData,
} from '../../src/experimental/training/attention-backward.js';

const options = {
  seqLen: 1,
  numHeads: 4,
  numKVHeads: 1,
  headDim: 1,
  scale: 1,
  causal: true,
};
const q = new Float32Array([1, 2, 3, 4]);
const k = new Float32Array([2]);
const v = new Float32Array([3]);
const dOutput = new Float32Array([1, 1, 1, 1]);
const softmax = computeAttentionSoftmaxData(q, k, options);

assert.deepEqual([...softmax], [1, 1, 1, 1]);

const result = computeAttentionBackwardData(q, k, v, softmax, dOutput, options);
assert.equal(result.dQ.length, 4);
assert.equal(result.dK.length, 1);
assert.equal(result.dV.length, 1);
assert.deepEqual([...result.dQ], [0, 0, 0, 0]);
assert.deepEqual([...result.dK], [0]);
assert.deepEqual([...result.dV], [4]);
assert.deepEqual(result.geometry, {
  seqLen: 1,
  numHeads: 4,
  numKVHeads: 1,
  headDim: 1,
  headsPerKV: 4,
  scale: 1,
  causal: true,
});

assert.throws(
  () => computeAttentionSoftmaxData(q, k, { ...options, numKVHeads: 3 }),
  /valid GQA geometry/
);

console.log('attention-backward-gqa.test: ok');
