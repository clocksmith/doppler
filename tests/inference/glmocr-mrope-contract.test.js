import assert from 'node:assert/strict';

import { buildRoPEPrecomputeAuxiliaryData } from '../../src/gpu/kernels/rope-precompute.js';
import {
  buildGlmOcrRopePositionPlan,
} from '../../src/inference/pipelines/vision/glmocr-rope.js';

const plan = buildGlmOcrRopePositionPlan({
  promptLength: 12,
  capacity: 15,
  imageStartOffset: 3,
  imageTokenLength: 6,
  gridHeight: 4,
  gridWidth: 6,
  mergeSize: 2,
});

assert.deepEqual(Array.from(plan.temporal), [
  0, 1, 2,
  3, 3, 3, 3, 3, 3,
  6, 7, 8, 9, 10, 11,
]);
assert.deepEqual(Array.from(plan.height), [
  0, 1, 2,
  3, 3, 3, 4, 4, 4,
  6, 7, 8, 9, 10, 11,
]);
assert.deepEqual(Array.from(plan.width), [
  0, 1, 2,
  3, 4, 5, 3, 4, 5,
  6, 7, 8, 9, 10, 11,
]);
assert.equal(plan.ropeDelta, -3);

const auxiliary = buildRoPEPrecomputeAuxiliaryData({
  maxSeqLen: plan.capacity,
  positionPlan: plan,
  mropeSection: [1, 1, 2],
}, 4);
assert.equal(auxiliary.code, 3);
assert.deepEqual(auxiliary.mropeSection, [1, 1, 2]);
assert.deepEqual(
  Array.from(auxiliary.factors),
  [...plan.temporal, ...plan.height, ...plan.width]
);

assert.throws(
  () => buildRoPEPrecomputeAuxiliaryData({
    maxSeqLen: plan.capacity,
    positionPlan: plan,
    mropeSection: [1, 1, 1],
  }, 4),
  /summing to 4/
);

assert.throws(
  () => buildGlmOcrRopePositionPlan({
    promptLength: 8,
    capacity: 8,
    imageStartOffset: 2,
    imageTokenLength: 3,
    gridHeight: 4,
    gridWidth: 4,
    mergeSize: 2,
  }),
  /does not match merged grid/
);

console.log('glmocr-mrope-contract.test: ok');
