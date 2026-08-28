import assert from 'node:assert/strict';

import {
  planVisionSpatialMergeDispatch,
} from '../../src/gpu/kernels/vision-spatial-merge.js';

assert.deepEqual(planVisionSpatialMergeDispatch(256), [1, 1, 1]);

const frozenValeraElements = 4_350 * 4 * 1_024;
const frozenValeraDispatch = planVisionSpatialMergeDispatch(frozenValeraElements);
assert.deepEqual(frozenValeraDispatch, [65_535, 2, 1]);
assert.ok(frozenValeraDispatch.every(value => value <= 65_535));
assert.ok(
  frozenValeraDispatch[0] * frozenValeraDispatch[1] * 256 >= frozenValeraElements
);

assert.throws(() => planVisionSpatialMergeDispatch(0), /positive integer/);
assert.throws(
  () => planVisionSpatialMergeDispatch((65_535 ** 2 * 256) + 1),
  /exceeding the two-axis WebGPU capacity/
);

console.log('vision_spatial_merge_dispatch-contract.test: ok');
