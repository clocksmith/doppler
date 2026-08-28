import assert from 'node:assert/strict';

import { planSplitQKVDispatch } from '../../src/gpu/kernels/split_qkv.js';

assert.deepEqual(planSplitQKVDispatch(256), [1, 1, 1]);
assert.deepEqual(planSplitQKVDispatch(17_400 * 3_072), [65_535, 4, 1]);

const oversize = planSplitQKVDispatch(17_400 * 3_072);
assert.ok(oversize.every(value => value <= 65_535));
assert.ok(oversize[0] * oversize[1] * 256 >= 17_400 * 3_072);

assert.throws(() => planSplitQKVDispatch(0), /positive safe integer/);
assert.throws(
  () => planSplitQKVDispatch((65_535 ** 2 * 256) + 1),
  /exceeding the two-axis WebGPU capacity/
);

console.log('split_qkv_dispatch-contract.test: ok');
