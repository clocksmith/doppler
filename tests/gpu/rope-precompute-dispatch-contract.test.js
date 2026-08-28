import assert from 'node:assert/strict';

import { planRoPEPrecomputeDispatch } from '../../src/gpu/kernels/rope-precompute.js';

const WORKGROUP_SIZE = 256;
const WEBGPU_LIMIT = 65535;

const device = {
  limits: {
    maxComputeWorkgroupsPerDimension: WEBGPU_LIMIT,
  },
};

assert.deepEqual(
  planRoPEPrecomputeDispatch(device, WEBGPU_LIMIT * WORKGROUP_SIZE),
  {
    dispatchStride: WEBGPU_LIMIT * WORKGROUP_SIZE,
    workgroups: [WEBGPU_LIMIT, 1, 1],
  }
);

assert.deepEqual(
  planRoPEPrecomputeDispatch(device, (WEBGPU_LIMIT + 1) * WORKGROUP_SIZE),
  {
    dispatchStride: 32768 * WORKGROUP_SIZE,
    workgroups: [32768, 2, 1],
  }
);

assert.deepEqual(
  planRoPEPrecomputeDispatch(device, 1),
  {
    dispatchStride: WORKGROUP_SIZE,
    workgroups: [1, 1, 1],
  }
);

assert.throws(
  () => planRoPEPrecomputeDispatch({
    limits: { maxComputeWorkgroupsPerDimension: 2 },
  }, 5 * WORKGROUP_SIZE),
  /exceeding the device's 2 x 2 dispatch capacity/
);

assert.throws(
  () => planRoPEPrecomputeDispatch(device, 0),
  /positive safe integer/
);

console.log('rope-precompute-dispatch-contract.test: ok');
