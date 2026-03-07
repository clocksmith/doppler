import assert from 'node:assert/strict';

import {
  DEFAULT_KERNEL_THRESHOLDS,
  getKernelThresholds,
  resetKernelThresholds,
  setKernelThresholds,
} from '../../src/config/schema/index.js';

resetKernelThresholds();

{
  const snapshot = getKernelThresholds();
  snapshot.attention.chunkedMaxKVLen = 123;
  snapshot.tuner.maxComputeWorkgroupSizeX = 42;

  const freshSnapshot = getKernelThresholds();
  assert.equal(freshSnapshot.attention.chunkedMaxKVLen, DEFAULT_KERNEL_THRESHOLDS.attention.chunkedMaxKVLen);
  assert.equal(freshSnapshot.tuner.maxComputeWorkgroupSizeX, DEFAULT_KERNEL_THRESHOLDS.tuner.maxComputeWorkgroupSizeX);
}

{
  setKernelThresholds({
    attention: {
      chunkedMaxKVLen: 4097,
    },
  });
  assert.equal(getKernelThresholds().attention.chunkedMaxKVLen, 4097);

  resetKernelThresholds();
  assert.equal(
    getKernelThresholds().attention.chunkedMaxKVLen,
    DEFAULT_KERNEL_THRESHOLDS.attention.chunkedMaxKVLen
  );
}

console.log('kernel-thresholds-state-isolation.test: ok');
