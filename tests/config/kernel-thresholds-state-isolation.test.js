import assert from 'node:assert/strict';

import {
  DEFAULT_KERNEL_THRESHOLDS,
  getKernelThresholds,
  getKernelThresholdSnapshot,
  resetKernelThresholds,
  setKernelThresholds,
} from '../../src/config/schema/index.js';

resetKernelThresholds();

{
  const snapshot = getKernelThresholds();
  snapshot.attention.chunkedMaxKVLen = 123;

  const freshSnapshot = getKernelThresholds();
  assert.equal(freshSnapshot.attention.chunkedMaxKVLen, DEFAULT_KERNEL_THRESHOLDS.attention.chunkedMaxKVLen);
}

{
  const before = getKernelThresholdSnapshot();
  assert.equal(getKernelThresholdSnapshot(), before);
  assert.throws(() => { before.attention.tierHeadDimLimits.tier1 = 1; }, TypeError);
  const attention = { tierHeadDimLimits: { tier1: 32, tier2: 16, tier3: 8 } };
  setKernelThresholds({ attention });
  attention.tierHeadDimLimits.tier1 = 999;
  const after = getKernelThresholdSnapshot();
  assert.notEqual(after, before);
  assert.equal(after.attention.tierHeadDimLimits.tier1, 32);
  assert.equal(before.attention.tierHeadDimLimits.tier1, DEFAULT_KERNEL_THRESHOLDS.attention.tierHeadDimLimits.tier1);
  const copy = getKernelThresholds();
  copy.attention.tierHeadDimLimits.tier1 = 777;
  assert.equal(after.attention.tierHeadDimLimits.tier1, 32);
  resetKernelThresholds();
  assert.equal(after.attention.tierHeadDimLimits.tier1, 32);
  assert.deepEqual(getKernelThresholdSnapshot(), DEFAULT_KERNEL_THRESHOLDS);
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
