import assert from 'node:assert/strict';

import { initRoPEFrequencies } from '../../src/inference/pipelines/text/init.js';

{
  const freqs = await initRoPEFrequencies({
    headDim: 256,
    rotaryDim: 64,
    maxSeqLen: 8,
    ropeTheta: 10000000,
    ropeLocalTheta: null,
    mropeInterleaved: true,
    mropeSection: [11, 11, 10],
    partialRotaryFactor: 0.25,
    ropeScale: 1,
    ropeLocalScale: 1,
    ropeScalingType: null,
    ropeLocalScalingType: null,
    ropeScaling: null,
    ropeLocalScaling: null,
  }, false);

  assert.equal(freqs.cos.length, 8 * 32);
  assert.equal(freqs.sin.length, 8 * 32);
}

{
  await assert.rejects(
    initRoPEFrequencies({
      headDim: 256,
      rotaryDim: 64,
      maxSeqLen: 8,
      ropeTheta: 10000000,
      ropeLocalTheta: null,
      mropeInterleaved: true,
      mropeSection: [10, 10, 10],
      partialRotaryFactor: 0.25,
      ropeScale: 1,
      ropeLocalScale: 1,
      ropeScalingType: null,
      ropeLocalScalingType: null,
      ropeScaling: null,
      ropeLocalScaling: null,
    }, false),
    /mropeSection expands to 60 dims, but rotaryDim is 64/
  );
}

// Standard RoPE (no MRoPE): full rotary dim (headDim=256)
{
  const freqs = await initRoPEFrequencies({
    headDim: 256,
    rotaryDim: undefined,
    maxSeqLen: 8,
    ropeTheta: 10000000,
    ropeLocalTheta: null,
    mropeInterleaved: false,
    mropeSection: null,
    partialRotaryFactor: null,
    ropeScale: 1,
    ropeLocalScale: 1,
    ropeScalingType: null,
    ropeLocalScalingType: null,
    ropeScaling: null,
    ropeLocalScaling: null,
  }, false);

  // Full headDim=256 → halfDim=128 → cos/sin length = 8 * 128
  assert.equal(freqs.cos.length, 8 * 128);
  assert.equal(freqs.sin.length, 8 * 128);
}

console.log('qwen-rope-runtime-config.test: ok');
