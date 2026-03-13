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

{
  const freqs = await initRoPEFrequencies({
    headDim: 8,
    rotaryDim: undefined,
    maxSeqLen: 4,
    ropeTheta: 10000,
    ropeLocalTheta: 10000,
    mropeInterleaved: false,
    mropeSection: null,
    partialRotaryFactor: null,
    ropeScale: 4,
    ropeLocalScale: 1,
    ropeScalingType: 'yarn',
    ropeLocalScalingType: null,
    ropeScaling: {
      factor: 4,
      beta_fast: 32,
      beta_slow: 1,
      original_max_position_embeddings: 1,
    },
    ropeLocalScaling: null,
  }, false);

  assert.ok(freqs.localCos instanceof Float32Array);
  assert.ok(freqs.localSin instanceof Float32Array);
  assert.ok(Math.abs(freqs.localCos[4] - Math.cos(1)) < 1e-6);
  assert.ok(Math.abs(freqs.localSin[4] - Math.sin(1)) < 1e-6);
}

console.log('qwen-rope-runtime-config.test: ok');
