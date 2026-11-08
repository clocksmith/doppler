import assert from 'node:assert/strict';

const { planVideoFeatureConcat } = await import('../../src/inference/pipelines/video/feature-concat.js');
const { sampleFrames } = await import('../../src/inference/pipelines/video/frame-sampler.js');

// Gemma 4 E2B: the vision tower is 768 wide, but its projector emits text-width (1,536) features.
// The concat must follow the width each frame declares, never the vision config's hidden size.
{
  const plan = planVideoFeatureConcat([
    { featureSize: 1536, numTokens: 280 },
    { featureSize: 1536, numTokens: 70 },
  ]);
  assert.equal(plan.featureSize, 1536);
  assert.equal(plan.totalTokens, 350);
  assert.equal(plan.totalBytes, 350 * 1536 * 4);
  assert.deepEqual(plan.copies, [
    { offset: 0, byteLength: 280 * 1536 * 4 },
    { offset: 280 * 1536 * 4, byteLength: 70 * 1536 * 4 },
  ]);
}

assert.throws(() => planVideoFeatureConcat([]), /No frames were encoded/);
assert.throws(() => planVideoFeatureConcat([{ numTokens: 4 }]), /Frame 0 must declare a positive integer featureSize/);
assert.throws(
  () => planVideoFeatureConcat([{ featureSize: 1536, numTokens: 4 }, { featureSize: 768, numTokens: 4 }]),
  /Frame 1 declares featureSize 768; frame 0 declares 1536/
);
assert.throws(
  () => planVideoFeatureConcat([{ featureSize: 1536, numTokens: 0 }]),
  /Frame 0 must produce a positive integer numTokens/
);

// One sample from many frames takes the first frame instead of computing a 0/0 index.
{
  const frames = Array.from({ length: 5 }, (_, i) => ({ pixels: new Uint8Array(4), width: 1, height: 1, id: i }));
  const one = sampleFrames(frames, 1);
  assert.equal(one.length, 1);
  assert.equal(one[0].frameIndex, 0);
  assert.equal(one[0].id, 0);

  const three = sampleFrames(frames, 3);
  assert.deepEqual(three.map((frame) => frame.frameIndex), [0, 2, 4]);
}

console.log('video-feature-concat.test: ok');
