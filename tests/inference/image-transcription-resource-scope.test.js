import assert from 'node:assert/strict';

import {
  createImageTranscriptionResourceScope,
} from '../../src/inference/pipelines/text/image-transcription.js';

const originalCos = { label: 'original-cos' };
const originalSin = { label: 'original-sin' };
const overrideCos = { label: 'override-cos' };
const overrideSin = { label: 'override-sin' };
const features = { label: 'encoded-image-features' };
const pipeline = {
  ropeFreqsCos: originalCos,
  ropeFreqsSin: originalSin,
};
const releasedFeatures = [];
let releasedRopeOverrides = 0;

const resources = createImageTranscriptionResourceScope(
  pipeline,
  { features },
  (buffer) => releasedFeatures.push(buffer)
);

await assert.rejects(
  resources.runGeneration(async () => {}),
  /requires an active resource scope/
);

await assert.rejects(
  resources.run(async () => {
    resources.setGlmOcrRopeOverride({
      cos: overrideCos,
      sin: overrideSin,
      release() {
        releasedRopeOverrides += 1;
      },
    });

    assert.equal(pipeline.ropeFreqsCos, overrideCos);
    assert.equal(pipeline.ropeFreqsSin, overrideSin);

    await resources.runGeneration(async () => {
      assert.throws(
        () => resources.setGlmOcrRopeOverride({
          cos: overrideCos,
          sin: overrideSin,
          release() {},
        }),
        /after transcription started/
      );
      throw new Error('generation failed');
    });
  }),
  /generation failed/
);

assert.equal(pipeline.ropeFreqsCos, originalCos);
assert.equal(pipeline.ropeFreqsSin, originalSin);
assert.equal(releasedRopeOverrides, 1);
assert.deepEqual(releasedFeatures, [features]);

await assert.rejects(
  resources.run(async () => {}),
  /may run only once/
);
assert.equal(releasedRopeOverrides, 1);
assert.deepEqual(releasedFeatures, [features]);

console.log('image-transcription-resource-scope.test: ok');
