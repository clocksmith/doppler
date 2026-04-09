import assert from 'node:assert/strict';

await import('../../src/inference/pipelines/text.js');
const { InferencePipeline } = await import('../../src/inference/pipelines/text.js');

{
  const pipeline = new InferencePipeline();
  let visionLoads = 0;
  pipeline.visionCapable = true;
  pipeline._loadVisionWeights = async () => {
    visionLoads += 1;
    pipeline.visionWeights = { loaded: true };
  };

  await pipeline._ensureVisionWeightsLoaded();
  await pipeline._ensureVisionWeightsLoaded();

  assert.equal(visionLoads, 1);
  assert.deepEqual(pipeline.visionWeights, { loaded: true });
}

{
  const pipeline = new InferencePipeline();
  let audioLoads = 0;
  pipeline.audioCapable = true;
  pipeline._loadAudioWeights = async () => {
    audioLoads += 1;
    pipeline.audioWeights = { loaded: true };
  };

  await pipeline._ensureAudioWeightsLoaded();
  await pipeline._ensureAudioWeightsLoaded();

  assert.equal(audioLoads, 1);
  assert.deepEqual(pipeline.audioWeights, { loaded: true });
}

{
  const pipeline = new InferencePipeline();
  await assert.rejects(
    () => pipeline._ensureVisionWeightsLoaded(),
    /does not support vision weights/
  );
  await assert.rejects(
    () => pipeline._ensureAudioWeightsLoaded(),
    /does not support audio weights/
  );
}

console.log('multimodal-lazy-weight-load.test: ok');
