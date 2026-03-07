import assert from 'node:assert/strict';

const { readBufferWithCleanup } = await import('../../src/inference/pipelines/text/logits/utils.js');

{
  let cleanupCount = 0;
  const reader = async () => new Uint8Array([1, 2, 3, 4]).buffer;
  const result = await readBufferWithCleanup({}, 4, () => {
    cleanupCount += 1;
  }, reader);

  assert.deepEqual(Array.from(new Uint8Array(result)), [1, 2, 3, 4]);
  assert.equal(cleanupCount, 1);
}

{
  let cleanupCount = 0;
  const reader = async () => {
    throw new Error('readback failed');
  };

  await assert.rejects(
    () => readBufferWithCleanup({}, 4, () => {
      cleanupCount += 1;
    }, reader),
    /readback failed/
  );

  assert.equal(cleanupCount, 1);
}

console.log('logits-utils-cleanup.test: ok');
