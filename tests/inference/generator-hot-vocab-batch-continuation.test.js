import assert from 'node:assert/strict';

const { shouldDisableBatchDecodeAfterShortBatch } = await import('../../src/inference/pipelines/text/generator.js');

assert.equal(
  shouldDisableBatchDecodeAfterShortBatch({
    hitStop: false,
    actualCount: 3,
    requestedCount: 8,
  }),
  true,
  'short non-stop batch should fall back from batch decode'
);

assert.equal(
  shouldDisableBatchDecodeAfterShortBatch({
    hitStop: true,
    actualCount: 3,
    requestedCount: 8,
  }),
  false,
  'EOS/stop batches must not disable batch decode early'
);

assert.equal(
  shouldDisableBatchDecodeAfterShortBatch({
    hitStop: false,
    actualCount: 8,
    requestedCount: 8,
  }),
  false,
  'full batches must not disable batch decode'
);

console.log('generator-hot-vocab-batch-continuation.test: ok');
