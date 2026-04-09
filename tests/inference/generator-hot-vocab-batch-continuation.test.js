import assert from 'node:assert/strict';

const {
  shouldDisableBatchDecodeAfterShortBatch,
  resolveHotVocabularyBatchDecodeAvailability,
} = await import('../../src/inference/pipelines/text/generator.js');

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

{
  const hotTokenIndexMap = new Uint32Array(8);
  hotTokenIndexMap.fill(2);
  hotTokenIndexMap[3] = 0;
  hotTokenIndexMap[5] = 1;

  assert.equal(
    resolveHotVocabularyBatchDecodeAvailability({
      hasRangeBackedPerLayerInputs: true,
      pleHotVocabularyRuntime: {
        sentinelIndex: 2,
        hotTokenIndexMap,
      },
      tokenId: 3,
    }),
    true,
    'current hot token should keep tokenizer_scores batch decode enabled'
  );

  assert.equal(
    resolveHotVocabularyBatchDecodeAvailability({
      hasRangeBackedPerLayerInputs: true,
      pleHotVocabularyRuntime: {
        sentinelIndex: 2,
        hotTokenIndexMap,
      },
      tokenId: 4,
    }),
    false,
    'current non-hot token must disable tokenizer_scores batch decode regardless of earlier hits'
  );
}

console.log('generator-hot-vocab-batch-continuation.test: ok');
