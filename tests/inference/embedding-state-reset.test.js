import assert from 'node:assert/strict';

import { InferencePipeline } from '../../src/inference/pipelines/text.js';

function embeddingTarget(prefillWithEmbedding) {
  let sequenceLength = 9;
  let resetCount = 0;
  return {
    target: {
      resetForBatch() {
        sequenceLength = 0;
        resetCount += 1;
      },
      async prefillWithEmbedding(prompt, options) {
        assert.equal(sequenceLength, 0);
        return prefillWithEmbedding(prompt, options, (value) => {
          sequenceLength = value;
        });
      },
    },
    state() {
      return { sequenceLength, resetCount };
    },
  };
}

{
  const fixture = embeddingTarget(async (prompt, options, setSequenceLength) => {
    assert.equal(prompt, 'stable prompt');
    assert.equal(options.__skipStateSnapshot, true);
    setSequenceLength(4);
    return {
      embedding: Float32Array.from([1, 0]),
      tokens: [1, 2, 3, 4],
      seqLen: 4,
      embeddingMode: 'last',
    };
  });
  const result = await InferencePipeline.prototype.embed.call(fixture.target, 'stable prompt');
  assert.deepEqual(Array.from(result.embedding), [1, 0]);
  assert.deepEqual(fixture.state(), { sequenceLength: 0, resetCount: 2 });
}

{
  const fixture = embeddingTarget(async (_prompt, _options, setSequenceLength) => {
    setSequenceLength(5);
    throw new Error('controlled embedding failure');
  });
  await assert.rejects(
    InferencePipeline.prototype.embed.call(fixture.target, 'failing prompt'),
    /controlled embedding failure/
  );
  assert.deepEqual(fixture.state(), { sequenceLength: 0, resetCount: 2 });
}

console.log('embedding-state-reset.test: ok');
