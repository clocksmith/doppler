import assert from 'node:assert/strict';
import { InferencePipeline } from '../../src/inference/pipelines/text.js';

const pipeline = Object.create(InferencePipeline.prototype);
pipeline.manifest = {
  inference: {
    supportsSequence: true,
    sequence: {
      alphabet: 'amino_acid',
      tokenEmbeddings: true,
      pooledEmbedding: {
        mode: 'mean',
        excludeTokenIds: [2, 3],
      },
      logits: true,
    },
  },
};
pipeline.modelConfig = { hiddenSize: 2, vocabSize: 7 };
let resets = 0;
let receivedOptions = null;
pipeline.resetForBatch = () => { resets += 1; };
pipeline.prefillWithEmbedding = async (_sequence, options) => {
  receivedOptions = options;
  return {
    tokens: [2, 5, 6, 3],
    tokenEmbeddings: new Float32Array([
      0, 0,
      2, 4,
      6, 8,
      0, 0,
    ]),
    logits: new Float32Array(4 * 7),
    phase: { totalMs: 1 },
  };
};

const result = await pipeline.encodeSequence('AC', { includeLogits: true });
assert.equal(resets, 2);
assert.equal(receivedOptions.__skipStateSnapshot, true);
assert.equal(receivedOptions.__returnTokenEmbeddings, true);
assert.equal(receivedOptions.__returnSequenceLogits, true);
assert.equal(result.alphabet, 'amino_acid');
assert.deepEqual(Array.from(result.tokenMask), [0, 1, 1, 0]);
assert.equal(result.includedTokenCount, 2);
assert.deepEqual(Array.from(result.pooledEmbedding), [4, 6]);
assert.equal(result.tokenEmbeddings.length, 8);
assert.equal(result.logits.length, 28);
assert.equal(result.embeddingDim, 2);
assert.equal(result.vocabSize, 7);

pipeline.manifest.inference.supportsSequence = false;
await assert.rejects(
  pipeline.encodeSequence('AC'),
  /does not declare sequence encoding support/
);

console.log('sequence-encoding-contract.test: ok');
