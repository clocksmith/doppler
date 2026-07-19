import assert from 'node:assert/strict';
import { computeLogits } from '../../src/inference/pipelines/text/logits/index.js';

const hiddenStates = new Float32Array([
  1, 2,
  3, 4,
]);
const finalNorm = new Float32Array([1, 1]);
const lmHead = new Float32Array(3 * 2);
const lmHeadBias = new Float32Array([0.25, -0.5, 1.75]);
const config = {
  hiddenSize: 2,
  vocabSize: 3,
  rmsNormEps: 1e-5,
  rmsNormWeightOffset: false,
  useTiedEmbeddings: false,
  embeddingVocabSize: null,
  finalLogitSoftcapping: null,
  logitInputScale: 1,
  activationDtype: 'f32',
};

const logits = await computeLogits(
  hiddenStates,
  2,
  { finalNorm, lmHead, lmHeadBias },
  config,
  false
);

assert.deepEqual(Array.from(logits), [
  0.25, -0.5, 1.75,
  0.25, -0.5, 1.75,
]);

await assert.rejects(
  computeLogits(
    hiddenStates,
    2,
    { finalNorm, lmHead, lmHeadBias: new Float32Array([1, 2]) },
    config,
    false
  ),
  /bias length mismatch/
);

console.log('logits-bias.test: ok');
