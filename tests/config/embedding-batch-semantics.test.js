import assert from 'node:assert/strict';
import { resolveEmbeddingBatchSemantics } from '../../src/capsule-runtime.js';
import { evaluateEmbeddingBatchSubdivision } from '../../src/config/embedding-reference.js';
import { createEmbeddingReferenceFixture } from '../helpers/embedding-reference-fixture.js';

const fixture = createEmbeddingReferenceFixture();
const manifest = { modelType: 'embedding', architecture: { hiddenSize: 4 },
  inference: { output: { embeddingPostprocessor: fixture.reference.embeddingContract.postprocessor } } };
const semantics = resolveEmbeddingBatchSemantics(manifest);
assert.deepEqual(semantics, {
  schema: 'doppler.embedding-batch-semantics/v1', operation: { name: 'embed', version: 1 },
  inputField: 'texts', unit: 'text', state: 'reset-per-text', outputOrder: 'input-order',
});
assert(Object.isFrozen(semantics));
assert(Object.isFrozen(semantics.operation));
assert.throws(() => resolveEmbeddingBatchSemantics({ ...manifest, modelType: 'transformer' }), /embedding support/);
assert.throws(() => resolveEmbeddingBatchSemantics({ ...manifest, inference: {} }), /embeddingPostprocessor/);
const unsupported = structuredClone(manifest);
unsupported.inference.output.embeddingPostprocessor.includePrompt = false;
assert.throws(() => resolveEmbeddingBatchSemantics(unsupported), /embeddingPostprocessor/);

// Synthetic observations protect the evaluator, not physical model accuracy.
fixture.reference.input.texts.push('one');
fixture.reference.outputs.push(structuredClone(fixture.reference.outputs[0]));
fixture.observation = structuredClone({ input: fixture.reference.input,
  embeddingContract: fixture.reference.embeddingContract, outputs: fixture.reference.outputs });
const unit = positions => ({ positions, observation: {
  input: { texts: positions.map(index => fixture.reference.input.texts[index]) },
  embeddingContract: fixture.reference.embeddingContract,
  outputs: positions.map(index => structuredClone(fixture.observation.outputs[index])),
} });
const units = [unit([2]), unit([0, 1])];
const evaluate = (value = units, whole = fixture.observation) =>
  evaluateEmbeddingBatchSubdivision(fixture.reference, whole, value);
assert.equal(evaluate().passed, true, 'out-of-order units preserve repeated inputs');
for (const [change, pattern] of [
  [value => value.pop(), /every input/],
  [value => value.push(structuredClone(value[0])), /unique/],
  [value => { value[0].positions[0] = 3; }, /in range/],
  [value => { value[1].observation.input.texts.reverse(); }, /unit inputs differ/],
  [value => { value[0].observation.outputs[0].embedding.pop(); }, /complete finite vector/],
  [value => { value[0].observation.embeddingContract.dimension = 5; }, /contract differs/],
]) {
  const invalid = structuredClone(units);
  change(invalid);
  assert.throws(() => evaluate(invalid), pattern);
}
const wrong = structuredClone(units);
wrong[0].observation.outputs[0].embedding[0] += 0.02;
assert.equal(evaluate(wrong).splitReference.passed, false);
const wrongTokens = structuredClone(units);
wrongTokens[0].observation.outputs[0].tokenIds[0] += 1;
assert.equal(evaluate(wrongTokens).passed, false);
const matchingWrong = structuredClone(fixture.observation);
matchingWrong.outputs[2].embedding[0] += 0.02;
const selfAgreement = evaluate(wrong, matchingWrong);
assert.equal(selfAgreement.splitWhole.passed, true);
assert.equal(selfAgreement.passed, false, 'self agreement cannot replace independent source correctness');
console.log('embedding-batch-semantics.test: passed');
