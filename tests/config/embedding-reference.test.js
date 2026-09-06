import assert from 'node:assert/strict';
import { assertEmbeddingReferenceTranscript, evaluateEmbeddingReference } from '../../src/config/embedding-reference.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import { createEmbeddingReferenceFixture } from '../helpers/embedding-reference-fixture.js';

const transcript = createEmbeddingReferenceFixture();
assert.equal(assertEmbeddingReferenceTranscript(transcript), transcript);
const tolerance = structuredClone(transcript);
tolerance.observation.outputs[0].embedding[2] = 0.005;
assert.doesNotThrow(() => assertEmbeddingReferenceTranscript(tolerance));
assert.equal(evaluateEmbeddingReference(tolerance.reference, tolerance.observation).checks[1].valueCount, 4);

for (const [change, pattern] of [
  [value => { value.observation.outputs[1].embedding[3] = 0.02; value.passed = true; }, /source comparison failed/],
  [value => { value.observation.outputs[0].tokenIds = [1, 9]; }, /source comparison failed/],
  [value => { value.observation.outputs.pop(); }, /every text/],
  [value => { value.observation.outputs[0].embedding = [1]; }, /complete finite vector/],
  [value => { value.observation.outputs[0].embedding[0] = NaN; }, /complete finite vector/],
  [value => { value.observation.input.texts.reverse(); }, /observed inputs differ/],
  [value => { value.observation.embeddingContract.postprocessor.normalize = null; }, /observed embedding contract differs/],
  [value => { value.referenceDigest = `sha256:${'2'.repeat(64)}`; }, /reference digest differs/],
  [value => { value.tokens = { ids: [1] }; }, /generation evidence cannot/],
  [value => { value.operation = 'encodeSequence'; }, /unsupported transcript/],
]) {
  const invalid = structuredClone(transcript);
  change(invalid);
  assert.throws(() => assertEmbeddingReferenceTranscript(invalid), pattern);
}
for (const change of [
  value => { value.tolerances.embeddingMaxAbs = -1; },
  value => { value.tolerances.tokenIds = 'similar'; },
  value => { value.source.revision = 'main'; },
  value => { value.source.files = []; },
]) {
  const invalid = structuredClone(transcript);
  change(invalid.reference);
  invalid.referenceDigest = computeCanonicalSha256(invalid.reference);
  assert.throws(() => assertEmbeddingReferenceTranscript(invalid), /Invalid embedding reference/);
}
console.log('embedding-reference.test: passed');
