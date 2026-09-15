import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { KERNEL_TOLERANCES } from '../../../tests/kernels/harness/tolerance.js';

const tolerance = KERNEL_TOLERANCES.matmul_f32;
const baselinePath = new URL('../../reusable-products-2026-09-14/physical-standalone.json', import.meta.url);
const candidatePath = new URL('./physical-standalone.json', import.meta.url);
const baselineBytes = await fs.readFile(baselinePath);
const candidateBytes = await fs.readFile(candidatePath);
const baseline = JSON.parse(baselineBytes), candidate = JSON.parse(candidateBytes);
assert(baseline.passed && candidate.passed);
assert.equal(baseline.browserVersion, candidate.browserVersion);
const baselineConfig = JSON.parse(await fs.readFile(new URL('../../reusable-products-2026-09-14/physical-standalone-config.json', import.meta.url)));
const candidateConfig = JSON.parse(await fs.readFile(new URL('./physical-standalone-config.json', import.meta.url)));
assert.deepEqual(candidateConfig.models, baselineConfig.models, 'exact model descriptors and semantic requests');
const results = [];
for (const current of candidate.results) {
  const previous = baseline.results.find((row) => row.operation === current.operation);
  assert(previous);
  assert.equal(current.hardware.vendor, previous.hardware.vendor);
  assert.equal(current.hardware.architecture, previous.hardware.architecture);
  const a = previous.completed.output, b = current.completed.output;
  let numbers = 0, maxAbsoluteError = 0;
  const compare = (expected, actual) => {
    if (typeof expected === 'number') {
      assert(Number.isFinite(actual) && Number.isFinite(expected));
      const error = Math.abs(actual - expected);
      assert(error <= tolerance.atol + tolerance.rtol * Math.abs(expected));
      maxAbsoluteError = Math.max(maxAbsoluteError, error); numbers++;
    } else if (Array.isArray(expected)) {
      assert(Array.isArray(actual));
      assert.equal(actual.length, expected.length);
      expected.forEach((value, index) => compare(value, actual[index]));
    } else if (expected && typeof expected === 'object') {
      assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort());
      for (const key of Object.keys(expected)) compare(expected[key], actual[key]);
    } else assert.deepEqual(actual, expected);
  };
  if (current.operation === 'embed') {
    assert.equal(a.embeddings.length, b.embeddings.length);
    a.embeddings.forEach((item, index) => compare(item.embedding, b.embeddings[index].embedding));
  } else if (current.operation === 'rerank') {
    assert.deepEqual(b.capsule, a.capsule);
    assert.deepEqual(b.target, a.target);
    assert.equal(b.evidence.inputHash, a.evidence.inputHash);
    assert.deepEqual(b.evidence.ranking, a.evidence.ranking);
    compare(a.evidence.scores, b.evidence.scores);
  } else {
    assert.deepEqual(b, a, 'generation text, tokens, stopping reason and resolved settings');
  }
  results.push({ operation: current.operation, passed: true, comparedNumbers: numbers, maxAbsoluteError });
}
const receipt = {
  schema: 'doppler.installed-output-regression/v1', passed: true, tolerance,
  baselineArchive: baseline.package.sha256, candidateArchive: candidate.package.sha256,
  baselineReceipt: createHash('sha256').update(baselineBytes).digest('hex'),
  candidateReceipt: createHash('sha256').update(candidateBytes).digest('hex'), results,
  scope: 'Regression against accepted Doppler outputs on the same browser/GPU class. No new source-model or reranker task-quality qualification.',
};
await fs.writeFile(new URL('./model-output-regression.json', import.meta.url), JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify(receipt));
