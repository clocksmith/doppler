import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const sample = JSON.parse(
  readFileSync(new URL('../../demo/fixtures/translate-compare-evidence.sample.json', import.meta.url), 'utf8')
);
const checklist = readFileSync(
  new URL('../../docs/translate-compare-student-promotion-checklist.md', import.meta.url),
  'utf8'
);

assert.equal(sample.schemaVersion, 1);
assert.equal(sample.teacher.modelId, 'translategemma-4b-it-q4k-ehf16-af32');
assert.equal(sample.student.modelId, 'translategemma-4b-1b-enes-q4k-ehf16-af32');
assert.equal(sample.teacher.bleu, 33.69725818764181);
assert.equal(sample.teacher.chrf, 60.80110939124903);
assert.equal(sample.student.bleu, 31.914861871372885);
assert.equal(sample.student.chrf, 58.21235700409009);
assert.match(sample.student.label, /Experimental/);
assert.ok(Array.isArray(sample.receipts));
assert.ok(sample.receipts.length >= 1);
assert.ok(sample.receipts.some((receipt) => receipt.label === 'Hosted browser/WebGPU benchmark'));

assert.match(checklist, /final student checkpoint selected/);
assert.match(checklist, /Translate -> Compare -> Proof layout/);
assert.match(checklist, /canonical catalog entry first/);

console.log('translate-compare-sample-fixture.test: ok');
