import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const schema = JSON.parse(readFileSync(new URL('../../demo/translate-compare-evidence.schema.json', import.meta.url), 'utf8'));
const doc = readFileSync(new URL('../../docs/translate-compare-evidence-contract.md', import.meta.url), 'utf8');
const demoSource = readFileSync(new URL('../../demo/demo.js', import.meta.url), 'utf8');

assert.equal(schema.title, 'Doppler Translate Compare Evidence Bundle');
assert.equal(schema.properties.schemaVersion.const, 1);
assert.deepEqual(schema.required, ['summary', 'caution', 'teacher', 'student', 'receipts']);
assert.deepEqual(schema.$defs.modelEvidence.required, ['label', 'modelId', 'bleu', 'chrf', 'sizeBytes']);

assert.match(doc, /translate-compare-evidence\.schema\.json/);
assert.match(doc, /student\.modelId/);
assert.match(doc, /If a metric is not frozen yet, send `null`/);

assert.match(demoSource, /Baseline parity is currently unsupported in public TJS ONNX exports/);

console.log('translate-compare-evidence-contract.test: ok');
