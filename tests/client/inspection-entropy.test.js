import assert from 'node:assert/strict';

import { buildInspectionTokenRecord, distributionEntropy } from '../../src/client/inspection.js';

const close = (actual, expected, label) => assert.ok(Math.abs(actual - expected) < 1e-9, `${label}: ${actual} vs ${expected}`);

// Uniform over four tokens: ln 4, whatever the shared offset.
close(distributionEntropy(new Float32Array([3, 3, 3, 3])), Math.log(4), 'uniform');
close(distributionEntropy([1000, 1000]), Math.log(2), 'large logits stay stable');
// One certain token: zero; masked entries add nothing.
close(distributionEntropy([0, -Infinity, -Infinity]), 0, 'masked entries');
// A known two-way split: p = sigmoid(1).
const p = 1 / (1 + Math.exp(-1));
close(distributionEntropy([1, 0]), -(p * Math.log(p) + (1 - p) * Math.log(1 - p)), 'two-way split');
assert.equal(distributionEntropy(null), null);
assert.equal(distributionEntropy([]), null);
assert.equal(distributionEntropy([0, Number.NaN]), null);
assert.equal(distributionEntropy([-Infinity, -Infinity]), null);

// Each inspection token record carries the full-distribution entropy at its step.
const tokenizer = { decode: (ids) => `t${ids[0]}` };
const record = buildInspectionTokenRecord(1, new Float32Array([0, 0, 0, 0]), tokenizer, 2, 0);
close(record.entropy, Math.log(4), 'record');
close(record.probability, 0.25, 'record probability unchanged');
assert.equal(buildInspectionTokenRecord(1, null, tokenizer, 2, 0).entropy, null);

console.log('inspection-entropy.test: ok');
