import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { buildCapabilityTransformPolicyReport } from '../../tools/check-capability-transform-policy.js';

const policy = JSON.parse(
  readFileSync('src/rules/inference/capability-transforms.rules.json', 'utf8')
);

// The checker owns rule validation (kind, dtypeEffect, evidence, and the single
// trailing empty-match default-noop rule). Assert its verdict over the live
// policy rather than restating individual rule literals here.
const report = await buildCapabilityTransformPolicyReport();

assert.equal(report.ok, true, report.errors.join('\n'));
assert.deepEqual(report.errors, []);
assert.equal(report.rules, policy.capabilityTransforms.length);
assert.equal(
  Object.values(report.kinds).reduce((sum, count) => sum + count, 0),
  report.rules,
  'every rule must be classified under exactly one kind'
);
assert.equal(report.kinds['default-noop'], 1, 'checker must account for exactly one default-noop rule');
assert.equal(
  report.kinds['unknown-kind'],
  undefined,
  'every rule must declare a kind'
);

console.log('capability-transform-policy-contract.test: ok');
