import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { buildCapabilityTransformPolicyReport } from '../../tools/check-capability-transform-policy.js';

const policy = JSON.parse(
  readFileSync('src/rules/inference/capability-transforms.rules.json', 'utf8')
);

const report = await buildCapabilityTransformPolicyReport();

assert.equal(report.ok, true, report.errors.join('\n'));
assert.equal(report.rules, policy.capabilityTransforms.length);

const defaultRule = policy.capabilityTransforms.at(-1);
assert.equal(defaultRule.kind, 'default-noop');
assert.equal(defaultRule.dtypeEffect, 'none');
assert.deepEqual(defaultRule.match, {});
assert.deepEqual(defaultRule.transforms, []);
assert.deepEqual(defaultRule.evidence, []);

console.log('capability-transform-policy-contract.test: ok');
