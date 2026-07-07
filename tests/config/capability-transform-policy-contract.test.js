import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { buildCapabilityTransformPolicyReport } from '../../tools/check-capability-transform-policy.js';

const policy = JSON.parse(
  readFileSync('src/rules/inference/capability-transforms.rules.json', 'utf8')
);

const report = await buildCapabilityTransformPolicyReport();

assert.equal(report.ok, true, report.errors.join('\n'));
assert.equal(report.rules, policy.capabilityTransforms.length);
assert.deepEqual(report.kinds, {
  'lane-mismatch-guard': 3,
  'platform-workaround': 1,
  'explicit-lane': 8,
  'capability-optimization': 1,
  'runtime-session-compatibility': 2,
  'hardware-compatibility': 3,
  'default-noop': 1,
});

const defaultRule = policy.capabilityTransforms.at(-1);
assert.equal(defaultRule.kind, 'default-noop');
assert.equal(defaultRule.dtypeEffect, 'none');
assert.deepEqual(defaultRule.match, {});
assert.deepEqual(defaultRule.transforms, []);
assert.deepEqual(defaultRule.evidence, []);

const byTransform = new Map();
for (const rule of policy.capabilityTransforms) {
  for (const transform of rule.transforms) {
    byTransform.set(transform, rule);
  }
}

assert.equal(byTransform.get('disableRetainQ4KMaterialization')?.kind, 'platform-workaround');
assert.equal(byTransform.get('disableRetainQ4KMaterialization')?.dtypeEffect, 'none');
assert.equal(byTransform.get('useQwenF16PrimaryMatmuls')?.kind, 'explicit-lane');
assert.equal(byTransform.get('useQwenF16PrimaryMatmuls')?.dtypeEffect, 'selective-f16');

const runtimeF32KvRule = policy.capabilityTransforms.find((rule) => {
  return rule.kind === 'runtime-session-compatibility'
    && rule.match.kvDtype === 'f32'
    && rule.match.hasSubgroups === true
    && rule.match.hasF16 === true;
});
assert.ok(runtimeF32KvRule, 'kvDtype=f32 runtime-session compatibility rule exists');
assert.equal(runtimeF32KvRule.dtypeEffect, 'full-f32');
assert.deepEqual(runtimeF32KvRule.transforms, ['widenToF32Activations']);

const noF16HardwareRule = policy.capabilityTransforms.find((rule) => {
  return rule.kind === 'hardware-compatibility'
    && rule.match.hasSubgroups === true
    && rule.match.hasF16 === false;
});
assert.ok(noF16HardwareRule, 'hasF16=false hardware compatibility rule exists');
assert.equal(noF16HardwareRule.dtypeEffect, 'full-f32');
assert.deepEqual(noF16HardwareRule.transforms, ['widenToF32Activations']);

console.log('capability-transform-policy-contract.test: ok');
