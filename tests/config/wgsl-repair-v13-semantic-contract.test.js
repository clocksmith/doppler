import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { evaluateWgslSemanticReadiness } from '../../src/tooling/wgsl-repair-semantic-gate.js';

const policy = JSON.parse(readFileSync('tools/policies/wgsl-repair-v13-semantic-policy.json', 'utf8'));
const preservation = JSON.parse(readFileSync(
  policy.predecessor.preservationVerificationPath,
  'utf8'
));
const committed = JSON.parse(readFileSync(
  'docs/status/wgsl-repair-v13-semantic-readiness-2026-07-13.json',
  'utf8'
));
const replayed = evaluateWgslSemanticReadiness({
  policy,
  predecessorVerified: true,
  preservationReceipt: preservation,
  taskEvidence: [],
});

assert.deepEqual(replayed, committed);
assert.equal(policy.status, 'frozen_requirements_populations_unmaterialized');
assert.equal(policy.taskContract.actualDispatchRequired, true);
assert.equal(policy.taskContract.cpuOracle.required, true);
assert.equal(policy.taskContract.shapeVariation.minimumDistinctShapesPerTask, 3);
assert.equal(policy.taskContract.workgroupVariation.minimumSemanticallyValidVariantsPerTask, 2);
assert.equal(policy.taskContract.bufferBounds.prefixCanaryElements, 16);
assert.equal(policy.taskContract.bufferBounds.suffixCanaryElements, 16);
assert.equal(policy.taskContract.metamorphic.minimumApplicableRelationsPerTask, 2);
assert.equal(policy.taskContract.historicalRegressions.required, true);
assert.equal(policy.metrics.primary, 'semantic_task_pass_at_1');
assert.equal(committed.decision, 'blocked');
assert.equal(committed.admission.wgslDoctorAllowed, false);
assert.equal(committed.admission.autonomousShaderAuthorAllowed, false);

console.log('wgsl-repair-v13-semantic-contract.test: ok');
