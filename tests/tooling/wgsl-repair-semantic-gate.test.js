import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  evaluateNumericAgreement,
  evaluateWgslSemanticReadiness,
  evaluateWgslSemanticTaskEvidence,
} from '../../src/tooling/wgsl-repair-semantic-gate.js';

const policy = JSON.parse(readFileSync('tools/policies/wgsl-repair-v13-semantic-policy.json', 'utf8'));
const preservationReceipt = JSON.parse(readFileSync(
  'docs/status/wgsl-repair-v12-adapter-preservation-verification-2026-07-13.json',
  'utf8'
));

const withinTolerance = evaluateNumericAgreement(
  [1, 2, 3],
  [1.000001, 2.00001, 2.99999],
  policy.taskContract.numericalAgreement.tiers.float32
);
assert.equal(withinTolerance.pass, true);
const outsideTolerance = evaluateNumericAgreement(
  [1],
  [1.1],
  policy.taskContract.numericalAgreement.tiers.float32
);
assert.equal(outsideTolerance.pass, false);
const nonFiniteMismatch = evaluateNumericAgreement(
  [1],
  [Number.NaN],
  policy.taskContract.numericalAgreement.tiers.float32
);
assert.equal(nonFiniteMismatch.pass, false);
assert.equal(nonFiniteMismatch.mismatches[0].reason, 'nonfinite_mismatch');

function variant(shapeId, shapeClass, workgroupId) {
  return {
    shapeId,
    shapeClass,
    workgroupId,
    dispatch: { status: 'pass', backend: 'chromium_webgpu' },
    oracle: {
      expected: [1, 2, 3],
      actual: [1.000001, 2.00001, 2.99999],
      tolerance: policy.taskContract.numericalAgreement.tiers.float32,
    },
    bufferBounds: {
      prefixCanaryIntact: true,
      suffixCanaryIntact: true,
      readOnlyBuffersUnchanged: true,
      outputPaddingUnchanged: true,
      validationErrorsAbsent: true,
    },
    metamorphic: [
      { id: 'input_permutation_equivariance', status: 'pass' },
      { id: 'tiling_equivalence', status: 'pass' },
    ],
  };
}

const passingEvidence = {
  taskId: 'semantic-unit-task',
  responseContractPass: true,
  compilation: { status: 'pass' },
  variants: [
    variant('shape-nominal', 'nominal', 'wg-64'),
    variant('shape-tail', 'non_workgroup_multiple', 'wg-128'),
    variant('shape-boundary', 'boundary_or_tail', 'wg-64'),
  ],
  historicalRegressionsPass: true,
};
const passed = evaluateWgslSemanticTaskEvidence(policy, passingEvidence);
assert.equal(passed.pass, true);
assert.equal(passed.distinctShapeCount, 3);
assert.equal(passed.distinctWorkgroupCount, 2);

const corrupted = structuredClone(passingEvidence);
corrupted.variants[1].oracle.actual[0] = 99;
corrupted.variants[2].bufferBounds.suffixCanaryIntact = false;
const failed = evaluateWgslSemanticTaskEvidence(policy, corrupted);
assert.equal(failed.pass, false);
assert.ok(failed.blockers.includes('cpu_oracle_mismatch'));
assert.ok(failed.blockers.includes('buffer_canary_corruption'));

const compileFailure = structuredClone(passingEvidence);
compileFailure.compilation.status = 'fail';
assert.ok(
  evaluateWgslSemanticTaskEvidence(policy, compileFailure).blockers.includes('compilation_failure')
);

const readiness = evaluateWgslSemanticReadiness({
  policy,
  predecessorVerified: true,
  preservationReceipt,
  taskEvidence: [],
});
assert.equal(readiness.decision, 'blocked');
assert.equal(readiness.admission.semanticEvaluationAllowed, false);
assert.equal(readiness.admission.semanticClaimAllowed, false);
assert.equal(readiness.admission.wgslDoctorAllowed, false);
assert.equal(readiness.admission.autonomousShaderAuthorAllowed, false);
assert.ok(readiness.blockers.includes('v12_adapter_external_preservation_incomplete'));
assert.ok(readiness.blockers.includes('semantic_dispatch_evidence_absent'));

console.log('wgsl-repair-semantic-gate.test: ok');
