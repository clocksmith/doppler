import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

import { buildPolicySchemaRegistryReport } from '../../tools/check-policy-schema-registry.js';
import { hashWgslSemanticEvidenceValue } from '../../src/tooling/wgsl-repair-semantic-gate.js';

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

function sha256File(filePath) {
  return createHash('sha256').update(readFileSync(filePath)).digest('hex');
}

function readJsonl(filePath) {
  return readFileSync(filePath, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
}

const policyPath = 'tools/policies/wgsl-writer-v3-campaign-policy.json';
const policy = readJson(policyPath);
const catalog = readJson(policy.mechanics.capabilityCatalog.path);
const packageSchema = readJson(policy.mechanics.responseSchema.path);
const predecessor = readJson(policy.predecessor.result.path);
const qualification = readJson(policy.mechanics.referenceQualification.receipt.path);
const diversityPolicy = readJson('tools/policies/wgsl-writer-v3-corpus-diversity-policy.json');
const diversityManifest = readJson(
  `${diversityPolicy.corpus.outputRoot}/corpus-manifest.json`
);
const diversityQualification = readJson(
  `${diversityPolicy.corpus.outputRoot}/reference-qualification.json`
);
const diversityTrainingPolicy = readJson(
  'tools/policies/wgsl-writer-v3-diversity-training-policy.json'
);

assert.equal(policy.policyId, 'doppler-wgsl-writer-v3-general-authoring');
assert.equal(policy.status, 'reference_qualified_corpus_materialization_blocked');
assert.equal(policy.targetCapability, catalog.targetCapability);
assert.equal(catalog.responseContract, 'doppler.wgsl-author-package/v1');
assert.equal(packageSchema.properties.schema.const, catalog.responseContract);
assert.equal(packageSchema.additionalProperties, false);
assert.equal(packageSchema.required.includes('requirements'), true);
assert.equal(packageSchema.required.includes('passes'), true);

for (const binding of [
  policy.predecessor.result,
  policy.mechanics.responseSchema,
  policy.mechanics.packageValidator,
  policy.mechanics.executionPlanner,
  policy.mechanics.formatCatalog,
  policy.mechanics.capabilityCatalog,
  policy.mechanics.browserExecutor,
  policy.mechanics.referenceQualification.library,
  policy.mechanics.referenceQualification.harness,
  policy.mechanics.referenceQualification.manifest,
  policy.mechanics.referenceQualification.receipt,
]) {
  assert.equal(sha256File(binding.path), binding.sha256, binding.path);
}

assert.equal(predecessor.experimentId, 'doppler-wgsl-writer-v2');
assert.equal(predecessor.parity.selectedSeed, policy.predecessor.selectedSeed);
assert.equal(predecessor.parity.decision, 'selected_adapter_parity_passed');
assert.equal(predecessor.generalWgslWriterClaim, false);
assert.equal(predecessor.productizationAllowed, false);
assert.equal(policy.predecessor.initializationMayBeTested, true);
assert.equal(policy.predecessor.capabilityClaimTransfers, false);
assert.equal(policy.predecessor.weightsPublished, false);

const expectedRoleCounts = {
  training: 8,
  calibration: 4,
  checkpoint_selection: 4,
  seed_confirmation: 4,
};
const roleIds = new Map();
const allIds = new Set();
for (const [role, count] of Object.entries(expectedRoleCounts)) {
  const families = catalog.families.filter((family) => family.populationRole === role);
  assert.equal(families.length, count, role);
  assert.equal(catalog.populationRoles[role].familyPlanCount, count, role);
  assert.equal(catalog.populationRoles[role].materialized, false, role);
  roleIds.set(role, new Set(families.map((family) => family.id)));
  const pipelineKinds = new Set(families.map((family) => family.pipelineKind));
  assert.equal(pipelineKinds.has('compute'), true, `${role}: compute`);
  assert.equal(pipelineKinds.has('render'), true, `${role}: render`);
  assert.equal(pipelineKinds.has('multi_pass'), true, `${role}: multi_pass`);
  for (const family of families) {
    assert.equal(allIds.has(family.id), false, family.id);
    allIds.add(family.id);
    assert.equal(family.verification.compilation, true, family.id);
    assert.equal(family.verification.actualExecution, true, family.id);
    assert.equal(family.verification.bufferBounds, true, family.id);
    assert.equal(family.verification.metamorphic, true, family.id);
    assert.equal(family.verification.historicalRegressions, true, family.id);
    assert.equal(family.verification.requiredVariations.length >= 2, true, family.id);
  }
}
for (const [leftRole, leftIds] of roleIds.entries()) {
  for (const [rightRole, rightIds] of roleIds.entries()) {
    if (leftRole >= rightRole) continue;
    assert.deepEqual(
      [...leftIds].filter((id) => rightIds.has(id)),
      [],
      `${leftRole}/${rightRole}`
    );
  }
}

assert.equal(catalog.promotion.materialized, false);
assert.equal(catalog.promotion.promotionAllowed, false);
assert.equal(catalog.promotion.naturalSpecificationsRequired, true);
assert.equal(catalog.promotion.familyDisjointFromAllDevelopmentRoles, true);
assert.equal(catalog.blockers.includes('package_executor_not_reference_qualified'), false);
assert.equal(catalog.blockers.includes('capability_tasks_and_oracles_not_materialized'), true);

assert.deepEqual(policy.authority, {
  corpusMaterialization: false,
  training: false,
  checkpointSelection: false,
  seedConfirmation: false,
  promotion: false,
  generalWgslWriterClaim: false,
  productization: false,
});
assert.equal(policy.mechanics.browserExecutor.status, 'reference_qualified');
assert.equal(policy.mechanics.referenceQualification.status, 'qualified');
assert.equal(qualification.decision, 'reference_package_mechanics_qualified');
assert.equal(qualification.summary.tasks, 4);
assert.equal(qualification.summary.runs, 8);
assert.equal(qualification.summary.passedTasks, 4);
assert.equal(qualification.summary.failedTasks, 0);
assert.equal(qualification.summary.deterministicReplayPassed, true);
assert.equal(qualification.summary.cleanupPassed, true);
assert.equal(qualification.runtime.identity.gpuBackend.detected, 'vulkan');
assert.equal(qualification.runtime.identity.webgpuAdapter.vendor, 'amd');
assert.equal(qualification.runtime.sessionCleanup.passed, true);
const { receiptHash, ...qualificationCore } = qualification;
assert.equal(hashWgslSemanticEvidenceValue(qualificationCore), receiptHash);
assert.equal(qualification.generalWgslWriterClaim, false);
assert.equal(qualification.productizationAllowed, false);
assert.equal(
  policy.blockers.includes('executable_package_browser_runner_is_not_reference_qualified'),
  false
);
assert.equal(
  policy.blockers.includes('executable_capability_tasks_cpu_and_raster_oracles_are_not_materialized'),
  true
);
assert.equal(policy.populationPlan.developmentRolesMaterialized, false);
assert.equal(policy.populationPlan.externalPromotion.materialized, false);
assert.equal(policy.training.allowed, false);
assert.equal(policy.training.workloadsFrozen, false);

const diversityRows = readJsonl(diversityManifest.roles.training.datasetPath);
assert.equal(diversityRows.length, 192);
assert.equal(new Set(diversityRows.map((row) => row.completionSha256)).size, 192);
for (const familyId of new Set(diversityRows.map((row) => row.semanticFamilyId))) {
  assert.equal(
    new Set(diversityRows
      .filter((row) => row.semanticFamilyId === familyId)
      .map((row) => row.completionSha256)).size,
    24,
    familyId
  );
}
assert.equal(diversityManifest.referenceQualification.tasks, 228);
assert.equal(diversityQualification.decision, 'reference_corpus_qualified');
assert.equal(diversityQualification.summary.tasks, 228);
assert.equal(diversityQualification.summary.runs, 456);
assert.equal(diversityQualification.summary.passedTasks, 228);
assert.equal(diversityQualification.summary.failedTasks, 0);
assert.equal(diversityQualification.summary.deterministicReplayPassed, true);
assert.equal(diversityQualification.summary.cleanupPassed, true);
assert.equal(diversityQualification.runtime.identity.gpuBackend.detected, 'vulkan');
assert.equal(diversityQualification.runtime.identity.webgpuAdapter.vendor, 'amd');
const { receiptHash: diversityReceiptHash, ...diversityCore } = diversityQualification;
assert.equal(hashWgslSemanticEvidenceValue(diversityCore), diversityReceiptHash);

assert.equal(diversityTrainingPolicy.repairEvidence.uniqueTrainingCompletionsBefore, 8);
assert.equal(diversityTrainingPolicy.repairEvidence.uniqueTrainingCompletionsAfter, 192);
assert.equal(diversityTrainingPolicy.repairEvidence.selectionGateChanged, false);
assert.equal(
  sha256File(diversityTrainingPolicy.repairEvidence.failedEvaluation.path),
  diversityTrainingPolicy.repairEvidence.failedEvaluation.sha256
);
assert.equal(diversityTrainingPolicy.evaluation.minimumSelectionSemanticPassRate, 0.5);
assert.equal(diversityTrainingPolicy.evaluation.minimumConfirmationMeanSemanticPassRate, 0.75);
for (const binding of Object.values(diversityTrainingPolicy.admission)) {
  assert.equal(sha256File(binding.path), binding.sha256, binding.path);
}

const registryReport = await buildPolicySchemaRegistryReport();
assert.equal(registryReport.ok, true, registryReport.errors.join('\n'));
const schemaRegistry = readJson('src/config/schema/policy-schema-registry.json');
assert.equal(
  schemaRegistry.policies.some((entry) => entry.id === 'wgsl-writer-v3-campaign-policy'),
  true
);
assert.equal(
  schemaRegistry.policies.some((entry) => entry.id === 'wgsl-writer-v3-corpus-diversity-policy'),
  true
);
assert.equal(
  schemaRegistry.policies.some((entry) => entry.id === 'wgsl-writer-v3-diversity-training-policy'),
  true
);

console.log('wgsl-writer-v3-contract.test: ok');
