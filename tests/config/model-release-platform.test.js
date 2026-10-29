import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

import {
  buildModelReleasePlatformReport,
  validateModelReleasePlatform,
} from '../../tools/check-model-release-platform.js';

const policy = JSON.parse(await fs.readFile('tools/policies/model-release-platform.json', 'utf8'));
const matrix = JSON.parse(await fs.readFile('src/config/goal-completion-matrix.json', 'utf8'));

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

for (const field of policy.adoptionGate.requiredEvidence) {
  const incomplete = clone(policy);
  incomplete.adoptionGate.requiredEvidence = incomplete.adoptionGate.requiredEvidence.filter((item) => item !== field);
  assert.equal((await validateModelReleasePlatform(incomplete, matrix)).ok, false, field);
}
{
  const peerDependent = clone(policy);
  peerDependent.adoptionGate.p2pRequired = true;
  assert.equal((await validateModelReleasePlatform(peerDependent, matrix)).ok, false);
}

{
  // The checked-in policy validates, and the report's summaries are derived from
  // the policy rows rather than maintained as separate lists.
  const report = await buildModelReleasePlatformReport();
  assert.equal(report.ok, true, report.errors.join('\n'));
  assert.equal(report.releaseActivationAuthority, policy.releaseContract.activationAuthority);
  assert.equal(report.promotionSequenceScope, policy.promotionSequenceScope);
  assert.deepEqual(
    report.migrationSurfaces,
    policy.apiConvergence.filter((row) => row.mode === 'migration-required').map((row) => row.id)
  );
  assert.deepEqual(
    report.partialRequirements,
    [...policy.capsule.requiredReleaseElements, ...policy.recovery]
      .filter((row) => row.implementationState === 'partial')
      .map((row) => row.id)
  );
  const gateIds = (state) => policy.promotionSequence
    .filter((row) => row.completionState === state)
    .map((row) => row.id);
  assert.deepEqual(report.implementedPromotionGates, gateIds('implemented'));
  assert.deepEqual(report.externalPromotionGates, gateIds('external-evidence-required'));
  // Every promotion gate is classified exactly once.
  assert.equal(
    report.implementedPromotionGates.length + report.externalPromotionGates.length,
    policy.promotionSequence.length
  );
}

{
  // An external gate without a defined blocker code is rejected.
  const broken = clone(policy);
  const gate = broken.promotionSequence.find((row) => row.completionState === 'external-evidence-required');
  assert.ok(gate, 'policy must contain an external-evidence gate to exercise this path');
  gate.blockerCode = 'undefined-blocker-code';
  const report = await validateModelReleasePlatform(broken, matrix);
  assert.ok(
    report.errors.includes(`promotionSequence.${gate.id}: external gates require a defined blockerCode`),
    report.errors.join('\n')
  );
}

{
  // A migration-required API surface needs a defined blocker; others must not carry one.
  const broken = clone(policy);
  const authoritative = broken.apiConvergence.find((row) => row.mode !== 'migration-required');
  authoritative.blockerCode = 'customer-electron-fleet-receipts-missing';
  const report = await validateModelReleasePlatform(broken, matrix);
  assert.ok(
    report.errors.includes(`apiConvergence.${authoritative.id}: non-migration row must have blockerCode null`),
    report.errors.join('\n')
  );
}

{
  const broken = clone(policy);
  const releaseCommand = broken.promotionSequence.find((row) => row.id === 'doppler-release-command');
  releaseCommand.blockerCode = 'customer-electron-fleet-receipts-missing';
  const report = await validateModelReleasePlatform(broken, matrix);
  assert.ok(report.errors.includes(
    'promotionSequence.doppler-release-command: implemented gates must have blockerCode null'
  ));
}

{
  const broken = clone(policy);
  broken.providers.doeRequired = true;
  const report = await validateModelReleasePlatform(broken, matrix);
  assert.ok(report.errors.includes('providers.doeRequired must be false'), report.errors.join('\n'));
}

{
  const broken = clone(policy);
  const migration = broken.capsule.requiredReleaseElements
    .find((row) => row.id === 'version-supersession-migration');
  migration.implementationState = 'partial';
  migration.blockerCode = null;
  const report = await validateModelReleasePlatform(broken, matrix);
  assert.ok(
    report.errors.includes('capsule.requiredReleaseElements.version-supersession-migration: partial rows require a blockerCode'),
    report.errors.join('\n')
  );
}

{
  const broken = clone(policy);
  broken.apiConvergence.find((row) => row.id === 'generation').evidencePaths = ['missing/path.js'];
  const report = await validateModelReleasePlatform(broken, matrix);
  assert.ok(
    report.errors.includes('apiConvergence.generation.evidencePaths references missing path missing/path.js'),
    report.errors.join('\n')
  );
}

{
  const broken = clone(matrix);
  const goal = broken.goals.find((row) => row.id === 'local-webgpu-product-surface');
  goal.rows = goal.rows.filter((row) => row.id !== 'canonical-production-release-contract');
  const report = await validateModelReleasePlatform(policy, broken);
  assert.ok(
    report.errors.includes('goal matrix is missing local-webgpu-product-surface/canonical-production-release-contract'),
    report.errors.join('\n')
  );
}

for (const field of policy.networkAcceptance.requiredEvidence) {
  const broken = clone(policy);
  broken.networkAcceptance.requiredEvidence = broken.networkAcceptance.requiredEvidence.filter((value) => value !== field);
  assert.equal((await validateModelReleasePlatform(broken, matrix)).ok, false, field);
}
for (const field of ['paymentRequired', 'doeRequired', 'agentCapsuleImprovementRequired', 'electronCustomerRequired']) {
  const broken = clone(policy);
  broken.networkAcceptance[field] = true;
  assert.equal((await validateModelReleasePlatform(broken, matrix)).ok, false, field);
}
{
  // The release contract carries no commercial offer; undeclared sections are rejected.
  const broken = clone(policy);
  broken.commercialOffer = { requiredForTechnicalAcceptance: false };
  const report = await validateModelReleasePlatform(broken, matrix);
  assert.ok(report.errors.includes('policy.commercialOffer is not allowed'), report.errors.join('\n'));
}
for (const [field, value] of [['activationAuthority', 'doppler-controlled'], ['selfPromotionAllowed', true]]) {
  const broken = clone(policy);
  broken.releaseContract[field] = value;
  assert.equal((await validateModelReleasePlatform(broken, matrix)).ok, false, field);
}
for (const [field, value] of [['privateInputDefault', 'shared'], ['delegation', 'automatic'], ['redistribution', 'automatic'], ['protocols', 'locked'], ['firstModelId', 'qwen']]) {
  const broken = clone(policy);
  broken.networkAcceptance[field] = value;
  assert.equal((await validateModelReleasePlatform(broken, matrix)).ok, false, field);
}
{
  const broken = clone(policy);
  broken.adoptionGate.paymentRequired = true;
  assert.equal((await validateModelReleasePlatform(broken, matrix)).ok, false);
}

console.log('model-release-platform.test: ok');
