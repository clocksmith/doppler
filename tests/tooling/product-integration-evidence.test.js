import assert from 'node:assert/strict';

import {
  computeProductIntegrationEvidenceSetDigest,
  validateProductIntegrationOutcomeEvidence,
  validateProductIntegrationOwnerConfirmation,
  validateProductIntegrationPromotionEvidence,
} from '../../tools/lib/product-integration-evidence.js';

const SHA_A = `sha256:${'a'.repeat(64)}`;
const SHA_B = `sha256:${'b'.repeat(64)}`;
const SHA_C = `sha256:${'c'.repeat(64)}`;
const REVISION_A = '1'.repeat(40);
const REVISION_B = '2'.repeat(40);

const passingObservations = {
  installToFirstVerifiedOutput: {
    surface: 'root-api',
    installSucceeded: true,
    firstVerifiedOutputMs: 100,
    maximumFirstVerifiedOutputMs: 200,
  },
  sourceTaskQualityRetention: {
    sourceScore: 100,
    dopplerScore: 95,
    retentionRatio: 0.95,
    minimumRetentionRatio: 0.9,
  },
  reliability: {
    attempts: 10,
    successes: 10,
    minimumSuccessRate: 0.9,
    crashes: 0,
    maximumCrashes: 0,
    ooms: 0,
    maximumOoms: 0,
    deviceLosses: 0,
    maximumDeviceLosses: 0,
  },
  memory: { peakBytes: 100, budgetBytes: 200 },
  coldWarmResponse: {
    sampleCount: 10,
    coldP50Ms: 50,
    coldP95Ms: 80,
    coldP95LimitMs: 100,
    warmP50Ms: 10,
    warmP95Ms: 20,
    warmP95LimitMs: 30,
  },
  browserHardwareQualification: {
    qualifiedTargets: ['chromium-test'],
    failedTargets: [],
    minimumQualifiedTargets: 1,
  },
  incumbentControl: {
    incumbentProviderId: 'incumbent',
    incumbentArtifactId: 'artifact@revision',
    comparisonReceiptDigest: SHA_C,
    incumbentAvailable: true,
    correctnessComparable: true,
  },
  upgradeRequalification: {
    fromVersion: '1.0.0',
    toVersion: '1.1.0',
    migrationSucceeded: true,
    identityPreserved: true,
    taskGatePassed: true,
  },
  rollbackRevocation: {
    knownSafeVersion: '1.0.0',
    rollbackSucceeded: true,
    revocationObserved: true,
    taskGatePassed: true,
  },
};

const failingObservations = {
  installToFirstVerifiedOutput: { installSucceeded: false },
  sourceTaskQualityRetention: { dopplerScore: 50, retentionRatio: 0.5 },
  reliability: { successes: 5 },
  memory: { peakBytes: 300 },
  coldWarmResponse: { coldP95Ms: 150 },
  browserHardwareQualification: { failedTargets: ['chromium-test'] },
  incumbentControl: { correctnessComparable: false },
  upgradeRequalification: { taskGatePassed: false },
  rollbackRevocation: { revocationObserved: false },
};

function outcomeReceipt(evidenceClass, passed, observations) {
  return {
    schema: 'doppler.product-integration-evidence/v1',
    evidenceClass,
    integrationId: 'fixture-integration',
    applicationName: 'Fixture Application',
    workload: 'generation',
    owner: 'clocksmith/fixture',
    applicationRevision: REVISION_A,
    harnessRevision: REVISION_B,
    environmentFingerprint: SHA_A,
    logicalModelId: 'fixture-logical-model',
    resolvedArtifactVariantId: SHA_B,
    resolvedExecutionId: SHA_C,
    capturedAtUtc: '2026-08-15T12:00:00.000Z',
    result: { passed },
    observations,
  };
}

for (const [evidenceClass, observations] of Object.entries(passingObservations)) {
  const passing = validateProductIntegrationOutcomeEvidence(
    outcomeReceipt(evidenceClass, true, observations)
  );
  assert.deepEqual(passing.errors, [], `${evidenceClass}: ${passing.errors.join('\n')}`);
  assert.deepEqual(passing.reasons, []);
  assert.equal(passing.applicationRevision, REVISION_A);
  assert.equal(passing.harnessRevision, REVISION_B);
  assert.equal(passing.environmentFingerprint, SHA_A);

  const failedValues = { ...observations, ...failingObservations[evidenceClass] };
  const failed = validateProductIntegrationOutcomeEvidence(
    outcomeReceipt(evidenceClass, false, failedValues)
  );
  assert.deepEqual(failed.errors, [], `${evidenceClass}: ${failed.errors.join('\n')}`);
  assert.deepEqual(failed.reasons, [`${evidenceClass}-not-passed`]);

  const falseClaim = validateProductIntegrationOutcomeEvidence(
    outcomeReceipt(evidenceClass, true, failedValues)
  );
  assert.ok(
    falseClaim.errors.some((error) => error.includes('result.passed does not match')),
    `${evidenceClass}: ${falseClaim.errors.join('\n')}`
  );
}

{
  const receipt = {
    schema: 'doppler.product-integration-owner-confirmation/v1',
    integrationId: 'fixture-integration',
    applicationName: 'Fixture Application',
    workload: 'generation',
    owner: 'clocksmith/fixture',
    ownerRepository: 'clocksmith/fixture',
    applicationRevision: REVISION_A,
    confirmedAtUtc: '2026-08-15T12:00:00.000Z',
    maintenanceStatus: 'active',
    statement: 'The named owner confirms active maintenance.',
  };
  const result = validateProductIntegrationOwnerConfirmation(receipt, {
    integrationId: receipt.integrationId,
    applicationName: receipt.applicationName,
    workload: receipt.workload,
    owner: receipt.owner,
    applicationRevision: receipt.applicationRevision,
    ownerConfirmedAtUtc: receipt.confirmedAtUtc,
  });
  assert.deepEqual(result.errors, []);
  assert.deepEqual(result.reasons, []);
  assert.equal(result.applicationRevision, REVISION_A);
  const inactive = validateProductIntegrationOwnerConfirmation({
    ...receipt,
    maintenanceStatus: 'inactive',
  });
  assert.deepEqual(inactive.errors, []);
  assert.deepEqual(inactive.reasons, ['owner-maintenance-not-active']);
}

{
  const evidenceReferences = {
    ownerConfirmation: { path: 'evidence/owner.json', digest: SHA_A },
    identity: { path: 'evidence/identity.json', digest: SHA_B },
  };
  const evidenceSetDigest = computeProductIntegrationEvidenceSetDigest(evidenceReferences);
  const receipt = {
    schema: 'doppler.product-integration-promotion-evidence/v1',
    integrationId: 'fixture-integration',
    applicationName: 'Fixture Application',
    workload: 'generation',
    owner: 'clocksmith/fixture',
    logicalModelId: 'fixture-logical-model',
    resolvedArtifactVariantId: SHA_B,
    resolvedExecutionId: SHA_C,
    applicationRevision: REVISION_A,
    harnessRevision: REVISION_B,
    environmentFingerprint: SHA_A,
    evidenceSetDigest,
    decision: 'promote-product-supported',
    authority: 'human',
    reviewer: 'fixture-reviewer',
    reviewerRevision: '3'.repeat(40),
    rationale: 'The exact evidence set qualifies this integration.',
    promotedAtUtc: '2026-08-15T13:00:00.000Z',
    qualifiedAtUtc: '2026-08-15T12:30:00.000Z',
    expiresAtUtc: '2026-09-15T12:30:00.000Z',
  };
  const expected = {
    integrationId: receipt.integrationId,
    applicationName: receipt.applicationName,
    workload: receipt.workload,
    owner: receipt.owner,
    logicalModelId: receipt.logicalModelId,
    resolvedArtifactVariantId: receipt.resolvedArtifactVariantId,
    resolvedExecutionId: receipt.resolvedExecutionId,
    applicationRevision: receipt.applicationRevision,
    harnessRevision: receipt.harnessRevision,
    environmentFingerprint: receipt.environmentFingerprint,
    evidenceSetDigest,
    qualifiedAtUtc: receipt.qualifiedAtUtc,
    expiresAtUtc: receipt.expiresAtUtc,
  };
  assert.deepEqual(validateProductIntegrationPromotionEvidence(receipt, expected).errors, []);
  const staleDigest = validateProductIntegrationPromotionEvidence(
    { ...receipt, evidenceSetDigest: SHA_C },
    expected
  );
  assert.ok(staleDigest.errors.some((error) => error.includes('evidenceSetDigest')));
}

console.log('product-integration-evidence.test: ok');
