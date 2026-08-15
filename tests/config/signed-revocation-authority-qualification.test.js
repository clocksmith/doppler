import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { computeCanonicalJsonSha256 } from '../../tools/lib/canonical-json.js';
import { REVOCATION_AUTHORITY_EVIDENCE_CLASSES } from '../../tools/lib/signed-revocation-authority-evidence.js';
import {
  buildSignedRevocationAuthorityQualificationReport,
  validateSignedRevocationAuthorityQualification,
} from '../../tools/check-signed-revocation-authority-qualification.js';

const POLICY_PATH = path.join(
  process.cwd(),
  'tools',
  'policies',
  'signed-revocation-authority-qualification.json'
);
const NOW = new Date('2026-08-15T12:00:00.000Z');
const TEST_ROOT = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-revocation-authority-'));
const HARNESS_REVISION = '1'.repeat(40);
const ENVIRONMENT_ID = `sha256:${'a'.repeat(64)}`;

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

async function writeEvidence(relativePath, receipt) {
  const filePath = path.join(TEST_ROOT, relativePath);
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(receipt, null, 2)}\n`, 'utf8');
  return { path: relativePath, digest: computeCanonicalJsonSha256(receipt) };
}

function qualifyAuthority(authority) {
  authority.lifecycle = 'active';
  authority.ownerConfirmedAtUtc = '2026-08-01T00:00:00.000Z';
  authority.claimAllowed = true;
  authority.deployment.endpointUrl = 'https://revocations.clocksmith.dev/v1/doppler';
  authority.deployment.authorityId = 'clocksmith-doppler-production-v1';
  authority.deployment.onlineKeyIds = ['online-2026-08'];
  authority.deployment.recoveryKeyIds = ['recovery-2026'];
  authority.deployment.durableStateStoreIds.browser = 'indexeddb-revocation-state-v1';
  authority.deployment.durableStateStoreIds.node = 'atomic-file-revocation-state-v1';
  authority.qualifiedAtUtc = '2026-08-01T02:00:00.000Z';
  authority.expiresAtUtc = '2026-10-01T00:00:00.000Z';
  authority.blockers = [];
}

function observationsFor(evidenceClass, authority) {
  const values = {
    endpointDeployment: {
      endpointUrl: authority.deployment.endpointUrl,
      authorityId: authority.deployment.authorityId,
      transportPolicy: 'https-no-redirect',
      tlsValidated: true,
      redirectCount: 0,
      signatureVerified: true,
    },
    packageTrustBinding: {
      authorityId: authority.deployment.authorityId,
      onlineKeyIds: authority.deployment.onlineKeyIds,
      recoveryKeyIds: authority.deployment.recoveryKeyIds,
      packageTrustMatched: true,
    },
    onlineKeyCustody: {
      keyIds: authority.deployment.onlineKeyIds,
      custodyDomainId: 'online-hsm-domain',
      nonExportable: true,
      accessReviewPassed: true,
    },
    recoveryKeyCustody: {
      keyIds: authority.deployment.recoveryKeyIds,
      custodyDomainId: 'offline-recovery-domain',
      nonExportable: true,
      accessReviewPassed: true,
    },
    custodySeparation: {
      onlineCustodyDomainId: 'online-hsm-domain',
      recoveryCustodyDomainId: 'offline-recovery-domain',
      independentOperators: true,
      separationVerified: true,
    },
    browserDurableState: {
      host: 'browser',
      storeId: authority.deployment.durableStateStoreIds.browser,
      atomicCommitPassed: true,
      restartPersistencePassed: true,
      rollbackProtectionPassed: true,
    },
    nodeDurableState: {
      host: 'node',
      storeId: authority.deployment.durableStateStoreIds.node,
      atomicCommitPassed: true,
      restartPersistencePassed: true,
      rollbackProtectionPassed: true,
    },
    refreshCurrent: {
      currentUpdateAccepted: true,
      signatureVerified: true,
      stateAdvanced: true,
    },
    onlineKeyRotation: {
      oldOnlineKeyRejected: true,
      newOnlineKeyAccepted: true,
      recoveryAuthorizationVerified: true,
      stateAdvanced: true,
    },
    exactReplay: {
      initialAccepted: true,
      replayAcceptedAsNoOp: true,
      stateUnchanged: true,
    },
    rewrittenReplayRejection: { rewrittenReplayRejected: true, stateUnchanged: true },
    sequenceRollbackRejection: { sequenceRollbackRejected: true, stateUnchanged: true },
    epochRollbackRejection: { epochRollbackRejected: true, stateUnchanged: true },
    offlineExpiry: {
      expiredStateRejected: true,
      networkFailureSurfaced: true,
      failClosed: true,
    },
    compromiseRecovery: {
      compromisedOnlineKeyRejected: true,
      recoveryUpdateAccepted: true,
      replacementOnlineKeyAccepted: true,
    },
    durableStoreRestart: {
      browserStateRecovered: true,
      nodeStateRecovered: true,
      monotonicStatePreserved: true,
    },
    loadedIdentityInvalidation: {
      loadedIdentityInvalidated: true,
      furtherExecutionRejected: true,
      applicationNotified: true,
    },
    applicationFailClosed: {
      applicationId: 'reploid-production',
      failureSurfaced: true,
      alternateExecutionSuppressed: true,
    },
    requalification: {
      allEvidenceReplayed: true,
      requiredDrillCount: 11,
      passedDrillCount: 11,
      identityUnchanged: true,
    },
  };
  return values[evidenceClass];
}

function authorityEvidence(authority, evidenceClass, overrides = {}) {
  return {
    schema: 'doppler.signed-revocation-authority-evidence/v1',
    evidenceClass,
    qualificationId: authority.id,
    owner: authority.owner,
    authorityId: authority.deployment.authorityId,
    harnessRevision: HARNESS_REVISION,
    environmentFingerprint: ENVIRONMENT_ID,
    capturedAtUtc: '2026-08-01T01:00:00.000Z',
    result: { passed: true },
    observations: observationsFor(evidenceClass, authority),
    ...overrides,
  };
}

async function populateEvidence(authority) {
  authority.evidence.ownerConfirmation = await writeEvidence(
    'evidence/owner-confirmation.json',
    {
      schema: 'doppler.signed-revocation-authority-owner-confirmation/v1',
      qualificationId: authority.id,
      owner: authority.owner,
      ownerRepository: 'clocksmith/doppler',
      ownerRevision: HARNESS_REVISION,
      confirmedAtUtc: authority.ownerConfirmedAtUtc,
      maintenanceStatus: 'active',
      statement: 'The named security owner confirms active production authority maintenance.',
    }
  );
  for (const evidenceClass of REVOCATION_AUTHORITY_EVIDENCE_CLASSES) {
    authority.evidence[evidenceClass] = await writeEvidence(
      `evidence/${evidenceClass}.json`,
      authorityEvidence(authority, evidenceClass)
    );
  }
}

const policy = JSON.parse(await fs.readFile(POLICY_PATH, 'utf8'));

{
  const report = await buildSignedRevocationAuthorityQualificationReport({
    policyPath: POLICY_PATH,
    now: NOW,
  });
  assert.equal(report.ok, true, report.errors.join('\n'));
  assert.equal(report.gateSatisfied, false);
  assert.equal(report.qualifiedAuthorities, 0);
  assert.equal(report.candidateAuthorities, 1);
  assert.equal(report.authorities[0].id, 'doppler-production-revocation-authority');
  assert.equal(report.authorities[0].qualified, false);
  assert.equal(report.authorities[0].endpointConfigured, false);
  assert.deepEqual(report.authorities[0].qualifiedHosts, []);
  assert.ok(report.authorities[0].missingEvidence.includes('compromiseRecovery'));
}

{
  const complete = clone(policy);
  const authority = complete.authorities[0];
  qualifyAuthority(authority);
  await populateEvidence(authority);
  const report = await validateSignedRevocationAuthorityQualification(complete, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.deepEqual(report.errors, []);
  assert.equal(report.gateSatisfied, true);
  assert.equal(report.qualifiedAuthorities, 1);
  assert.deepEqual(report.authorities[0].qualifiedHosts, ['browser', 'node']);
}

{
  const overlappingKeys = clone(policy);
  overlappingKeys.authorities[0].deployment.onlineKeyIds = ['shared-key'];
  overlappingKeys.authorities[0].deployment.recoveryKeyIds = ['shared-key'];
  const report = await validateSignedRevocationAuthorityQualification(overlappingKeys, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.ok(
    report.errors.includes(
      'doppler-production-revocation-authority: online and recovery key IDs must be disjoint'
    ),
    report.errors.join('\n')
  );
}

{
  const falseClaim = clone(policy);
  falseClaim.authorities[0].claimAllowed = true;
  const report = await validateSignedRevocationAuthorityQualification(falseClaim, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.ok(
    report.errors.includes(
      'doppler-production-revocation-authority: claimAllowed authority does not satisfy production qualification'
    ),
    report.errors.join('\n')
  );
  assert.equal(report.gateSatisfied, false);
}

{
  const futureClaim = clone(policy);
  const authority = futureClaim.authorities[0];
  qualifyAuthority(authority);
  authority.qualifiedAtUtc = '2026-09-01T00:00:00.000Z';
  authority.expiresAtUtc = '2026-08-20T00:00:00.000Z';
  await populateEvidence(authority);
  const report = await validateSignedRevocationAuthorityQualification(futureClaim, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(report.authorities[0].qualificationReasons.includes('qualification-date-in-future'));
  assert.ok(
    report.authorities[0].qualificationReasons.includes(
      'qualification-expiry-not-after-qualification'
    )
  );
  assert.equal(report.gateSatisfied, false);
}

{
  const existenceOnly = clone(policy);
  for (const field of Object.keys(existenceOnly.authorities[0].evidence)) {
    existenceOnly.authorities[0].evidence[field] = 'docs/revocation.md';
  }
  const report = await validateSignedRevocationAuthorityQualification(existenceOnly, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.ok(report.errors.some((error) => error.includes('must be an object')));
  assert.equal(report.gateSatisfied, false);
}

{
  const falsePass = clone(policy);
  const authority = falsePass.authorities[0];
  qualifyAuthority(authority);
  await populateEvidence(authority);
  const reference = authority.evidence.compromiseRecovery;
  const receipt = authorityEvidence(authority, 'compromiseRecovery', {
    observations: {
      ...observationsFor('compromiseRecovery', authority),
      recoveryUpdateAccepted: false,
    },
  });
  authority.evidence.compromiseRecovery = await writeEvidence(reference.path, receipt);
  const report = await validateSignedRevocationAuthorityQualification(falsePass, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(report.errors.some((error) => error.includes('result.passed does not match')));
  assert.equal(report.gateSatisfied, false);
}

{
  const custodyMismatch = clone(policy);
  const authority = custodyMismatch.authorities[0];
  qualifyAuthority(authority);
  await populateEvidence(authority);
  const receipt = authorityEvidence(authority, 'custodySeparation', {
    observations: {
      ...observationsFor('custodySeparation', authority),
      onlineCustodyDomainId: 'unrelated-online-domain',
    },
  });
  authority.evidence.custodySeparation = await writeEvidence(
    authority.evidence.custodySeparation.path,
    receipt
  );
  const report = await validateSignedRevocationAuthorityQualification(custodyMismatch, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.includes(
      'doppler-production-revocation-authority: custody separation domains do not match custody receipts'
    )
  );
  assert.equal(report.gateSatisfied, false);
}

await fs.rm(TEST_ROOT, { recursive: true, force: true });

console.log('signed-revocation-authority-qualification.test: ok');
