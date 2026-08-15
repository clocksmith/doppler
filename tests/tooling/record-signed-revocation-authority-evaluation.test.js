import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { computeCanonicalJsonSha256 } from '../../tools/lib/canonical-json.js';
import { REVOCATION_AUTHORITY_EVIDENCE_CLASSES } from '../../tools/lib/signed-revocation-authority-evidence.js';
import {
  parseArgs,
  recordSignedRevocationAuthorityEvaluation,
} from '../../tools/record-signed-revocation-authority-evaluation.js';

const SOURCE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const NOW = new Date('2026-08-16T00:00:00.000Z');
const QUALIFICATION_ID = 'doppler-production-revocation-authority';
const AUTHORITY_ID = 'clocksmith-doppler-production-v1';
const ENDPOINT_URL = 'https://revocations.clocksmith.dev/v1/doppler';
const ONLINE_KEYS = ['online-2026-08'];
const RECOVERY_KEYS = ['recovery-2026'];
const BROWSER_STORE = 'indexeddb-revocation-state-v1';
const NODE_STORE = 'atomic-file-revocation-state-v1';
const HARNESS_REVISION = '1'.repeat(40);
const ENVIRONMENT_ID = `sha256:${'a'.repeat(64)}`;

async function writeJson(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function observationsFor(evidenceClass) {
  return {
    endpointDeployment: {
      endpointUrl: ENDPOINT_URL,
      authorityId: AUTHORITY_ID,
      transportPolicy: 'https-no-redirect',
      tlsValidated: true,
      redirectCount: 0,
      signatureVerified: true,
    },
    packageTrustBinding: {
      authorityId: AUTHORITY_ID,
      onlineKeyIds: ONLINE_KEYS,
      recoveryKeyIds: RECOVERY_KEYS,
      packageTrustMatched: true,
    },
    onlineKeyCustody: {
      keyIds: ONLINE_KEYS,
      custodyDomainId: 'online-hsm-domain',
      nonExportable: true,
      accessReviewPassed: true,
    },
    recoveryKeyCustody: {
      keyIds: RECOVERY_KEYS,
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
      storeId: BROWSER_STORE,
      atomicCommitPassed: true,
      restartPersistencePassed: true,
      rollbackProtectionPassed: true,
    },
    nodeDurableState: {
      host: 'node',
      storeId: NODE_STORE,
      atomicCommitPassed: true,
      restartPersistencePassed: true,
      rollbackProtectionPassed: true,
    },
    refreshCurrent: { currentUpdateAccepted: true, signatureVerified: true, stateAdvanced: true },
    onlineKeyRotation: {
      oldOnlineKeyRejected: true,
      newOnlineKeyAccepted: true,
      recoveryAuthorizationVerified: true,
      stateAdvanced: true,
    },
    exactReplay: { initialAccepted: true, replayAcceptedAsNoOp: true, stateUnchanged: true },
    rewrittenReplayRejection: { rewrittenReplayRejected: true, stateUnchanged: true },
    sequenceRollbackRejection: { sequenceRollbackRejected: true, stateUnchanged: true },
    epochRollbackRejection: { epochRollbackRejected: true, stateUnchanged: true },
    offlineExpiry: { expiredStateRejected: true, networkFailureSurfaced: true, failClosed: true },
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
  }[evidenceClass];
}

function authorityEvidence(evidenceClass, overrides = {}) {
  return {
    schema: 'doppler.signed-revocation-authority-evidence/v1',
    evidenceClass,
    qualificationId: QUALIFICATION_ID,
    owner: 'doppler-security',
    authorityId: AUTHORITY_ID,
    harnessRevision: HARNESS_REVISION,
    environmentFingerprint: ENVIRONMENT_ID,
    capturedAtUtc: '2026-08-15T19:00:00.000Z',
    result: { passed: true },
    observations: observationsFor(evidenceClass),
    ...overrides,
  };
}

function capture(evidencePaths) {
  return {
    schema: 'doppler.signed-revocation-authority-evaluation-capture/v1',
    qualificationId: QUALIFICATION_ID,
    evaluatedAtUtc: '2026-08-15T20:00:00.000Z',
    expiresAtUtc: '2026-09-14T20:00:00.000Z',
    evidencePaths,
  };
}

async function createFixture() {
  const repoRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-authority-record-'));
  const policy = JSON.parse(await fs.readFile(
    path.join(SOURCE_ROOT, 'tools/policies/signed-revocation-authority-qualification.json'),
    'utf8'
  ));
  const policyPath = path.join(
    repoRoot,
    'tools/policies/signed-revocation-authority-qualification.json'
  );
  const capturePath = path.join(repoRoot, 'capture.json');
  const outputPolicyPath = path.join(repoRoot, 'authority.next.json');
  const evidencePaths = { ownerConfirmation: 'evidence/owner-confirmation.json' };
  for (const evidenceClass of REVOCATION_AUTHORITY_EVIDENCE_CLASSES) {
    evidencePaths[evidenceClass] = `evidence/${evidenceClass}.json`;
  }
  await writeJson(policyPath, policy);
  await writeJson(path.join(repoRoot, evidencePaths.ownerConfirmation), {
    schema: 'doppler.signed-revocation-authority-owner-confirmation/v1',
    qualificationId: QUALIFICATION_ID,
    owner: 'doppler-security',
    ownerRepository: 'clocksmith/doppler',
    ownerRevision: HARNESS_REVISION,
    confirmedAtUtc: '2026-08-15T18:00:00.000Z',
    maintenanceStatus: 'active',
    statement: 'The named security owner confirms active production authority maintenance.',
  });
  for (const evidenceClass of REVOCATION_AUTHORITY_EVIDENCE_CLASSES) {
    await writeJson(
      path.join(repoRoot, evidencePaths[evidenceClass]),
      authorityEvidence(evidenceClass)
    );
  }
  await writeJson(capturePath, capture(evidencePaths));
  return { repoRoot, policyPath, capturePath, outputPolicyPath, evidencePaths };
}

async function withFixture(run) {
  const fixture = await createFixture();
  try {
    await run(fixture);
  } finally {
    await fs.rm(fixture.repoRoot, { recursive: true, force: true });
  }
}

await withFixture(async (fixture) => {
  const result = await recordSignedRevocationAuthorityEvaluation({ ...fixture, now: NOW });
  assert.equal(result.claimAllowed, false);
  assert.equal(result.lifecycle, 'candidate');
  assert.equal(result.authorityId, AUTHORITY_ID);
  assert.equal(result.endpointUrl, ENDPOINT_URL);
  assert.deepEqual(result.blockers, [
    'production-authority-evaluation-awaiting-explicit-promotion',
  ]);
  const output = JSON.parse(await fs.readFile(fixture.outputPolicyPath, 'utf8'));
  const authority = output.authorities.find((entry) => entry.id === QUALIFICATION_ID);
  assert.deepEqual(authority.deployment.onlineKeyIds, ONLINE_KEYS);
  assert.deepEqual(authority.deployment.recoveryKeyIds, RECOVERY_KEYS);
  assert.equal(authority.deployment.durableStateStoreIds.browser, BROWSER_STORE);
  assert.equal(authority.deployment.durableStateStoreIds.node, NODE_STORE);
  for (const [field, evidencePath] of Object.entries(fixture.evidencePaths)) {
    const receipt = JSON.parse(await fs.readFile(path.join(fixture.repoRoot, evidencePath), 'utf8'));
    assert.deepEqual(authority.evidence[field], {
      path: evidencePath,
      digest: computeCanonicalJsonSha256(receipt),
    });
  }
});

await withFixture(async (fixture) => {
  const evidencePath = path.join(fixture.repoRoot, fixture.evidencePaths.compromiseRecovery);
  await writeJson(evidencePath, authorityEvidence('compromiseRecovery', {
    result: { passed: false },
    observations: { ...observationsFor('compromiseRecovery'), recoveryUpdateAccepted: false },
  }));
  const result = await recordSignedRevocationAuthorityEvaluation({ ...fixture, now: NOW });
  assert.ok(result.blockers.includes('compromiseRecovery-not-passed'));
});

await withFixture(async (fixture) => {
  const evidencePath = path.join(fixture.repoRoot, fixture.evidencePaths.offlineExpiry);
  await writeJson(evidencePath, authorityEvidence('offlineExpiry', {
    authorityId: 'different-production-authority',
  }));
  await assert.rejects(
    () => recordSignedRevocationAuthorityEvaluation({ ...fixture, now: NOW }),
    /offlineExpiry evidence is invalid.*does not match expected/
  );
});

await withFixture(async (fixture) => {
  const evidencePath = path.join(fixture.repoRoot, fixture.evidencePaths.refreshCurrent);
  await writeJson(evidencePath, authorityEvidence('refreshCurrent', {
    capturedAtUtc: '2025-01-01T00:00:00.000Z',
  }));
  await assert.rejects(
    () => recordSignedRevocationAuthorityEvaluation({ ...fixture, now: NOW }),
    /Retained authority evidence exceeds the 90-day age limit/
  );
});

await withFixture(async (fixture) => {
  await recordSignedRevocationAuthorityEvaluation({ ...fixture, now: NOW });
  await fs.copyFile(fixture.outputPolicyPath, fixture.policyPath);
  await assert.rejects(
    () => recordSignedRevocationAuthorityEvaluation({ ...fixture, now: NOW }),
    /already contains evaluation state/
  );
  const result = await recordSignedRevocationAuthorityEvaluation({
    ...fixture,
    now: NOW,
    replace: true,
  });
  assert.equal(result.claimAllowed, false);
});

assert.throws(
  () => parseArgs([
    '--capture',
    'capture.json',
    '--out',
    'tools/policies/signed-revocation-authority-qualification.json',
  ]),
  /explicit --apply/
);
assert.equal(
  parseArgs(['--capture', 'capture.json', '--apply']).outputPolicyPath,
  path.join(SOURCE_ROOT, 'tools/policies/signed-revocation-authority-qualification.json')
);

console.log('record-signed-revocation-authority-evaluation.test: ok');
