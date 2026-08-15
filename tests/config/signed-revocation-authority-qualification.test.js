import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';

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

function clone(value) {
  return JSON.parse(JSON.stringify(value));
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
  authority.lifecycle = 'active';
  authority.ownerConfirmedAtUtc = '2026-08-01T00:00:00.000Z';
  authority.claimAllowed = true;
  authority.deployment.endpointUrl = 'https://revocations.clocksmith.dev/v1/doppler';
  authority.deployment.authorityId = 'clocksmith-doppler-production-v1';
  authority.deployment.onlineKeyIds = ['online-2026-08'];
  authority.deployment.recoveryKeyIds = ['recovery-2026'];
  authority.deployment.durableStateStoreIds.browser = 'indexeddb-revocation-state-v1';
  authority.deployment.durableStateStoreIds.node = 'atomic-file-revocation-state-v1';
  authority.qualifiedAtUtc = '2026-08-01T00:00:00.000Z';
  authority.expiresAtUtc = '2026-10-01T00:00:00.000Z';
  for (const field of Object.keys(authority.evidence)) {
    authority.evidence[field] = 'docs/revocation.md';
  }
  authority.blockers = [];
  const report = await validateSignedRevocationAuthorityQualification(complete, {
    repoRoot: process.cwd(),
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
  authority.lifecycle = 'active';
  authority.ownerConfirmedAtUtc = '2026-08-01T00:00:00.000Z';
  authority.claimAllowed = true;
  authority.deployment.endpointUrl = 'https://revocations.clocksmith.dev/v1/doppler';
  authority.deployment.authorityId = 'clocksmith-doppler-production-v1';
  authority.deployment.onlineKeyIds = ['online-2026-08'];
  authority.deployment.recoveryKeyIds = ['recovery-2026'];
  authority.deployment.durableStateStoreIds.browser = 'indexeddb-revocation-state-v1';
  authority.deployment.durableStateStoreIds.node = 'atomic-file-revocation-state-v1';
  authority.qualifiedAtUtc = '2026-09-01T00:00:00.000Z';
  authority.expiresAtUtc = '2026-08-20T00:00:00.000Z';
  for (const field of Object.keys(authority.evidence)) {
    authority.evidence[field] = 'docs/revocation.md';
  }
  authority.blockers = [];
  const report = await validateSignedRevocationAuthorityQualification(futureClaim, {
    repoRoot: process.cwd(),
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

console.log('signed-revocation-authority-qualification.test: ok');
