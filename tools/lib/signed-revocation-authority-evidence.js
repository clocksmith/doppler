import { computeCanonicalJsonSha256 } from './canonical-json.js';

const OWNER_SCHEMA = 'doppler.signed-revocation-authority-owner-confirmation/v1';
const EVIDENCE_SCHEMA = 'doppler.signed-revocation-authority-evidence/v1';
const PROMOTION_SCHEMA = 'doppler.signed-revocation-authority-promotion-evidence/v1';
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const REVISION_PATTERN = /^[0-9a-f]{40}$/;

export const REVOCATION_AUTHORITY_EVIDENCE_CLASSES = Object.freeze([
  'endpointDeployment',
  'packageTrustBinding',
  'onlineKeyCustody',
  'recoveryKeyCustody',
  'custodySeparation',
  'browserDurableState',
  'nodeDurableState',
  'refreshCurrent',
  'onlineKeyRotation',
  'exactReplay',
  'rewrittenReplayRejection',
  'sequenceRollbackRejection',
  'epochRollbackRejection',
  'offlineExpiry',
  'compromiseRecovery',
  'durableStoreRestart',
  'loadedIdentityInvalidation',
  'applicationFailClosed',
  'requalification',
]);

const OBSERVATION_FIELDS = Object.freeze({
  endpointDeployment: [
    'endpointUrl', 'authorityId', 'transportPolicy', 'tlsValidated', 'redirectCount',
    'signatureVerified',
  ],
  packageTrustBinding: [
    'authorityId', 'onlineKeyIds', 'recoveryKeyIds', 'packageTrustMatched',
  ],
  onlineKeyCustody: ['keyIds', 'custodyDomainId', 'nonExportable', 'accessReviewPassed'],
  recoveryKeyCustody: ['keyIds', 'custodyDomainId', 'nonExportable', 'accessReviewPassed'],
  custodySeparation: [
    'onlineCustodyDomainId', 'recoveryCustodyDomainId', 'independentOperators',
    'separationVerified',
  ],
  browserDurableState: [
    'host', 'storeId', 'atomicCommitPassed', 'restartPersistencePassed',
    'rollbackProtectionPassed',
  ],
  nodeDurableState: [
    'host', 'storeId', 'atomicCommitPassed', 'restartPersistencePassed',
    'rollbackProtectionPassed',
  ],
  refreshCurrent: ['currentUpdateAccepted', 'signatureVerified', 'stateAdvanced'],
  onlineKeyRotation: [
    'oldOnlineKeyRejected', 'newOnlineKeyAccepted', 'recoveryAuthorizationVerified',
    'stateAdvanced',
  ],
  exactReplay: ['initialAccepted', 'replayAcceptedAsNoOp', 'stateUnchanged'],
  rewrittenReplayRejection: ['rewrittenReplayRejected', 'stateUnchanged'],
  sequenceRollbackRejection: ['sequenceRollbackRejected', 'stateUnchanged'],
  epochRollbackRejection: ['epochRollbackRejected', 'stateUnchanged'],
  offlineExpiry: ['expiredStateRejected', 'networkFailureSurfaced', 'failClosed'],
  compromiseRecovery: [
    'compromisedOnlineKeyRejected', 'recoveryUpdateAccepted',
    'replacementOnlineKeyAccepted',
  ],
  durableStoreRestart: [
    'browserStateRecovered', 'nodeStateRecovered', 'monotonicStatePreserved',
  ],
  loadedIdentityInvalidation: [
    'loadedIdentityInvalidated', 'furtherExecutionRejected', 'applicationNotified',
  ],
  applicationFailClosed: [
    'applicationId', 'failureSurfaced', 'alternateExecutionSuppressed',
  ],
  requalification: [
    'allEvidenceReplayed', 'requiredDrillCount', 'passedDrillCount', 'identityUnchanged',
  ],
});

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function exactKeys(value, fields, label, errors) {
  if (!isPlainObject(value)) {
    errors.push(`${label} must be an object`);
    return false;
  }
  const expected = new Set(fields);
  for (const field of Object.keys(value)) {
    if (!expected.has(field)) errors.push(`${label}.${field} is not supported`);
  }
  for (const field of fields) {
    if (!Object.hasOwn(value, field)) errors.push(`${label}.${field} is required`);
  }
  return true;
}

function text(value, label, errors) {
  const normalized = normalizeText(value);
  if (!normalized) errors.push(`${label} must be a non-empty string`);
  return normalized || null;
}

function sha256(value, label, errors) {
  const normalized = text(value, label, errors)?.toLowerCase() ?? null;
  if (normalized && !SHA256_PATTERN.test(normalized)) errors.push(`${label} must be a SHA-256 identity`);
  return normalized;
}

function revision(value, label, errors) {
  const normalized = text(value, label, errors)?.toLowerCase() ?? null;
  if (normalized && !REVISION_PATTERN.test(normalized)) {
    errors.push(`${label} must be a 40-character commit revision`);
  }
  return normalized;
}

function instant(value, label, errors) {
  const normalized = text(value, label, errors);
  const parsed = normalized ? new Date(normalized) : null;
  if (!parsed || !Number.isFinite(parsed.getTime()) || parsed.toISOString() !== normalized) {
    errors.push(`${label} must be an ISO instant`);
    return null;
  }
  return parsed;
}

function boolean(value, label, errors) {
  if (typeof value !== 'boolean') {
    errors.push(`${label} must be boolean`);
    return false;
  }
  return value;
}

function integer(value, label, errors, minimum = 0) {
  if (!Number.isInteger(value) || value < minimum) {
    errors.push(`${label} must be an integer >= ${minimum}`);
    return null;
  }
  return value;
}

function strings(value, label, errors) {
  if (!Array.isArray(value)) {
    errors.push(`${label} must be an array`);
    return [];
  }
  const result = value.map((entry, index) => text(entry, `${label}[${index}]`, errors)).filter(Boolean);
  if (new Set(result).size !== result.length) errors.push(`${label} contains duplicates`);
  return result;
}

function sameSequence(left, right) {
  return Array.isArray(right)
    && left.length === right.length
    && left.every((entry, index) => entry === right[index]);
}

function match(actual, expected, label, errors) {
  if (expected != null && actual != null && actual !== expected) {
    errors.push(`${label} ${actual} does not match expected ${expected}`);
  }
}

function matchList(actual, expected, label, errors) {
  if (Array.isArray(expected) && !sameSequence(actual, expected)) {
    errors.push(`${label} does not match expected key identities`);
  }
}

function allTrue(value, fields, errors) {
  let passed = true;
  for (const field of fields) {
    if (!boolean(value[field], `observations.${field}`, errors)) passed = false;
  }
  return passed;
}

function validateObservations(evidenceClass, value, expected, errors) {
  const fields = OBSERVATION_FIELDS[evidenceClass];
  if (!fields || !exactKeys(value, fields, `${evidenceClass} observations`, errors)) return false;
  if (evidenceClass === 'endpointDeployment') {
    const endpointUrl = text(value.endpointUrl, 'observations.endpointUrl', errors);
    const authorityId = text(value.authorityId, 'observations.authorityId', errors);
    const transportPolicy = text(value.transportPolicy, 'observations.transportPolicy', errors);
    match(endpointUrl, expected.endpointUrl, 'observed endpointUrl', errors);
    match(authorityId, expected.authorityId, 'observed authorityId', errors);
    const transportPassed = transportPolicy === 'https-no-redirect';
    const booleansPassed = allTrue(value, ['tlsValidated', 'signatureVerified'], errors);
    const redirectsPassed = integer(
      value.redirectCount,
      'observations.redirectCount',
      errors
    ) === 0;
    return transportPassed && booleansPassed && redirectsPassed;
  }
  if (evidenceClass === 'packageTrustBinding') {
    const authorityId = text(value.authorityId, 'observations.authorityId', errors);
    const online = strings(value.onlineKeyIds, 'observations.onlineKeyIds', errors);
    const recovery = strings(value.recoveryKeyIds, 'observations.recoveryKeyIds', errors);
    match(authorityId, expected.authorityId, 'observed authorityId', errors);
    matchList(online, expected.onlineKeyIds, 'observed onlineKeyIds', errors);
    matchList(recovery, expected.recoveryKeyIds, 'observed recoveryKeyIds', errors);
    return boolean(value.packageTrustMatched, 'observations.packageTrustMatched', errors);
  }
  if (evidenceClass === 'onlineKeyCustody' || evidenceClass === 'recoveryKeyCustody') {
    const keyIds = strings(value.keyIds, 'observations.keyIds', errors);
    matchList(
      keyIds,
      evidenceClass === 'onlineKeyCustody' ? expected.onlineKeyIds : expected.recoveryKeyIds,
      'observed keyIds',
      errors
    );
    text(value.custodyDomainId, 'observations.custodyDomainId', errors);
    return allTrue(value, ['nonExportable', 'accessReviewPassed'], errors);
  }
  if (evidenceClass === 'custodySeparation') {
    const online = text(value.onlineCustodyDomainId, 'observations.onlineCustodyDomainId', errors);
    const recovery = text(
      value.recoveryCustodyDomainId,
      'observations.recoveryCustodyDomainId',
      errors
    );
    const booleansPassed = allTrue(
      value,
      ['independentOperators', 'separationVerified'],
      errors
    );
    return online !== recovery && booleansPassed;
  }
  if (evidenceClass === 'browserDurableState' || evidenceClass === 'nodeDurableState') {
    const host = evidenceClass === 'browserDurableState' ? 'browser' : 'node';
    match(text(value.host, 'observations.host', errors), host, 'observed host', errors);
    match(
      text(value.storeId, 'observations.storeId', errors),
      expected.durableStateStoreIds?.[host],
      'observed storeId',
      errors
    );
    return allTrue(value, [
      'atomicCommitPassed', 'restartPersistencePassed', 'rollbackProtectionPassed',
    ], errors);
  }
  if (evidenceClass === 'requalification') {
    const required = integer(value.requiredDrillCount, 'observations.requiredDrillCount', errors, 1);
    const passed = integer(value.passedDrillCount, 'observations.passedDrillCount', errors);
    const replayed = boolean(
      value.allEvidenceReplayed,
      'observations.allEvidenceReplayed',
      errors
    );
    const unchanged = boolean(value.identityUnchanged, 'observations.identityUnchanged', errors);
    return replayed
      && unchanged
      && required === expected.requiredDrillCount
      && passed === required;
  }
  if (evidenceClass === 'applicationFailClosed') {
    text(value.applicationId, 'observations.applicationId', errors);
    return allTrue(value, ['failureSurfaced', 'alternateExecutionSuppressed'], errors);
  }
  const booleanFields = fields;
  return allTrue(value, booleanFields, errors);
}

export function validateRevocationAuthorityOwnerConfirmation(receipt, expected = {}) {
  const errors = [];
  const reasons = [];
  const fields = [
    'schema', 'qualificationId', 'owner', 'ownerRepository', 'ownerRevision',
    'confirmedAtUtc', 'maintenanceStatus', 'statement',
  ];
  if (!exactKeys(receipt, fields, 'owner confirmation', errors)) {
    return { errors, reasons: ['owner-confirmation-invalid'], confirmedAt: null };
  }
  if (receipt.schema !== OWNER_SCHEMA) errors.push(`owner confirmation.schema must be ${OWNER_SCHEMA}`);
  const qualificationId = text(receipt.qualificationId, 'owner confirmation.qualificationId', errors);
  const owner = text(receipt.owner, 'owner confirmation.owner', errors);
  text(receipt.ownerRepository, 'owner confirmation.ownerRepository', errors);
  revision(receipt.ownerRevision, 'owner confirmation.ownerRevision', errors);
  const confirmedAt = instant(receipt.confirmedAtUtc, 'owner confirmation.confirmedAtUtc', errors);
  text(receipt.statement, 'owner confirmation.statement', errors);
  match(qualificationId, expected.qualificationId, 'owner confirmation qualificationId', errors);
  match(owner, expected.owner, 'owner confirmation owner', errors);
  if (
    expected.ownerConfirmedAtUtc
    && confirmedAt
    && confirmedAt.toISOString() !== expected.ownerConfirmedAtUtc
  ) {
    errors.push('owner confirmation timestamp does not match ownerConfirmedAtUtc');
  }
  if (receipt.maintenanceStatus !== 'active') reasons.push('owner-maintenance-not-active');
  return { errors, reasons, confirmedAt };
}

export function validateRevocationAuthorityEvidence(receipt, expected = {}) {
  const errors = [];
  const reasons = [];
  const fields = [
    'schema', 'evidenceClass', 'qualificationId', 'owner', 'authorityId',
    'harnessRevision', 'environmentFingerprint', 'capturedAtUtc', 'result', 'observations',
  ];
  if (!exactKeys(receipt, fields, 'authority evidence', errors)) {
    return {
      errors,
      reasons: ['authority-evidence-invalid'],
      capturedAt: null,
      observations: null,
    };
  }
  if (receipt.schema !== EVIDENCE_SCHEMA) errors.push(`authority evidence.schema must be ${EVIDENCE_SCHEMA}`);
  const evidenceClass = text(receipt.evidenceClass, 'authority evidence.evidenceClass', errors);
  if (evidenceClass && !REVOCATION_AUTHORITY_EVIDENCE_CLASSES.includes(evidenceClass)) {
    errors.push('authority evidence.evidenceClass is not recognized');
  }
  const qualificationId = text(receipt.qualificationId, 'authority evidence.qualificationId', errors);
  const owner = text(receipt.owner, 'authority evidence.owner', errors);
  const authorityId = text(receipt.authorityId, 'authority evidence.authorityId', errors);
  const harnessRevision = revision(
    receipt.harnessRevision,
    'authority evidence.harnessRevision',
    errors
  );
  const environmentFingerprint = sha256(
    receipt.environmentFingerprint,
    'authority evidence.environmentFingerprint',
    errors
  );
  const capturedAt = instant(receipt.capturedAtUtc, 'authority evidence.capturedAtUtc', errors);
  match(evidenceClass, expected.evidenceClass, 'authority evidence class', errors);
  match(qualificationId, expected.qualificationId, 'authority evidence qualificationId', errors);
  match(owner, expected.owner, 'authority evidence owner', errors);
  match(authorityId, expected.authorityId, 'authority evidence authorityId', errors);
  match(harnessRevision, expected.harnessRevision, 'authority evidence harnessRevision', errors);
  match(
    environmentFingerprint,
    expected.environmentFingerprint,
    'authority evidence environmentFingerprint',
    errors
  );
  let claimedPassed = null;
  if (exactKeys(receipt.result, ['passed'], 'authority evidence.result', errors)) {
    if (typeof receipt.result.passed !== 'boolean') {
      errors.push('authority evidence.result.passed must be boolean');
    } else {
      claimedPassed = receipt.result.passed;
    }
  }
  const derivedPassed = evidenceClass
    ? validateObservations(evidenceClass, receipt.observations, expected, errors)
    : false;
  if (claimedPassed !== null && claimedPassed !== derivedPassed) {
    errors.push(`${evidenceClass} result.passed does not match its observations`);
  }
  if (claimedPassed !== true || derivedPassed !== true) {
    reasons.push(`${evidenceClass || 'authority-evidence'}-not-passed`);
  }
  return {
    errors,
    reasons,
    capturedAt,
    observations: receipt.observations,
    harnessRevision,
    environmentFingerprint,
  };
}

export function computeRevocationAuthorityEvidenceSetDigest(evidenceReferences) {
  return computeCanonicalJsonSha256(evidenceReferences);
}

export function validateRevocationAuthorityPromotionEvidence(receipt, expected = {}) {
  const errors = [];
  const fields = [
    'schema',
    'qualificationId',
    'owner',
    'authorityId',
    'endpointUrl',
    'onlineKeyIds',
    'recoveryKeyIds',
    'durableStateStoreIds',
    'harnessRevision',
    'environmentFingerprint',
    'evidenceSetDigest',
    'decision',
    'authority',
    'reviewer',
    'reviewerRevision',
    'rationale',
    'promotedAtUtc',
    'qualifiedAtUtc',
    'expiresAtUtc',
  ];
  if (!exactKeys(receipt, fields, 'authority promotion evidence', errors)) {
    return { errors, promotedAt: null };
  }
  if (receipt.schema !== PROMOTION_SCHEMA) {
    errors.push(`authority promotion evidence.schema must be ${PROMOTION_SCHEMA}`);
  }
  const binding = {
    qualificationId: text(
      receipt.qualificationId,
      'authority promotion evidence.qualificationId',
      errors
    ),
    owner: text(receipt.owner, 'authority promotion evidence.owner', errors),
    authorityId: text(receipt.authorityId, 'authority promotion evidence.authorityId', errors),
    endpointUrl: text(receipt.endpointUrl, 'authority promotion evidence.endpointUrl', errors),
    harnessRevision: revision(
      receipt.harnessRevision,
      'authority promotion evidence.harnessRevision',
      errors
    ),
    environmentFingerprint: sha256(
      receipt.environmentFingerprint,
      'authority promotion evidence.environmentFingerprint',
      errors
    ),
  };
  for (const [field, actual] of Object.entries(binding)) {
    match(actual, expected[field], `authority promotion evidence.${field}`, errors);
  }
  const onlineKeyIds = strings(
    receipt.onlineKeyIds,
    'authority promotion evidence.onlineKeyIds',
    errors
  );
  const recoveryKeyIds = strings(
    receipt.recoveryKeyIds,
    'authority promotion evidence.recoveryKeyIds',
    errors
  );
  matchList(
    onlineKeyIds,
    expected.onlineKeyIds,
    'authority promotion evidence.onlineKeyIds',
    errors
  );
  matchList(
    recoveryKeyIds,
    expected.recoveryKeyIds,
    'authority promotion evidence.recoveryKeyIds',
    errors
  );
  exactKeys(
    receipt.durableStateStoreIds,
    ['browser', 'node'],
    'authority promotion evidence.durableStateStoreIds',
    errors
  );
  for (const host of ['browser', 'node']) {
    const storeId = text(
      receipt.durableStateStoreIds?.[host],
      `authority promotion evidence.durableStateStoreIds.${host}`,
      errors
    );
    match(
      storeId,
      expected.durableStateStoreIds?.[host],
      `authority promotion evidence.durableStateStoreIds.${host}`,
      errors
    );
  }
  const evidenceSetDigest = sha256(
    receipt.evidenceSetDigest,
    'authority promotion evidence.evidenceSetDigest',
    errors
  );
  match(
    evidenceSetDigest,
    expected.evidenceSetDigest,
    'authority promotion evidence.evidenceSetDigest',
    errors
  );
  if (receipt.decision !== 'promote-production-authority') {
    errors.push('authority promotion evidence.decision must be promote-production-authority');
  }
  if (receipt.authority !== 'human') {
    errors.push('authority promotion evidence.authority must be human');
  }
  text(receipt.reviewer, 'authority promotion evidence.reviewer', errors);
  revision(receipt.reviewerRevision, 'authority promotion evidence.reviewerRevision', errors);
  text(receipt.rationale, 'authority promotion evidence.rationale', errors);
  const promotedAt = instant(
    receipt.promotedAtUtc,
    'authority promotion evidence.promotedAtUtc',
    errors
  );
  const qualifiedAt = instant(
    receipt.qualifiedAtUtc,
    'authority promotion evidence.qualifiedAtUtc',
    errors
  );
  const expiresAt = instant(
    receipt.expiresAtUtc,
    'authority promotion evidence.expiresAtUtc',
    errors
  );
  match(
    receipt.qualifiedAtUtc,
    expected.qualifiedAtUtc,
    'authority promotion evidence.qualifiedAtUtc',
    errors
  );
  match(
    receipt.expiresAtUtc,
    expected.expiresAtUtc,
    'authority promotion evidence.expiresAtUtc',
    errors
  );
  if (promotedAt && qualifiedAt && promotedAt.getTime() < qualifiedAt.getTime()) {
    errors.push('authority promotion evidence.promotedAtUtc must not predate qualifiedAtUtc');
  }
  if (qualifiedAt && expiresAt && expiresAt.getTime() <= qualifiedAt.getTime()) {
    errors.push('authority promotion evidence.expiresAtUtc must follow qualifiedAtUtc');
  }
  if (promotedAt && expiresAt && promotedAt.getTime() >= expiresAt.getTime()) {
    errors.push('authority promotion evidence.promotedAtUtc must predate expiresAtUtc');
  }
  return { errors, promotedAt };
}
