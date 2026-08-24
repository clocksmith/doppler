export const PACK_RELEASE_SCHEMA_ID = 'doppler.pack-release/v1';
export const PACK_STATE_SNAPSHOT_SCHEMA_ID = 'doppler.pack-state-snapshot/v1';

const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const ID_PATTERN = /^[a-z][a-z0-9-]*$/;
const SEMVER_PATTERN = /^[0-9]+\.[0-9]+\.[0-9]+$/;
const REJECTION_TYPES = new Set([
  'acceptance-failed',
  'application-gate-failed',
  'artifact-invalid',
  'evidence-expired',
  'migration-required',
  'revoked',
  'unsupported-device',
]);

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function requireObject(value, label, errors) {
  if (!isObject(value)) {
    errors.push(`${label} must be an object.`);
    return false;
  }
  return true;
}

function requireExactKeys(value, allowed, label, errors) {
  if (!isObject(value)) return;
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) errors.push(`${label}.${key} is not allowed.`);
  }
}

function requireString(value, label, errors) {
  if (typeof value !== 'string' || !value.trim()) errors.push(`${label} must be a non-empty string.`);
}

function requireId(value, label, errors) {
  if (!ID_PATTERN.test(value || '')) errors.push(`${label} must be a kebab-case identifier.`);
}

function requireDigest(value, label, errors) {
  if (!SHA256_PATTERN.test(value || '')) errors.push(`${label} must be a SHA-256 digest.`);
}

function validateIdentity(value, label, errors) {
  if (!requireObject(value, label, errors)) return;
  requireExactKeys(value, new Set(['id', 'digest']), label, errors);
  requireId(value.id, `${label}.id`, errors);
  requireDigest(value.digest, `${label}.digest`, errors);
}

function validateSource(source, errors) {
  if (!requireObject(source, 'release.source', errors)) return;
  requireExactKeys(source, new Set(['repository', 'revision', 'revisionDigest', 'provenanceDigest', 'license']), 'release.source', errors);
  requireString(source.repository, 'release.source.repository', errors);
  requireString(source.revision, 'release.source.revision', errors);
  requireDigest(source.revisionDigest, 'release.source.revisionDigest', errors);
  requireDigest(source.provenanceDigest, 'release.source.provenanceDigest', errors);
  const license = source.license;
  if (!requireObject(license, 'release.source.license', errors)) return;
  requireExactKeys(license, new Set(['spdxId', 'name', 'sourceUrl', 'textDigest']), 'release.source.license', errors);
  requireString(license.spdxId, 'release.source.license.spdxId', errors);
  requireString(license.name, 'release.source.license.name', errors);
  requireString(license.sourceUrl, 'release.source.license.sourceUrl', errors);
  requireDigest(license.textDigest, 'release.source.license.textDigest', errors);
}

function validateApplication(application, errors) {
  if (!requireObject(application, 'release.application', errors)) return;
  requireExactKeys(application, new Set([
    'applicationId', 'applicationRevision', 'applicationRevisionDigest', 'workload', 'oracle',
  ]), 'release.application', errors);
  requireId(application.applicationId, 'release.application.applicationId', errors);
  requireString(application.applicationRevision, 'release.application.applicationRevision', errors);
  requireDigest(application.applicationRevisionDigest, 'release.application.applicationRevisionDigest', errors);
  validateIdentity(application.workload, 'release.application.workload', errors);
  validateIdentity(application.oracle, 'release.application.oracle', errors);
}

function validateExclusions(exclusions, errors) {
  if (!requireObject(exclusions, 'release.exclusions', errors)) return;
  requireExactKeys(exclusions, new Set(['rejectionTypes', 'known']), 'release.exclusions', errors);
  if (!Array.isArray(exclusions.rejectionTypes) || exclusions.rejectionTypes.length === 0) {
    errors.push('release.exclusions.rejectionTypes must be a non-empty array.');
  } else {
    const seen = new Set();
    for (const [index, code] of exclusions.rejectionTypes.entries()) {
      if (!REJECTION_TYPES.has(code)) errors.push(`release.exclusions.rejectionTypes[${index}] is unsupported.`);
      if (seen.has(code)) errors.push(`release.exclusions.rejectionTypes contains duplicate "${code}".`);
      seen.add(code);
    }
  }
  if (!Array.isArray(exclusions.known)) {
    errors.push('release.exclusions.known must be an array.');
    return;
  }
  for (const [index, exclusion] of exclusions.known.entries()) {
    const label = `release.exclusions.known[${index}]`;
    if (!requireObject(exclusion, label, errors)) continue;
    requireExactKeys(exclusion, new Set(['code', 'scope', 'reason', 'evidenceDigest']), label, errors);
    if (!REJECTION_TYPES.has(exclusion.code)) errors.push(`${label}.code is unsupported.`);
    if (!exclusions.rejectionTypes.includes(exclusion.code)) errors.push(`${label}.code is not declared in rejectionTypes.`);
    requireString(exclusion.scope, `${label}.scope`, errors);
    requireString(exclusion.reason, `${label}.reason`, errors);
    requireDigest(exclusion.evidenceDigest, `${label}.evidenceDigest`, errors);
  }
}

function validatePackRef(value, label, errors) {
  if (!requireObject(value, label, errors)) return;
  requireExactKeys(value, new Set(['packId', 'semanticRoot']), label, errors);
  requireString(value.packId, `${label}.packId`, errors);
  requireDigest(value.semanticRoot, `${label}.semanticRoot`, errors);
}

function validateLifecycle(lifecycle, errors) {
  if (!requireObject(lifecycle, 'release.lifecycle', errors)) return;
  requireExactKeys(lifecycle, new Set(['releaseVersion', 'supersedes', 'migration', 'failedUpgrade']), 'release.lifecycle', errors);
  if (!SEMVER_PATTERN.test(lifecycle.releaseVersion || '')) errors.push('release.lifecycle.releaseVersion must be semantic versioning.');
  if (lifecycle.supersedes !== null) validatePackRef(lifecycle.supersedes, 'release.lifecycle.supersedes', errors);
  if (lifecycle.migration !== null) {
    const migration = lifecycle.migration;
    if (requireObject(migration, 'release.lifecycle.migration', errors)) {
      requireExactKeys(migration, new Set(['id', 'policyDigest', 'required']), 'release.lifecycle.migration', errors);
      requireId(migration.id, 'release.lifecycle.migration.id', errors);
      requireDigest(migration.policyDigest, 'release.lifecycle.migration.policyDigest', errors);
      if (typeof migration.required !== 'boolean') errors.push('release.lifecycle.migration.required must be boolean.');
    }
  }
  const failedUpgrade = lifecycle.failedUpgrade;
  if (!requireObject(failedUpgrade, 'release.lifecycle.failedUpgrade', errors)) return;
  requireExactKeys(failedUpgrade, new Set(['preservePrevious', 'previousPackId', 'previousSemanticRoot']), 'release.lifecycle.failedUpgrade', errors);
  if (failedUpgrade.preservePrevious !== true) errors.push('release.lifecycle.failedUpgrade.preservePrevious must be true.');
  requireString(failedUpgrade.previousPackId, 'release.lifecycle.failedUpgrade.previousPackId', errors);
  requireDigest(failedUpgrade.previousSemanticRoot, 'release.lifecycle.failedUpgrade.previousSemanticRoot', errors);
  if (lifecycle.supersedes !== null
    && (failedUpgrade.previousPackId !== lifecycle.supersedes.packId
      || failedUpgrade.previousSemanticRoot !== lifecycle.supersedes.semanticRoot)) {
    errors.push('release.lifecycle.failedUpgrade must preserve the superseded Pack.');
  }
}

function validateRevocation(revocation, errors) {
  if (!requireObject(revocation, 'release.revocation', errors)) return;
  requireExactKeys(revocation, new Set(['authorityId', 'policyDigest', 'offlineExpirySeconds', 'failClosedAfterExpiry']), 'release.revocation', errors);
  requireId(revocation.authorityId, 'release.revocation.authorityId', errors);
  requireDigest(revocation.policyDigest, 'release.revocation.policyDigest', errors);
  if (!Number.isSafeInteger(revocation.offlineExpirySeconds) || revocation.offlineExpirySeconds < 1) {
    errors.push('release.revocation.offlineExpirySeconds must be a positive integer.');
  }
  if (revocation.failClosedAfterExpiry !== true) errors.push('release.revocation.failClosedAfterExpiry must be true.');
}

function validateStateSnapshot(stateSnapshot, targetIds, errors) {
  if (!requireObject(stateSnapshot, 'release.stateSnapshot', errors)) return;
  requireExactKeys(stateSnapshot, new Set(['schema', 'format', 'identityDigest', 'portableAcrossTargetIds']), 'release.stateSnapshot', errors);
  if (stateSnapshot.schema !== PACK_STATE_SNAPSHOT_SCHEMA_ID) {
    errors.push(`release.stateSnapshot.schema must be "${PACK_STATE_SNAPSHOT_SCHEMA_ID}".`);
  }
  requireString(stateSnapshot.format, 'release.stateSnapshot.format', errors);
  requireDigest(stateSnapshot.identityDigest, 'release.stateSnapshot.identityDigest', errors);
  if (!Array.isArray(stateSnapshot.portableAcrossTargetIds) || stateSnapshot.portableAcrossTargetIds.length === 0) {
    errors.push('release.stateSnapshot.portableAcrossTargetIds must be a non-empty array.');
    return;
  }
  const declaredTargets = new Set(targetIds);
  for (const [index, targetId] of stateSnapshot.portableAcrossTargetIds.entries()) {
    requireId(targetId, `release.stateSnapshot.portableAcrossTargetIds[${index}]`, errors);
    if (declaredTargets.size > 0 && !declaredTargets.has(targetId)) {
      errors.push(`release.stateSnapshot target "${targetId}" is not carried by the Pack.`);
    }
  }
}

export function validatePackReleaseContract(release, options = {}) {
  const errors = [];
  if (!requireObject(release, 'release', errors)) return { ok: false, errors };
  requireExactKeys(release, new Set(['schema', 'source', 'application', 'exclusions', 'lifecycle', 'revocation', 'stateSnapshot']), 'release', errors);
  if (release.schema !== PACK_RELEASE_SCHEMA_ID) errors.push(`release.schema must be "${PACK_RELEASE_SCHEMA_ID}".`);
  validateSource(release.source, errors);
  validateApplication(release.application, errors);
  validateExclusions(release.exclusions, errors);
  validateLifecycle(release.lifecycle, errors);
  validateRevocation(release.revocation, errors);
  validateStateSnapshot(release.stateSnapshot, options.targetIds || [], errors);
  return { ok: errors.length === 0, errors };
}
