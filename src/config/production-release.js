import { sha256Hex } from '../formats/sha256.js';
import { stableSortObject } from '../formats/stable-sort-object.js';

export const PRODUCTION_RELEASE_SCHEMA_ID = 'doppler.production-release/v1';
export const PRODUCTION_RELEASE_SCHEMA_VERSION = 1;

const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const ID_PATTERN = /^[a-z][a-z0-9-]*$/;
const RELEASE_ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*-release-[0-9a-f]{16}$/;
const EVIDENCE_CLASSES = new Set(['reference-fixture', 'external-candidate', 'external-production']);
const OPERATING_SYSTEMS = new Set(['windows', 'macos']);
const ARCHITECTURES = new Set(['x64', 'arm64']);
const GPU_VENDORS = new Set(['amd', 'apple', 'intel', 'nvidia']);

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function canonicalJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function requireObject(value, label, errors) {
  if (!isObject(value)) {
    errors.push(`${label} must be an object.`);
    return false;
  }
  return true;
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

function requireRepoPath(value, label, errors) {
  requireString(value, label, errors);
  if (typeof value === 'string' && (value.startsWith('/') || value.split('/').includes('..'))) {
    errors.push(`${label} must be a repository-relative path.`);
  }
}

function requireExactKeys(value, allowed, label, errors) {
  if (!isObject(value)) return;
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) errors.push(`${label}.${key} is not allowed.`);
  }
}

function requireStringArray(value, allowed, label, errors) {
  if (!Array.isArray(value) || value.length === 0) {
    errors.push(`${label} must be a non-empty array.`);
    return;
  }
  for (const [index, entry] of value.entries()) {
    if (typeof entry !== 'string' || !entry.trim()) errors.push(`${label}[${index}] must be a non-empty string.`);
    else if (allowed && !allowed.has(entry)) errors.push(`${label}[${index}] is unsupported.`);
  }
}

function validateCandidate(candidate, errors) {
  if (!requireObject(candidate, 'candidate', errors)) return;
  requireExactKeys(candidate, new Set(['logicalModelId', 'sourceRevision', 'sourceRevisionDigest', 'packPath', 'packSemanticRoot']), 'candidate', errors);
  requireString(candidate.logicalModelId, 'candidate.logicalModelId', errors);
  requireString(candidate.sourceRevision, 'candidate.sourceRevision', errors);
  requireDigest(candidate.sourceRevisionDigest, 'candidate.sourceRevisionDigest', errors);
  requireRepoPath(candidate.packPath, 'candidate.packPath', errors);
  requireDigest(candidate.packSemanticRoot, 'candidate.packSemanticRoot', errors);
}

function validateApplication(application, errors) {
  if (!requireObject(application, 'application', errors)) return;
  requireExactKeys(application, new Set(['applicationId', 'platform', 'revision', 'revisionDigest', 'rendererEntry', 'mainEntry']), 'application', errors);
  requireId(application.applicationId, 'application.applicationId', errors);
  if (application.platform !== 'electron') errors.push('application.platform must be "electron".');
  requireString(application.revision, 'application.revision', errors);
  requireDigest(application.revisionDigest, 'application.revisionDigest', errors);
  requireRepoPath(application.rendererEntry, 'application.rendererEntry', errors);
  requireRepoPath(application.mainEntry, 'application.mainEntry', errors);
}

function validateIdentity(identity, label, errors) {
  if (!requireObject(identity, label, errors)) return;
  requireExactKeys(identity, new Set(['id', 'digest']), label, errors);
  requireId(identity.id, `${label}.id`, errors);
  requireDigest(identity.digest, `${label}.digest`, errors);
}

function validateMaxThreshold(threshold, label, errors) {
  if (!requireObject(threshold, label, errors)) return;
  requireExactKeys(threshold, new Set(['maximum']), label, errors);
  if (typeof threshold.maximum !== 'number' || threshold.maximum < 0) {
    errors.push(`${label}.maximum must be a non-negative number.`);
  }
}

function validateAcceptance(acceptance, errors) {
  if (!requireObject(acceptance, 'acceptance', errors)) return;
  requireExactKeys(acceptance, new Set(['workload', 'oracle', 'tests', 'thresholds', 'incumbentControl']), 'acceptance', errors);
  validateIdentity(acceptance.workload, 'acceptance.workload', errors);
  validateIdentity(acceptance.oracle, 'acceptance.oracle', errors);
  if (!Array.isArray(acceptance.tests) || acceptance.tests.length === 0) {
    errors.push('acceptance.tests must be a non-empty array.');
  } else {
    for (const [index, test] of acceptance.tests.entries()) {
      const label = `acceptance.tests[${index}]`;
      if (!requireObject(test, label, errors)) continue;
      requireExactKeys(test, new Set(['id', 'command', 'workdir', 'timeoutMs', 'evidenceSchema']), label, errors);
      requireId(test.id, `${label}.id`, errors);
      requireStringArray(test.command, null, `${label}.command`, errors);
      requireRepoPath(test.workdir, `${label}.workdir`, errors);
      if (!Number.isSafeInteger(test.timeoutMs) || test.timeoutMs < 1) errors.push(`${label}.timeoutMs must be a positive integer.`);
      requireString(test.evidenceSchema, `${label}.evidenceSchema`, errors);
    }
  }
  if (requireObject(acceptance.thresholds, 'acceptance.thresholds', errors)) {
    requireExactKeys(acceptance.thresholds, new Set(['quality', 'coldLatencyMs', 'warmLatencyMs', 'peakMemoryBytes', 'failureRate']), 'acceptance.thresholds', errors);
    const quality = acceptance.thresholds.quality;
    if (requireObject(quality, 'acceptance.thresholds.quality', errors)) {
      requireExactKeys(quality, new Set(['metric', 'minimum']), 'acceptance.thresholds.quality', errors);
      requireId(quality.metric, 'acceptance.thresholds.quality.metric', errors);
      if (typeof quality.minimum !== 'number') errors.push('acceptance.thresholds.quality.minimum must be a number.');
    }
    validateMaxThreshold(acceptance.thresholds.coldLatencyMs, 'acceptance.thresholds.coldLatencyMs', errors);
    validateMaxThreshold(acceptance.thresholds.warmLatencyMs, 'acceptance.thresholds.warmLatencyMs', errors);
    validateMaxThreshold(acceptance.thresholds.peakMemoryBytes, 'acceptance.thresholds.peakMemoryBytes', errors);
    validateMaxThreshold(acceptance.thresholds.failureRate, 'acceptance.thresholds.failureRate', errors);
  }
  const control = acceptance.incumbentControl;
  if (requireObject(control, 'acceptance.incumbentControl', errors)) {
    requireExactKeys(control, new Set(['providerId', 'artifactRevision', 'executionDigest']), 'acceptance.incumbentControl', errors);
    requireId(control.providerId, 'acceptance.incumbentControl.providerId', errors);
    requireString(control.artifactRevision, 'acceptance.incumbentControl.artifactRevision', errors);
    requireDigest(control.executionDigest, 'acceptance.incumbentControl.executionDigest', errors);
  }
}

function validateSupportedDevices(supportedDevices, errors) {
  if (!requireObject(supportedDevices, 'supportedDevices', errors)) return;
  requireExactKeys(supportedDevices, new Set(['policyId', 'policyDigest', 'targets', 'receiptMode']), 'supportedDevices', errors);
  requireId(supportedDevices.policyId, 'supportedDevices.policyId', errors);
  requireDigest(supportedDevices.policyDigest, 'supportedDevices.policyDigest', errors);
  if (supportedDevices.receiptMode !== 'customer-operated-agent') {
    errors.push('supportedDevices.receiptMode must be "customer-operated-agent".');
  }
  if (!Array.isArray(supportedDevices.targets) || supportedDevices.targets.length < 2) {
    errors.push('supportedDevices.targets must include Windows and macOS targets.');
    return;
  }
  const operatingSystems = new Set();
  for (const [index, target] of supportedDevices.targets.entries()) {
    const label = `supportedDevices.targets[${index}]`;
    if (!requireObject(target, label, errors)) continue;
    requireExactKeys(target, new Set(['id', 'os', 'osVersionRange', 'architectures', 'electronVersionRange', 'gpuVendors', 'driverPolicy']), label, errors);
    requireId(target.id, `${label}.id`, errors);
    if (!OPERATING_SYSTEMS.has(target.os)) errors.push(`${label}.os is unsupported.`);
    else operatingSystems.add(target.os);
    requireString(target.osVersionRange, `${label}.osVersionRange`, errors);
    requireStringArray(target.architectures, ARCHITECTURES, `${label}.architectures`, errors);
    requireString(target.electronVersionRange, `${label}.electronVersionRange`, errors);
    requireStringArray(target.gpuVendors, GPU_VENDORS, `${label}.gpuVendors`, errors);
    if (target.driverPolicy !== 'exact-receipt-required') errors.push(`${label}.driverPolicy must require exact receipts.`);
  }
  for (const os of OPERATING_SYSTEMS) {
    if (!operatingSystems.has(os)) errors.push(`supportedDevices.targets must include ${os}.`);
  }
}

function validateReleaseTarget(target, label, requireAuthority, errors) {
  if (!requireObject(target, label, errors)) return;
  const allowed = new Set(['releaseId', 'packSemanticRoot']);
  if (requireAuthority) allowed.add('authority');
  requireExactKeys(target, allowed, label, errors);
  requireString(target.releaseId, `${label}.releaseId`, errors);
  requireDigest(target.packSemanticRoot, `${label}.packSemanticRoot`, errors);
  if (requireAuthority && target.authority !== 'customer') errors.push(`${label}.authority must be "customer".`);
}

function validateRollout(rollout, errors) {
  if (!requireObject(rollout, 'rollout', errors)) return;
  requireExactKeys(rollout, new Set(['rulesDigest', 'activationAuthority', 'selfPromotionAllowed', 'stages']), 'rollout', errors);
  requireDigest(rollout.rulesDigest, 'rollout.rulesDigest', errors);
  if (rollout.activationAuthority !== 'customer') errors.push('rollout.activationAuthority must be "customer".');
  if (rollout.selfPromotionAllowed !== false) errors.push('rollout.selfPromotionAllowed must be false.');
  if (!Array.isArray(rollout.stages) || rollout.stages.length === 0) {
    errors.push('rollout.stages must be a non-empty array.');
    return;
  }
  for (const [index, stage] of rollout.stages.entries()) {
    const label = `rollout.stages[${index}]`;
    if (!requireObject(stage, label, errors)) continue;
    requireExactKeys(stage, new Set(['id', 'eligibleFleetPercent', 'requiredObservationDigest']), label, errors);
    requireId(stage.id, `${label}.id`, errors);
    if (typeof stage.eligibleFleetPercent !== 'number' || stage.eligibleFleetPercent <= 0 || stage.eligibleFleetPercent > 100) {
      errors.push(`${label}.eligibleFleetPercent must be in (0, 100].`);
    }
    requireDigest(stage.requiredObservationDigest, `${label}.requiredObservationDigest`, errors);
  }
}

function validateRevocation(revocation, errors) {
  if (!requireObject(revocation, 'revocation', errors)) return;
  requireExactKeys(revocation, new Set(['authorityId', 'policyDigest', 'offlineExpirySeconds', 'failClosedAfterExpiry']), 'revocation', errors);
  requireId(revocation.authorityId, 'revocation.authorityId', errors);
  requireDigest(revocation.policyDigest, 'revocation.policyDigest', errors);
  if (!Number.isSafeInteger(revocation.offlineExpirySeconds) || revocation.offlineExpirySeconds < 1) {
    errors.push('revocation.offlineExpirySeconds must be a positive integer.');
  }
  if (revocation.failClosedAfterExpiry !== true) errors.push('revocation.failClosedAfterExpiry must be true.');
}

function validateClaimBoundary(release, errors) {
  const boundary = release.claimBoundary;
  if (!requireObject(boundary, 'claimBoundary', errors)) return;
  requireExactKeys(boundary, new Set(['externalCustomer', 'commercialClaimAllowed']), 'claimBoundary', errors);
  if (typeof boundary.externalCustomer !== 'boolean') errors.push('claimBoundary.externalCustomer must be boolean.');
  if (typeof boundary.commercialClaimAllowed !== 'boolean') errors.push('claimBoundary.commercialClaimAllowed must be boolean.');
  if (release.evidenceClass === 'reference-fixture'
    && (boundary.externalCustomer !== false || boundary.commercialClaimAllowed !== false)) {
    errors.push('reference-fixture releases cannot claim an external customer or commercial evidence.');
  }
  if (boundary.commercialClaimAllowed && release.evidenceClass !== 'external-production') {
    errors.push('commercial claims require evidenceClass "external-production".');
  }
}

export function getProductionReleaseSemanticPayload(release) {
  const { releaseId: _releaseId, ...payload } = release;
  return payload;
}

export function hashProductionRelease(release) {
  return `sha256:${sha256Hex(canonicalJson(getProductionReleaseSemanticPayload(release)))}`;
}

export function validateProductionRelease(release) {
  const errors = [];
  if (!isObject(release)) return { ok: false, errors: ['Doppler Production Release v1 must be a non-null object.'] };
  requireExactKeys(release, new Set([
    'schema', 'schemaVersion', 'releaseId', 'createdAtUtc', 'evidenceClass', 'candidate', 'application',
    'acceptance', 'supportedDevices', 'previousRelease', 'rollout', 'rollback', 'revocation', 'dataCustody',
    'claimBoundary',
  ]), 'release', errors);
  if (release.schema !== PRODUCTION_RELEASE_SCHEMA_ID) errors.push(`schema must be "${PRODUCTION_RELEASE_SCHEMA_ID}".`);
  if (release.schemaVersion !== PRODUCTION_RELEASE_SCHEMA_VERSION) errors.push(`schemaVersion must be ${PRODUCTION_RELEASE_SCHEMA_VERSION}.`);
  if (!RELEASE_ID_PATTERN.test(release.releaseId || '')) errors.push('releaseId must be a derived production release identifier.');
  requireString(release.createdAtUtc, 'createdAtUtc', errors);
  if (typeof release.createdAtUtc === 'string' && Number.isNaN(Date.parse(release.createdAtUtc))) errors.push('createdAtUtc must be an ISO date-time.');
  if (!EVIDENCE_CLASSES.has(release.evidenceClass)) errors.push('evidenceClass is unsupported.');
  validateCandidate(release.candidate, errors);
  validateApplication(release.application, errors);
  validateAcceptance(release.acceptance, errors);
  validateSupportedDevices(release.supportedDevices, errors);
  validateReleaseTarget(release.previousRelease, 'previousRelease', false, errors);
  validateRollout(release.rollout, errors);
  validateReleaseTarget(release.rollback, 'rollback', true, errors);
  validateRevocation(release.revocation, errors);
  const custody = release.dataCustody;
  if (requireObject(custody, 'dataCustody', errors)) {
    requireExactKeys(custody, new Set(['policyDigest', 'promptRetention', 'outputRetention', 'telemetryExport']), 'dataCustody', errors);
    requireDigest(custody.policyDigest, 'dataCustody.policyDigest', errors);
    if (!new Set(['none', 'customer-controlled']).has(custody.promptRetention)) errors.push('dataCustody.promptRetention is unsupported.');
    if (!new Set(['none', 'customer-controlled']).has(custody.outputRetention)) errors.push('dataCustody.outputRetention is unsupported.');
    if (!new Set(['none', 'customer-approved-redacted']).has(custody.telemetryExport)) errors.push('dataCustody.telemetryExport is unsupported.');
  }
  validateClaimBoundary(release, errors);
  const semanticDigest = hashProductionRelease(release);
  const expectedReleaseId = `${release.application?.applicationId || 'invalid'}-${release.candidate?.logicalModelId || 'invalid'}-release-${semanticDigest.slice(7, 23)}`;
  if (release.releaseId !== expectedReleaseId) errors.push(`releaseId must be derived from the semantic payload (${expectedReleaseId}).`);
  return { ok: errors.length === 0, errors };
}

export function assertProductionRelease(release) {
  const validation = validateProductionRelease(release);
  if (!validation.ok) throw new Error(`Invalid Doppler Production Release v1: ${validation.errors.join('; ')}`);
  return release;
}
