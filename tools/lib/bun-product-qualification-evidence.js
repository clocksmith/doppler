import { computeCanonicalJsonSha256 } from './canonical-json.js';

const EVIDENCE_SCHEMA = 'doppler.bun-product-qualification-evidence/v1';
const PROMOTION_SCHEMA = 'doppler.bun-product-promotion-evidence/v1';
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const REVISION_PATTERN = /^[0-9a-f]{40}$/;

export const BUN_QUALIFICATION_EVIDENCE_CLASSES = Object.freeze([
  'surfaceConformance',
  'lifecycle',
  'correctnessQuality',
  'reliability',
  'memory',
  'coldWarmResponse',
  'incumbentControl',
  'upgradeRequalification',
  'rollbackRevocation',
]);

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

function text(value, label, errors, nullable = false) {
  if (nullable && value === null) return null;
  const normalized = normalizeText(value);
  if (!normalized) errors.push(`${label} must be a non-empty string${nullable ? ' or null' : ''}`);
  return normalized || null;
}

function sha256(value, label, errors, nullable = false) {
  const normalized = text(value, label, errors, nullable)?.toLowerCase() ?? null;
  if (normalized && !SHA256_PATTERN.test(normalized)) {
    errors.push(`${label} must be a SHA-256 identity${nullable ? ' or null' : ''}`);
  }
  return normalized;
}

function revision(value, label, errors) {
  const normalized = text(value, label, errors)?.toLowerCase() ?? null;
  if (normalized && !REVISION_PATTERN.test(normalized)) {
    errors.push(`${label} must be a lowercase 40-hex revision`);
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
  if (typeof value !== 'boolean') errors.push(`${label} must be boolean`);
  return value === true;
}

function integer(value, label, errors, minimum = 0) {
  if (!Number.isInteger(value) || value < minimum) {
    errors.push(`${label} must be an integer of at least ${minimum}`);
    return null;
  }
  return value;
}

function number(value, label, errors, minimum = 0) {
  if (!Number.isFinite(value) || value < minimum) {
    errors.push(`${label} must be a finite number of at least ${minimum}`);
    return null;
  }
  return value;
}

function match(actual, expected, label, errors) {
  if (expected != null && actual != null && actual !== expected) {
    errors.push(`${label} does not match the qualification`);
  }
}

function validateBinding(receipt, expected, label, errors) {
  const qualificationId = text(receipt.qualificationId, `${label}.qualificationId`, errors);
  const workload = text(receipt.workload, `${label}.workload`, errors);
  const logicalModelId = text(receipt.logicalModelId, `${label}.logicalModelId`, errors);
  const manifestVariantId = text(receipt.manifestVariantId, `${label}.manifestVariantId`, errors);
  const resolvedArtifactVariantId = sha256(
    receipt.resolvedArtifactVariantId,
    `${label}.resolvedArtifactVariantId`,
    errors
  );
  const resolvedExecutionId = sha256(
    receipt.resolvedExecutionId,
    `${label}.resolvedExecutionId`,
    errors
  );
  const bunVersion = text(receipt.bunVersion, `${label}.bunVersion`, errors);
  const webgpuImplementationId = text(
    receipt.webgpuImplementationId,
    `${label}.webgpuImplementationId`,
    errors
  );
  const providerId = text(receipt.providerId, `${label}.providerId`, errors);
  revision(receipt.harnessRevision, `${label}.harnessRevision`, errors);
  sha256(receipt.environmentFingerprint, `${label}.environmentFingerprint`, errors);
  for (const [field, actual] of Object.entries({
    qualificationId,
    workload,
    logicalModelId,
    manifestVariantId,
    resolvedArtifactVariantId,
    resolvedExecutionId,
    bunVersion,
    webgpuImplementationId,
    providerId,
  })) {
    match(actual, expected[field], `${label}.${field}`, errors);
  }
}

function surfacePassed(value, errors) {
  const fields = [
    'rootApiPassed',
    'cliPassed',
    'semanticParityPassed',
    'unsupportedFallbackUsed',
  ];
  exactKeys(value, fields, 'observations', errors);
  const results = fields.slice(0, 3).map((field) => boolean(value?.[field], `observations.${field}`, errors));
  if (typeof value?.unsupportedFallbackUsed !== 'boolean') {
    errors.push('observations.unsupportedFallbackUsed must be boolean');
  }
  return results.every(Boolean) && value?.unsupportedFallbackUsed === false;
}

function lifecyclePassed(value, errors) {
  const fields = [
    'loadPassed',
    'executePassed',
    'unloadPassed',
    'repeatedSessions',
    'minimumRepeatedSessions',
  ];
  exactKeys(value, fields, 'observations', errors);
  const stages = fields.slice(0, 3).map((field) => boolean(value?.[field], `observations.${field}`, errors));
  const actual = integer(value?.repeatedSessions, 'observations.repeatedSessions', errors, 1);
  const minimum = integer(
    value?.minimumRepeatedSessions,
    'observations.minimumRepeatedSessions',
    errors,
    1
  );
  return stages.every(Boolean) && actual !== null && minimum !== null && actual >= minimum;
}

function correctnessQualityPassed(value, expected, errors) {
  const fields = ['correctnessClass', 'correctnessPassed', 'heldOutSetDigest', 'qualityPassed'];
  exactKeys(value, fields, 'observations', errors);
  const correctnessClass = text(value?.correctnessClass, 'observations.correctnessClass', errors);
  match(correctnessClass, expected.correctnessClass, 'observations.correctnessClass', errors);
  sha256(value?.heldOutSetDigest, 'observations.heldOutSetDigest', errors);
  return boolean(value?.correctnessPassed, 'observations.correctnessPassed', errors)
    && boolean(value?.qualityPassed, 'observations.qualityPassed', errors);
}

function reliabilityPassed(value, errors) {
  const fields = [
    'attempts',
    'successes',
    'minimumSuccessRate',
    'crashes',
    'maximumCrashes',
    'ooms',
    'maximumOoms',
    'deviceLosses',
    'maximumDeviceLosses',
  ];
  exactKeys(value, fields, 'observations', errors);
  const attempts = integer(value?.attempts, 'observations.attempts', errors, 1);
  const successes = integer(value?.successes, 'observations.successes', errors);
  const rate = number(value?.minimumSuccessRate, 'observations.minimumSuccessRate', errors);
  const crashes = integer(value?.crashes, 'observations.crashes', errors);
  const maxCrashes = integer(value?.maximumCrashes, 'observations.maximumCrashes', errors);
  const ooms = integer(value?.ooms, 'observations.ooms', errors);
  const maxOoms = integer(value?.maximumOoms, 'observations.maximumOoms', errors);
  const losses = integer(value?.deviceLosses, 'observations.deviceLosses', errors);
  const maxLosses = integer(
    value?.maximumDeviceLosses,
    'observations.maximumDeviceLosses',
    errors
  );
  if (rate !== null && rate > 1) errors.push('observations.minimumSuccessRate must not exceed 1');
  if (attempts !== null && successes !== null && successes > attempts) {
    errors.push('observations.successes must not exceed attempts');
  }
  return attempts !== null
    && successes !== null
    && rate !== null
    && successes / attempts >= rate
    && crashes <= maxCrashes
    && ooms <= maxOoms
    && losses <= maxLosses;
}

function memoryPassed(value, errors) {
  exactKeys(value, ['peakBytes', 'budgetBytes'], 'observations', errors);
  const peak = integer(value?.peakBytes, 'observations.peakBytes', errors);
  const budget = integer(value?.budgetBytes, 'observations.budgetBytes', errors, 1);
  return peak !== null && budget !== null && peak <= budget;
}

function coldWarmPassed(value, errors) {
  const fields = [
    'sampleCount',
    'minimumSampleCount',
    'coldP50Ms',
    'coldP95Ms',
    'coldP95LimitMs',
    'warmP50Ms',
    'warmP95Ms',
    'warmP95LimitMs',
  ];
  exactKeys(value, fields, 'observations', errors);
  const samples = integer(value?.sampleCount, 'observations.sampleCount', errors, 1);
  const minimum = integer(value?.minimumSampleCount, 'observations.minimumSampleCount', errors, 1);
  const coldP50 = number(value?.coldP50Ms, 'observations.coldP50Ms', errors);
  const coldP95 = number(value?.coldP95Ms, 'observations.coldP95Ms', errors);
  const coldLimit = number(value?.coldP95LimitMs, 'observations.coldP95LimitMs', errors);
  const warmP50 = number(value?.warmP50Ms, 'observations.warmP50Ms', errors);
  const warmP95 = number(value?.warmP95Ms, 'observations.warmP95Ms', errors);
  const warmLimit = number(value?.warmP95LimitMs, 'observations.warmP95LimitMs', errors);
  if (coldP50 !== null && coldP95 !== null && coldP50 > coldP95) {
    errors.push('observations.coldP50Ms must not exceed coldP95Ms');
  }
  if (warmP50 !== null && warmP95 !== null && warmP50 > warmP95) {
    errors.push('observations.warmP50Ms must not exceed warmP95Ms');
  }
  return samples !== null
    && minimum !== null
    && samples >= minimum
    && coldP95 !== null
    && coldLimit !== null
    && coldP95 <= coldLimit
    && warmP95 !== null
    && warmLimit !== null
    && warmP95 <= warmLimit;
}

function incumbentPassed(value, errors) {
  const fields = [
    'outcome',
    'incumbentProviderId',
    'incumbentArtifactId',
    'discoveryReceiptDigest',
    'comparisonReceiptDigest',
    'correctnessComparable',
  ];
  exactKeys(value, fields, 'observations', errors);
  const outcome = text(value?.outcome, 'observations.outcome', errors);
  if (!['compared', 'no-eligible-incumbent'].includes(outcome)) {
    errors.push('observations.outcome is not recognized');
  }
  const providerId = text(
    value?.incumbentProviderId,
    'observations.incumbentProviderId',
    errors,
    true
  );
  const artifactId = text(
    value?.incumbentArtifactId,
    'observations.incumbentArtifactId',
    errors,
    true
  );
  sha256(value?.discoveryReceiptDigest, 'observations.discoveryReceiptDigest', errors);
  const comparison = sha256(
    value?.comparisonReceiptDigest,
    'observations.comparisonReceiptDigest',
    errors,
    true
  );
  if (typeof value?.correctnessComparable !== 'boolean') {
    errors.push('observations.correctnessComparable must be boolean');
  }
  if (outcome === 'compared') {
    return Boolean(providerId && artifactId && comparison && value?.correctnessComparable === true);
  }
  return outcome === 'no-eligible-incumbent'
    && providerId === null
    && artifactId === null
    && comparison === null
    && value?.correctnessComparable === false;
}

function upgradePassed(value, expected, errors) {
  const fields = [
    'fromBunVersion',
    'toBunVersion',
    'migrationSucceeded',
    'identityPreserved',
    'taskGatePassed',
  ];
  exactKeys(value, fields, 'observations', errors);
  const fromVersion = text(value?.fromBunVersion, 'observations.fromBunVersion', errors);
  const toVersion = text(value?.toBunVersion, 'observations.toBunVersion', errors);
  match(toVersion, expected.bunVersion, 'observations.toBunVersion', errors);
  if (fromVersion && toVersion && fromVersion === toVersion) {
    errors.push('observations.fromBunVersion must differ from toBunVersion');
  }
  return [
    'migrationSucceeded',
    'identityPreserved',
    'taskGatePassed',
  ].map((field) => boolean(value?.[field], `observations.${field}`, errors)).every(Boolean);
}

function rollbackPassed(value, errors) {
  const fields = [
    'knownSafeRuntimeRevision',
    'rollbackSucceeded',
    'revocationObserved',
    'taskGatePassed',
  ];
  exactKeys(value, fields, 'observations', errors);
  revision(value?.knownSafeRuntimeRevision, 'observations.knownSafeRuntimeRevision', errors);
  return [
    'rollbackSucceeded',
    'revocationObserved',
    'taskGatePassed',
  ].map((field) => boolean(value?.[field], `observations.${field}`, errors)).every(Boolean);
}

const EVIDENCE_VALIDATORS = Object.freeze({
  surfaceConformance: (value, expected, errors) => surfacePassed(value, errors),
  lifecycle: (value, expected, errors) => lifecyclePassed(value, errors),
  correctnessQuality: correctnessQualityPassed,
  reliability: (value, expected, errors) => reliabilityPassed(value, errors),
  memory: (value, expected, errors) => memoryPassed(value, errors),
  coldWarmResponse: (value, expected, errors) => coldWarmPassed(value, errors),
  incumbentControl: (value, expected, errors) => incumbentPassed(value, errors),
  upgradeRequalification: upgradePassed,
  rollbackRevocation: (value, expected, errors) => rollbackPassed(value, errors),
});

export function validateBunProductQualificationEvidence(receipt, expected = {}) {
  const errors = [];
  const reasons = [];
  const fields = [
    'schema',
    'evidenceClass',
    'qualificationId',
    'workload',
    'logicalModelId',
    'manifestVariantId',
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'bunVersion',
    'webgpuImplementationId',
    'providerId',
    'harnessRevision',
    'environmentFingerprint',
    'capturedAtUtc',
    'result',
    'observations',
  ];
  if (!exactKeys(receipt, fields, 'Bun qualification evidence', errors)) {
    return { errors, reasons: ['bun-evidence-invalid'], passed: null, capturedAt: null };
  }
  if (receipt.schema !== EVIDENCE_SCHEMA) {
    errors.push(`Bun qualification evidence.schema must be ${EVIDENCE_SCHEMA}`);
  }
  const evidenceClass = text(
    receipt.evidenceClass,
    'Bun qualification evidence.evidenceClass',
    errors
  );
  if (evidenceClass && !BUN_QUALIFICATION_EVIDENCE_CLASSES.includes(evidenceClass)) {
    errors.push('Bun qualification evidence.evidenceClass is not recognized');
  }
  match(
    evidenceClass,
    expected.evidenceClass,
    'Bun qualification evidence.evidenceClass',
    errors
  );
  validateBinding(receipt, expected, 'Bun qualification evidence', errors);
  const capturedAt = instant(
    receipt.capturedAtUtc,
    'Bun qualification evidence.capturedAtUtc',
    errors
  );
  let passed = null;
  if (exactKeys(receipt.result, ['passed'], 'Bun qualification evidence.result', errors)) {
    if (typeof receipt.result.passed !== 'boolean') {
      errors.push('Bun qualification evidence.result.passed must be boolean');
    }
    const validator = EVIDENCE_VALIDATORS[evidenceClass];
    const derivedPassed = validator ? validator(receipt.observations, expected, errors) : false;
    if (typeof receipt.result.passed === 'boolean' && receipt.result.passed !== derivedPassed) {
      errors.push('Bun qualification evidence.result.passed does not match observations');
    }
    passed = errors.length === 0 ? derivedPassed : null;
  }
  if (passed !== true) reasons.push(`${evidenceClass || 'bun'}-gate-failed`);
  return { errors, reasons, passed, capturedAt };
}

export function computeBunQualificationEvidenceSetDigest(evidenceReferences) {
  return computeCanonicalJsonSha256(evidenceReferences);
}

export function validateBunProductPromotionEvidence(receipt, expected = {}) {
  const errors = [];
  const fields = [
    'schema',
    'qualificationId',
    'workload',
    'logicalModelId',
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'bunVersion',
    'webgpuImplementationId',
    'providerId',
    'evidenceSetDigest',
    'decision',
    'authority',
    'promotedAtUtc',
    'qualifiedAtUtc',
    'expiresAtUtc',
  ];
  if (!exactKeys(receipt, fields, 'Bun promotion evidence', errors)) {
    return { errors, promotedAt: null };
  }
  if (receipt.schema !== PROMOTION_SCHEMA) {
    errors.push(`Bun promotion evidence.schema must be ${PROMOTION_SCHEMA}`);
  }
  const binding = {
    qualificationId: text(
      receipt.qualificationId,
      'Bun promotion evidence.qualificationId',
      errors
    ),
    workload: text(receipt.workload, 'Bun promotion evidence.workload', errors),
    logicalModelId: text(
      receipt.logicalModelId,
      'Bun promotion evidence.logicalModelId',
      errors
    ),
    resolvedArtifactVariantId: sha256(
      receipt.resolvedArtifactVariantId,
      'Bun promotion evidence.resolvedArtifactVariantId',
      errors
    ),
    resolvedExecutionId: sha256(
      receipt.resolvedExecutionId,
      'Bun promotion evidence.resolvedExecutionId',
      errors
    ),
    bunVersion: text(receipt.bunVersion, 'Bun promotion evidence.bunVersion', errors),
    webgpuImplementationId: text(
      receipt.webgpuImplementationId,
      'Bun promotion evidence.webgpuImplementationId',
      errors
    ),
    providerId: text(receipt.providerId, 'Bun promotion evidence.providerId', errors),
  };
  for (const [field, actual] of Object.entries(binding)) {
    match(actual, expected[field], `Bun promotion evidence.${field}`, errors);
  }
  const evidenceSetDigest = sha256(
    receipt.evidenceSetDigest,
    'Bun promotion evidence.evidenceSetDigest',
    errors
  );
  match(evidenceSetDigest, expected.evidenceSetDigest, 'Bun promotion evidence.evidenceSetDigest', errors);
  if (receipt.decision !== 'promote') errors.push('Bun promotion evidence.decision must be promote');
  if (receipt.authority !== 'human') errors.push('Bun promotion evidence.authority must be human');
  const promotedAt = instant(receipt.promotedAtUtc, 'Bun promotion evidence.promotedAtUtc', errors);
  const qualifiedAt = instant(
    receipt.qualifiedAtUtc,
    'Bun promotion evidence.qualifiedAtUtc',
    errors
  );
  const expiresAt = instant(receipt.expiresAtUtc, 'Bun promotion evidence.expiresAtUtc', errors);
  match(receipt.qualifiedAtUtc, expected.qualifiedAtUtc, 'Bun promotion evidence.qualifiedAtUtc', errors);
  match(receipt.expiresAtUtc, expected.expiresAtUtc, 'Bun promotion evidence.expiresAtUtc', errors);
  if (promotedAt && qualifiedAt && promotedAt.getTime() > qualifiedAt.getTime()) {
    errors.push('Bun promotion evidence.promotedAtUtc must not follow qualifiedAtUtc');
  }
  if (qualifiedAt && expiresAt && expiresAt.getTime() <= qualifiedAt.getTime()) {
    errors.push('Bun promotion evidence.expiresAtUtc must follow qualifiedAtUtc');
  }
  return { errors, promotedAt };
}
