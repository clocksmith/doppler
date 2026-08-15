const OWNER_SCHEMA = 'doppler.product-integration-owner-confirmation/v1';
const OUTCOME_SCHEMA = 'doppler.product-integration-evidence/v1';
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const REVISION_PATTERN = /^[0-9a-f]{40}$/;

export const PRODUCT_OUTCOME_EVIDENCE_CLASSES = Object.freeze([
  'installToFirstVerifiedOutput',
  'sourceTaskQualityRetention',
  'reliability',
  'memory',
  'coldWarmResponse',
  'browserHardwareQualification',
  'incumbentControl',
  'upgradeRequalification',
  'rollbackRevocation',
]);

const OBSERVATION_FIELDS = Object.freeze({
  installToFirstVerifiedOutput: [
    'surface',
    'installSucceeded',
    'firstVerifiedOutputMs',
    'maximumFirstVerifiedOutputMs',
  ],
  sourceTaskQualityRetention: [
    'sourceScore',
    'dopplerScore',
    'retentionRatio',
    'minimumRetentionRatio',
  ],
  reliability: [
    'attempts',
    'successes',
    'minimumSuccessRate',
    'crashes',
    'maximumCrashes',
    'ooms',
    'maximumOoms',
    'deviceLosses',
    'maximumDeviceLosses',
  ],
  memory: ['peakBytes', 'budgetBytes'],
  coldWarmResponse: [
    'sampleCount',
    'coldP50Ms',
    'coldP95Ms',
    'coldP95LimitMs',
    'warmP50Ms',
    'warmP95Ms',
    'warmP95LimitMs',
  ],
  browserHardwareQualification: [
    'qualifiedTargets',
    'failedTargets',
    'minimumQualifiedTargets',
  ],
  incumbentControl: [
    'incumbentProviderId',
    'incumbentArtifactId',
    'comparisonReceiptDigest',
    'incumbentAvailable',
    'correctnessComparable',
  ],
  upgradeRequalification: [
    'fromVersion',
    'toVersion',
    'migrationSucceeded',
    'identityPreserved',
    'taskGatePassed',
  ],
  rollbackRevocation: [
    'knownSafeVersion',
    'rollbackSucceeded',
    'revocationObserved',
    'taskGatePassed',
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

function requiredText(value, label, errors) {
  const normalized = normalizeText(value);
  if (!normalized) errors.push(`${label} must be a non-empty string`);
  return normalized || null;
}

function requiredSha256(value, label, errors) {
  const normalized = requiredText(value, label, errors)?.toLowerCase() ?? null;
  if (normalized && !SHA256_PATTERN.test(normalized)) {
    errors.push(`${label} must be a SHA-256 identity`);
  }
  return normalized;
}

function requiredRevision(value, label, errors) {
  const normalized = requiredText(value, label, errors)?.toLowerCase() ?? null;
  if (normalized && !REVISION_PATTERN.test(normalized)) {
    errors.push(`${label} must be a 40-character commit revision`);
  }
  return normalized;
}

function isoInstant(value, label, errors) {
  const normalized = requiredText(value, label, errors);
  const instant = normalized ? new Date(normalized) : null;
  if (!instant || !Number.isFinite(instant.getTime()) || instant.toISOString() !== normalized) {
    errors.push(`${label} must be an ISO instant`);
    return null;
  }
  return instant;
}

function matchExpected(actual, expected, label, errors) {
  if (expected != null && actual != null && actual !== expected) {
    errors.push(`${label} ${actual} does not match expected ${expected}`);
  }
}

function booleanValue(value, label, errors) {
  if (typeof value !== 'boolean') {
    errors.push(`${label} must be boolean`);
    return false;
  }
  return value;
}

function finiteNumber(value, label, errors, minimum = null) {
  if (!Number.isFinite(value) || (minimum != null && value < minimum)) {
    errors.push(`${label} must be a finite number${minimum != null ? ` >= ${minimum}` : ''}`);
    return null;
  }
  return value;
}

function integerValue(value, label, errors, minimum = 0) {
  if (!Number.isInteger(value) || value < minimum) {
    errors.push(`${label} must be an integer >= ${minimum}`);
    return null;
  }
  return value;
}

function stringList(value, label, errors) {
  if (!Array.isArray(value)) {
    errors.push(`${label} must be an array`);
    return [];
  }
  const values = value.map((entry, index) => requiredText(entry, `${label}[${index}]`, errors));
  const normalized = values.filter(Boolean);
  if (new Set(normalized).size !== normalized.length) errors.push(`${label} contains duplicates`);
  return normalized;
}

function ratioMatches(actual, expected) {
  return Math.abs(actual - expected) <= Math.max(1e-9, Math.abs(expected) * 1e-9);
}

function validateInstall(value, errors) {
  requiredText(value.surface, 'observations.surface', errors);
  const installed = booleanValue(value.installSucceeded, 'observations.installSucceeded', errors);
  const actual = finiteNumber(value.firstVerifiedOutputMs, 'observations.firstVerifiedOutputMs', errors, 0);
  const limit = finiteNumber(
    value.maximumFirstVerifiedOutputMs,
    'observations.maximumFirstVerifiedOutputMs',
    errors,
    0
  );
  return installed && actual != null && limit != null && actual <= limit;
}

function validateQuality(value, errors) {
  const source = finiteNumber(value.sourceScore, 'observations.sourceScore', errors, 0);
  const doppler = finiteNumber(value.dopplerScore, 'observations.dopplerScore', errors, 0);
  const ratio = finiteNumber(value.retentionRatio, 'observations.retentionRatio', errors, 0);
  const minimum = finiteNumber(
    value.minimumRetentionRatio,
    'observations.minimumRetentionRatio',
    errors,
    0
  );
  if (source === 0) errors.push('observations.sourceScore must be greater than zero');
  if (source && doppler != null && ratio != null && !ratioMatches(ratio, doppler / source)) {
    errors.push('observations.retentionRatio does not match dopplerScore/sourceScore');
  }
  return source > 0 && ratio != null && minimum != null && ratio >= minimum;
}

function validateReliability(value, errors) {
  const attempts = integerValue(value.attempts, 'observations.attempts', errors, 1);
  const successes = integerValue(value.successes, 'observations.successes', errors);
  const minimumRate = finiteNumber(
    value.minimumSuccessRate,
    'observations.minimumSuccessRate',
    errors,
    0
  );
  const crashes = integerValue(value.crashes, 'observations.crashes', errors);
  const maxCrashes = integerValue(value.maximumCrashes, 'observations.maximumCrashes', errors);
  const ooms = integerValue(value.ooms, 'observations.ooms', errors);
  const maxOoms = integerValue(value.maximumOoms, 'observations.maximumOoms', errors);
  const losses = integerValue(value.deviceLosses, 'observations.deviceLosses', errors);
  const maxLosses = integerValue(
    value.maximumDeviceLosses,
    'observations.maximumDeviceLosses',
    errors
  );
  if (minimumRate != null && minimumRate > 1) {
    errors.push('observations.minimumSuccessRate must be <= 1');
  }
  if (attempts != null && successes != null && successes > attempts) {
    errors.push('observations.successes must not exceed attempts');
  }
  return attempts > 0
    && successes != null
    && minimumRate != null
    && successes / attempts >= minimumRate
    && crashes <= maxCrashes
    && ooms <= maxOoms
    && losses <= maxLosses;
}

function validateMemory(value, errors) {
  const peak = integerValue(value.peakBytes, 'observations.peakBytes', errors);
  const budget = integerValue(value.budgetBytes, 'observations.budgetBytes', errors, 1);
  return peak != null && budget != null && peak <= budget;
}

function validateColdWarm(value, errors) {
  const samples = integerValue(value.sampleCount, 'observations.sampleCount', errors, 1);
  const coldP50 = finiteNumber(value.coldP50Ms, 'observations.coldP50Ms', errors, 0);
  const coldP95 = finiteNumber(value.coldP95Ms, 'observations.coldP95Ms', errors, 0);
  const coldLimit = finiteNumber(value.coldP95LimitMs, 'observations.coldP95LimitMs', errors, 0);
  const warmP50 = finiteNumber(value.warmP50Ms, 'observations.warmP50Ms', errors, 0);
  const warmP95 = finiteNumber(value.warmP95Ms, 'observations.warmP95Ms', errors, 0);
  const warmLimit = finiteNumber(value.warmP95LimitMs, 'observations.warmP95LimitMs', errors, 0);
  if (coldP50 != null && coldP95 != null && coldP50 > coldP95) {
    errors.push('observations.coldP50Ms must not exceed coldP95Ms');
  }
  if (warmP50 != null && warmP95 != null && warmP50 > warmP95) {
    errors.push('observations.warmP50Ms must not exceed warmP95Ms');
  }
  return samples > 0
    && coldP95 != null
    && coldLimit != null
    && warmP95 != null
    && warmLimit != null
    && coldP95 <= coldLimit
    && warmP95 <= warmLimit;
}

function validateBrowserHardware(value, errors) {
  const qualified = stringList(value.qualifiedTargets, 'observations.qualifiedTargets', errors);
  const failed = stringList(value.failedTargets, 'observations.failedTargets', errors);
  const minimum = integerValue(
    value.minimumQualifiedTargets,
    'observations.minimumQualifiedTargets',
    errors,
    1
  );
  return minimum != null && qualified.length >= minimum && failed.length === 0;
}

function validateIncumbent(value, errors) {
  requiredText(value.incumbentProviderId, 'observations.incumbentProviderId', errors);
  requiredText(value.incumbentArtifactId, 'observations.incumbentArtifactId', errors);
  requiredSha256(
    value.comparisonReceiptDigest,
    'observations.comparisonReceiptDigest',
    errors
  );
  const available = booleanValue(
    value.incumbentAvailable,
    'observations.incumbentAvailable',
    errors
  );
  const comparable = booleanValue(
    value.correctnessComparable,
    'observations.correctnessComparable',
    errors
  );
  return available && comparable;
}

function validateUpgrade(value, errors) {
  requiredText(value.fromVersion, 'observations.fromVersion', errors);
  requiredText(value.toVersion, 'observations.toVersion', errors);
  const migration = booleanValue(
    value.migrationSucceeded,
    'observations.migrationSucceeded',
    errors
  );
  const identity = booleanValue(
    value.identityPreserved,
    'observations.identityPreserved',
    errors
  );
  const task = booleanValue(value.taskGatePassed, 'observations.taskGatePassed', errors);
  return migration && identity && task;
}

function validateRollback(value, errors) {
  requiredText(value.knownSafeVersion, 'observations.knownSafeVersion', errors);
  const rollback = booleanValue(
    value.rollbackSucceeded,
    'observations.rollbackSucceeded',
    errors
  );
  const revocation = booleanValue(
    value.revocationObserved,
    'observations.revocationObserved',
    errors
  );
  const task = booleanValue(value.taskGatePassed, 'observations.taskGatePassed', errors);
  return rollback && revocation && task;
}

const OBSERVATION_VALIDATORS = Object.freeze({
  installToFirstVerifiedOutput: validateInstall,
  sourceTaskQualityRetention: validateQuality,
  reliability: validateReliability,
  memory: validateMemory,
  coldWarmResponse: validateColdWarm,
  browserHardwareQualification: validateBrowserHardware,
  incumbentControl: validateIncumbent,
  upgradeRequalification: validateUpgrade,
  rollbackRevocation: validateRollback,
});

export function validateProductIntegrationOwnerConfirmation(receipt, expected = {}) {
  const errors = [];
  const reasons = [];
  const fields = [
    'schema',
    'integrationId',
    'applicationName',
    'workload',
    'owner',
    'ownerRepository',
    'applicationRevision',
    'confirmedAtUtc',
    'maintenanceStatus',
    'statement',
  ];
  if (!exactKeys(receipt, fields, 'owner confirmation', errors)) {
    return { errors, reasons: ['owner-confirmation-invalid'], confirmedAt: null };
  }
  if (receipt.schema !== OWNER_SCHEMA) {
    errors.push(`owner confirmation.schema must be ${OWNER_SCHEMA}`);
  }
  const integrationId = requiredText(receipt.integrationId, 'owner confirmation.integrationId', errors);
  const applicationName = requiredText(
    receipt.applicationName,
    'owner confirmation.applicationName',
    errors
  );
  const workload = requiredText(receipt.workload, 'owner confirmation.workload', errors);
  const owner = requiredText(receipt.owner, 'owner confirmation.owner', errors);
  requiredText(receipt.ownerRepository, 'owner confirmation.ownerRepository', errors);
  requiredRevision(
    receipt.applicationRevision,
    'owner confirmation.applicationRevision',
    errors
  );
  const confirmedAt = isoInstant(
    receipt.confirmedAtUtc,
    'owner confirmation.confirmedAtUtc',
    errors
  );
  requiredText(receipt.statement, 'owner confirmation.statement', errors);
  matchExpected(integrationId, expected.integrationId, 'owner confirmation integrationId', errors);
  matchExpected(applicationName, expected.applicationName, 'owner confirmation applicationName', errors);
  matchExpected(workload, expected.workload, 'owner confirmation workload', errors);
  matchExpected(owner, expected.owner, 'owner confirmation owner', errors);
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

export function validateProductIntegrationOutcomeEvidence(receipt, expected = {}) {
  const errors = [];
  const reasons = [];
  const fields = [
    'schema',
    'evidenceClass',
    'integrationId',
    'applicationName',
    'workload',
    'owner',
    'applicationRevision',
    'harnessRevision',
    'environmentFingerprint',
    'logicalModelId',
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'capturedAtUtc',
    'result',
    'observations',
  ];
  if (!exactKeys(receipt, fields, 'product integration evidence', errors)) {
    return { errors, reasons: ['product-integration-evidence-invalid'], capturedAt: null };
  }
  if (receipt.schema !== OUTCOME_SCHEMA) {
    errors.push(`product integration evidence.schema must be ${OUTCOME_SCHEMA}`);
  }
  const evidenceClass = requiredText(
    receipt.evidenceClass,
    'product integration evidence.evidenceClass',
    errors
  );
  if (evidenceClass && !PRODUCT_OUTCOME_EVIDENCE_CLASSES.includes(evidenceClass)) {
    errors.push('product integration evidence.evidenceClass is not recognized');
  }
  const identityFields = [
    'integrationId',
    'applicationName',
    'workload',
    'owner',
    'logicalModelId',
  ];
  const identity = Object.fromEntries(identityFields.map((field) => [
    field,
    requiredText(receipt[field], `product integration evidence.${field}`, errors),
  ]));
  requiredRevision(
    receipt.applicationRevision,
    'product integration evidence.applicationRevision',
    errors
  );
  requiredRevision(receipt.harnessRevision, 'product integration evidence.harnessRevision', errors);
  requiredSha256(
    receipt.environmentFingerprint,
    'product integration evidence.environmentFingerprint',
    errors
  );
  const artifactId = requiredSha256(
    receipt.resolvedArtifactVariantId,
    'product integration evidence.resolvedArtifactVariantId',
    errors
  );
  const executionId = requiredSha256(
    receipt.resolvedExecutionId,
    'product integration evidence.resolvedExecutionId',
    errors
  );
  const capturedAt = isoInstant(
    receipt.capturedAtUtc,
    'product integration evidence.capturedAtUtc',
    errors
  );
  matchExpected(evidenceClass, expected.evidenceClass, 'product evidence class', errors);
  for (const field of identityFields) {
    matchExpected(identity[field], expected[field], `product evidence ${field}`, errors);
  }
  matchExpected(
    artifactId,
    expected.resolvedArtifactVariantId,
    'product evidence artifact identity',
    errors
  );
  matchExpected(
    executionId,
    expected.resolvedExecutionId,
    'product evidence execution identity',
    errors
  );
  let claimedPassed = null;
  if (exactKeys(receipt.result, ['passed'], 'product integration evidence.result', errors)) {
    if (typeof receipt.result.passed !== 'boolean') {
      errors.push('product integration evidence.result.passed must be boolean');
    } else {
      claimedPassed = receipt.result.passed;
    }
  }
  let derivedPassed = false;
  const observationFields = OBSERVATION_FIELDS[evidenceClass];
  if (observationFields && exactKeys(
    receipt.observations,
    observationFields,
    `${evidenceClass} observations`,
    errors
  )) {
    derivedPassed = OBSERVATION_VALIDATORS[evidenceClass](receipt.observations, errors);
  }
  if (claimedPassed !== null && claimedPassed !== derivedPassed) {
    errors.push(`${evidenceClass} result.passed does not match its observations and thresholds`);
  }
  if (claimedPassed !== true) reasons.push(`${evidenceClass}-not-passed`);
  return { errors, reasons, capturedAt };
}
