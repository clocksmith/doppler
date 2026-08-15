import { computeCanonicalJsonSha256 } from './canonical-json.js';

const EVIDENCE_SCHEMA = 'doppler.provider-conformance-evidence/v1';
const PROVIDER_PROMOTION_SCHEMA = 'doppler.provider-conformance-provider-promotion/v1';
const SUITE_PROMOTION_SCHEMA = 'doppler.provider-conformance-suite-promotion/v1';
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const REVISION_PATTERN = /^[0-9a-f]{40}$/;
const LIFECYCLE_RESULTS = new Set(['passed', 'failed', 'not-run']);

export const PROVIDER_CONFORMANCE_EVIDENCE_CLASSES = Object.freeze([
  'modelContract',
  'resolutionIdentity',
  'operations',
  'lifecycle',
  'correctness',
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

function text(value, label, errors) {
  const normalized = normalizeText(value);
  if (!normalized) errors.push(`${label} must be a non-empty string`);
  return normalized || null;
}

function sha256(value, label, errors) {
  const normalized = text(value, label, errors)?.toLowerCase() ?? null;
  if (normalized && !SHA256_PATTERN.test(normalized)) {
    errors.push(`${label} must be a SHA-256 identity`);
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

function stringArray(value, label, errors) {
  if (!Array.isArray(value) || value.length === 0) {
    errors.push(`${label} must be a non-empty array`);
    return [];
  }
  const values = value
    .map((entry, index) => text(entry, `${label}[${index}]`, errors))
    .filter(Boolean);
  if (new Set(values).size !== values.length) errors.push(`${label} contains duplicate entries`);
  return values;
}

function sameMembers(left, right) {
  const a = [...left].sort();
  const b = [...right].sort();
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function match(actual, expected, label, errors) {
  if (expected != null && actual != null && actual !== expected) {
    errors.push(`${label} does not match the provider qualification`);
  }
}

function validateBinding(receipt, expected, errors) {
  const label = 'provider conformance evidence';
  const values = {
    suiteId: text(receipt.suiteId, `${label}.suiteId`, errors),
    laneId: text(receipt.laneId, `${label}.laneId`, errors),
    workload: text(receipt.workload, `${label}.workload`, errors),
    logicalModelId: text(receipt.logicalModelId, `${label}.logicalModelId`, errors),
    manifestVariantId: text(receipt.manifestVariantId, `${label}.manifestVariantId`, errors),
    resolvedArtifactVariantId: sha256(
      receipt.resolvedArtifactVariantId,
      `${label}.resolvedArtifactVariantId`,
      errors
    ),
    resolvedExecutionId: sha256(
      receipt.resolvedExecutionId,
      `${label}.resolvedExecutionId`,
      errors
    ),
    implementationId: text(receipt.implementationId, `${label}.implementationId`, errors),
    harnessRevision: revision(receipt.harnessRevision, `${label}.harnessRevision`, errors),
    environmentFingerprint: sha256(
      receipt.environmentFingerprint,
      `${label}.environmentFingerprint`,
      errors
    ),
    providerReceiptDigest: sha256(
      receipt.providerReceiptDigest,
      `${label}.providerReceiptDigest`,
      errors
    ),
  };
  for (const [field, value] of Object.entries(values)) {
    match(value, expected[field], `${label}.${field}`, errors);
  }
  return values;
}

function modelContractPassed(observations, expected, errors) {
  const fields = [
    'manifestDigest',
    'tokenizerDigest',
    'executionGraphDigest',
    'runtimePolicyDigest',
    'artifactValidated',
    'tokenizerIdentityMatched',
    'executionGraphIdentityMatched',
    'runtimePolicyExplicit',
  ];
  exactKeys(observations, fields, 'observations', errors);
  const manifestDigest = sha256(observations?.manifestDigest, 'observations.manifestDigest', errors);
  sha256(observations?.tokenizerDigest, 'observations.tokenizerDigest', errors);
  sha256(observations?.executionGraphDigest, 'observations.executionGraphDigest', errors);
  sha256(observations?.runtimePolicyDigest, 'observations.runtimePolicyDigest', errors);
  match(
    manifestDigest,
    expected.resolvedArtifactVariantId,
    'observations.manifestDigest',
    errors
  );
  return fields.slice(4).map((field) => (
    boolean(observations?.[field], `observations.${field}`, errors)
  )).every(Boolean);
}

function resolutionIdentityPassed(observations, expected, errors) {
  const fields = [
    'logicalModelResolved',
    'manifestVariantMatched',
    'artifactDigestMatched',
    'executionDigestMatched',
    'fallbackUsed',
  ];
  exactKeys(observations, fields, 'observations', errors);
  const matched = fields.slice(0, 4).map((field) => (
    boolean(observations?.[field], `observations.${field}`, errors)
  ));
  if (typeof observations?.fallbackUsed !== 'boolean') {
    errors.push('observations.fallbackUsed must be boolean');
  }
  return matched.every(Boolean) && observations?.fallbackUsed === false;
}

function operationsPassed(observations, expected, errors) {
  const fields = ['declaredOperations', 'observedOperations', 'unsupportedOperationUsed'];
  exactKeys(observations, fields, 'observations', errors);
  const declared = stringArray(
    observations?.declaredOperations,
    'observations.declaredOperations',
    errors
  );
  const observed = stringArray(
    observations?.observedOperations,
    'observations.observedOperations',
    errors
  );
  if (!sameMembers(declared, expected.declaredOperations || [])) {
    errors.push('observations.declaredOperations does not match the suite');
  }
  if (!sameMembers(observed, expected.declaredOperations || [])) {
    errors.push('observations.observedOperations does not match the suite');
  }
  if (typeof observations?.unsupportedOperationUsed !== 'boolean') {
    errors.push('observations.unsupportedOperationUsed must be boolean');
  }
  return sameMembers(declared, expected.declaredOperations || [])
    && sameMembers(observed, expected.declaredOperations || [])
    && observations?.unsupportedOperationUsed === false;
}

function lifecyclePassed(observations, expected, errors) {
  const fields = ['load', 'execute', 'unload', 'repeatedSessions', 'minimumRepeatedSessions'];
  exactKeys(observations, fields, 'observations', errors);
  for (const stage of ['load', 'execute', 'unload']) {
    if (!LIFECYCLE_RESULTS.has(observations?.[stage])) {
      errors.push(`observations.${stage} is not recognized`);
    }
  }
  const sessions = integer(observations?.repeatedSessions, 'observations.repeatedSessions', errors, 1);
  const minimum = integer(
    observations?.minimumRepeatedSessions,
    'observations.minimumRepeatedSessions',
    errors,
    1
  );
  return observations?.load === 'passed'
    && observations?.execute === 'passed'
    && observations?.unload === 'passed'
    && sessions !== null
    && minimum !== null
    && sessions >= minimum;
}

function thresholdPassed(score, threshold, higherIsBetter) {
  if (score === null || threshold === null) return false;
  return higherIsBetter ? score >= threshold : score <= threshold;
}

function correctnessPassed(observations, expected, errors) {
  const correctnessClass = text(
    observations?.correctnessClass,
    'observations.correctnessClass',
    errors
  );
  match(correctnessClass, expected.correctnessClass, 'observations.correctnessClass', errors);
  if (correctnessClass === 'exact-token') {
    const fields = [
      'correctnessClass',
      'referenceOutputDigest',
      'providerOutputDigest',
      'tokenParityPassed',
      'deterministicContinuationPassed',
    ];
    exactKeys(observations, fields, 'observations', errors);
    const reference = sha256(
      observations?.referenceOutputDigest,
      'observations.referenceOutputDigest',
      errors
    );
    const provider = sha256(
      observations?.providerOutputDigest,
      'observations.providerOutputDigest',
      errors
    );
    return reference !== null
      && reference === provider
      && boolean(observations?.tokenParityPassed, 'observations.tokenParityPassed', errors)
      && boolean(
        observations?.deterministicContinuationPassed,
        'observations.deterministicContinuationPassed',
        errors
      );
  }
  if (correctnessClass === 'tolerance-bounded-numerical') {
    const fields = [
      'correctnessClass',
      'referenceOutputDigest',
      'providerOutputDigest',
      'shapeMatched',
      'finiteOutputs',
      'maxAbsoluteError',
      'maximumAbsoluteError',
    ];
    exactKeys(observations, fields, 'observations', errors);
    sha256(observations?.referenceOutputDigest, 'observations.referenceOutputDigest', errors);
    sha256(observations?.providerOutputDigest, 'observations.providerOutputDigest', errors);
    const observed = number(observations?.maxAbsoluteError, 'observations.maxAbsoluteError', errors);
    const maximum = number(
      observations?.maximumAbsoluteError,
      'observations.maximumAbsoluteError',
      errors
    );
    return boolean(observations?.shapeMatched, 'observations.shapeMatched', errors)
      && boolean(observations?.finiteOutputs, 'observations.finiteOutputs', errors)
      && observed !== null
      && maximum !== null
      && observed <= maximum;
  }
  if (correctnessClass === 'semantic' || correctnessClass === 'held-out-task-metric') {
    const fields = [
      'correctnessClass',
      'heldOutSetDigest',
      'referenceScore',
      'providerScore',
      'minimumAcceptedScore',
      'maximumReferenceDelta',
      'higherIsBetter',
      'orderingAgreementPassed',
    ];
    exactKeys(observations, fields, 'observations', errors);
    sha256(observations?.heldOutSetDigest, 'observations.heldOutSetDigest', errors);
    const reference = number(observations?.referenceScore, 'observations.referenceScore', errors);
    const provider = number(observations?.providerScore, 'observations.providerScore', errors);
    const threshold = number(
      observations?.minimumAcceptedScore,
      'observations.minimumAcceptedScore',
      errors
    );
    const maximumDelta = number(
      observations?.maximumReferenceDelta,
      'observations.maximumReferenceDelta',
      errors
    );
    const higherIsBetter = boolean(
      observations?.higherIsBetter,
      'observations.higherIsBetter',
      errors
    );
    return thresholdPassed(provider, threshold, higherIsBetter)
      && reference !== null
      && provider !== null
      && maximumDelta !== null
      && Math.abs(reference - provider) <= maximumDelta
      && boolean(
        observations?.orderingAgreementPassed,
        'observations.orderingAgreementPassed',
        errors
      );
  }
  errors.push('observations.correctnessClass is not recognized');
  return false;
}

const EVIDENCE_VALIDATORS = Object.freeze({
  modelContract: modelContractPassed,
  resolutionIdentity: resolutionIdentityPassed,
  operations: operationsPassed,
  lifecycle: lifecyclePassed,
  correctness: correctnessPassed,
});

function summaryFor(evidenceClass, observations, passed) {
  if (evidenceClass === 'operations') return { operations: observations?.observedOperations || [] };
  if (evidenceClass === 'lifecycle') {
    return {
      lifecycle: {
        load: observations?.load,
        execute: observations?.execute,
        unload: observations?.unload,
      },
    };
  }
  if (evidenceClass === 'correctness') {
    return { correctness: { class: observations?.correctnessClass, passed } };
  }
  return {};
}

export function validateProviderConformanceEvidence(receipt, expected = {}) {
  const errors = [];
  const reasons = [];
  const fields = [
    'schema',
    'evidenceClass',
    'suiteId',
    'laneId',
    'workload',
    'logicalModelId',
    'manifestVariantId',
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'implementationId',
    'harnessRevision',
    'environmentFingerprint',
    'providerReceiptDigest',
    'capturedAtUtc',
    'result',
    'observations',
  ];
  if (!exactKeys(receipt, fields, 'provider conformance evidence', errors)) {
    return {
      errors,
      reasons: ['provider-conformance-evidence-invalid'],
      passed: null,
      capturedAt: null,
      summary: {},
    };
  }
  if (receipt.schema !== EVIDENCE_SCHEMA) {
    errors.push(`provider conformance evidence.schema must be ${EVIDENCE_SCHEMA}`);
  }
  const evidenceClass = text(
    receipt.evidenceClass,
    'provider conformance evidence.evidenceClass',
    errors
  );
  if (evidenceClass && !PROVIDER_CONFORMANCE_EVIDENCE_CLASSES.includes(evidenceClass)) {
    errors.push('provider conformance evidence.evidenceClass is not recognized');
  }
  match(
    evidenceClass,
    expected.evidenceClass,
    'provider conformance evidence.evidenceClass',
    errors
  );
  validateBinding(receipt, expected, errors);
  const capturedAt = instant(
    receipt.capturedAtUtc,
    'provider conformance evidence.capturedAtUtc',
    errors
  );
  let passed = null;
  if (exactKeys(receipt.result, ['passed'], 'provider conformance evidence.result', errors)) {
    if (typeof receipt.result.passed !== 'boolean') {
      errors.push('provider conformance evidence.result.passed must be boolean');
    }
    const validator = EVIDENCE_VALIDATORS[evidenceClass];
    const derivedPassed = validator ? validator(receipt.observations, expected, errors) : false;
    if (typeof receipt.result.passed === 'boolean' && receipt.result.passed !== derivedPassed) {
      errors.push('provider conformance evidence.result.passed does not match observations');
    }
    passed = errors.length === 0 ? derivedPassed : null;
  }
  if (passed !== true) reasons.push(`${evidenceClass || 'provider'}-gate-failed`);
  return {
    errors,
    reasons,
    passed,
    capturedAt,
    summary: summaryFor(evidenceClass, receipt.observations, passed),
  };
}

export function computeProviderConformanceEvidenceSetDigest(evidenceReferences) {
  return computeCanonicalJsonSha256(evidenceReferences);
}

export function computeProviderConformanceProviderSetDigest(providerBindings) {
  return computeCanonicalJsonSha256(
    [...providerBindings].sort((left, right) => left.laneId.localeCompare(right.laneId))
  );
}

export function validateProviderConformanceProviderPromotion(receipt, expected = {}) {
  const errors = [];
  const fields = [
    'schema',
    'suiteId',
    'laneId',
    'logicalModelId',
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'implementationId',
    'evidenceSetDigest',
    'decision',
    'authority',
    'reviewer',
    'rationale',
    'promotedAtUtc',
    'qualifiedAtUtc',
    'expiresAtUtc',
  ];
  if (!exactKeys(receipt, fields, 'provider promotion evidence', errors)) {
    return { errors, promotedAt: null };
  }
  if (receipt.schema !== PROVIDER_PROMOTION_SCHEMA) {
    errors.push(`provider promotion evidence.schema must be ${PROVIDER_PROMOTION_SCHEMA}`);
  }
  const values = {
    suiteId: text(receipt.suiteId, 'provider promotion evidence.suiteId', errors),
    laneId: text(receipt.laneId, 'provider promotion evidence.laneId', errors),
    logicalModelId: text(
      receipt.logicalModelId,
      'provider promotion evidence.logicalModelId',
      errors
    ),
    resolvedArtifactVariantId: sha256(
      receipt.resolvedArtifactVariantId,
      'provider promotion evidence.resolvedArtifactVariantId',
      errors
    ),
    resolvedExecutionId: sha256(
      receipt.resolvedExecutionId,
      'provider promotion evidence.resolvedExecutionId',
      errors
    ),
    implementationId: text(
      receipt.implementationId,
      'provider promotion evidence.implementationId',
      errors
    ),
    evidenceSetDigest: sha256(
      receipt.evidenceSetDigest,
      'provider promotion evidence.evidenceSetDigest',
      errors
    ),
    qualifiedAtUtc: text(
      receipt.qualifiedAtUtc,
      'provider promotion evidence.qualifiedAtUtc',
      errors
    ),
    expiresAtUtc: text(
      receipt.expiresAtUtc,
      'provider promotion evidence.expiresAtUtc',
      errors
    ),
  };
  for (const [field, value] of Object.entries(values)) {
    match(value, expected[field], `provider promotion evidence.${field}`, errors);
  }
  if (receipt.decision !== 'promote') errors.push('provider promotion evidence.decision must be promote');
  if (receipt.authority !== 'human') errors.push('provider promotion evidence.authority must be human');
  text(receipt.reviewer, 'provider promotion evidence.reviewer', errors);
  text(receipt.rationale, 'provider promotion evidence.rationale', errors);
  const promotedAt = instant(
    receipt.promotedAtUtc,
    'provider promotion evidence.promotedAtUtc',
    errors
  );
  const qualifiedAt = instant(
    receipt.qualifiedAtUtc,
    'provider promotion evidence.qualifiedAtUtc',
    errors
  );
  const expiresAt = instant(
    receipt.expiresAtUtc,
    'provider promotion evidence.expiresAtUtc',
    errors
  );
  if (promotedAt && qualifiedAt && promotedAt.getTime() < qualifiedAt.getTime()) {
    errors.push('provider promotion evidence.promotedAtUtc must not predate qualifiedAtUtc');
  }
  if (promotedAt && expiresAt && promotedAt.getTime() >= expiresAt.getTime()) {
    errors.push('provider promotion evidence.promotedAtUtc must predate expiresAtUtc');
  }
  return { errors, promotedAt };
}

export function validateProviderConformanceSuitePromotion(receipt, expected = {}) {
  const errors = [];
  const fields = [
    'schema',
    'suiteId',
    'workload',
    'logicalModelId',
    'resolvedArtifactVariantId',
    'requiredProviderLaneIds',
    'providerSetDigest',
    'decision',
    'authority',
    'reviewer',
    'rationale',
    'promotedAtUtc',
    'expiresAtUtc',
  ];
  if (!exactKeys(receipt, fields, 'suite promotion evidence', errors)) {
    return { errors, promotedAt: null };
  }
  if (receipt.schema !== SUITE_PROMOTION_SCHEMA) {
    errors.push(`suite promotion evidence.schema must be ${SUITE_PROMOTION_SCHEMA}`);
  }
  const values = {
    suiteId: text(receipt.suiteId, 'suite promotion evidence.suiteId', errors),
    workload: text(receipt.workload, 'suite promotion evidence.workload', errors),
    logicalModelId: text(receipt.logicalModelId, 'suite promotion evidence.logicalModelId', errors),
    resolvedArtifactVariantId: sha256(
      receipt.resolvedArtifactVariantId,
      'suite promotion evidence.resolvedArtifactVariantId',
      errors
    ),
    providerSetDigest: sha256(
      receipt.providerSetDigest,
      'suite promotion evidence.providerSetDigest',
      errors
    ),
    expiresAtUtc: text(receipt.expiresAtUtc, 'suite promotion evidence.expiresAtUtc', errors),
  };
  for (const [field, value] of Object.entries(values)) {
    match(value, expected[field], `suite promotion evidence.${field}`, errors);
  }
  const laneIds = stringArray(
    receipt.requiredProviderLaneIds,
    'suite promotion evidence.requiredProviderLaneIds',
    errors
  );
  if (!sameMembers(laneIds, expected.requiredProviderLaneIds || [])) {
    errors.push('suite promotion evidence.requiredProviderLaneIds does not match the suite');
  }
  if (receipt.decision !== 'promote') errors.push('suite promotion evidence.decision must be promote');
  if (receipt.authority !== 'human') errors.push('suite promotion evidence.authority must be human');
  text(receipt.reviewer, 'suite promotion evidence.reviewer', errors);
  text(receipt.rationale, 'suite promotion evidence.rationale', errors);
  const promotedAt = instant(
    receipt.promotedAtUtc,
    'suite promotion evidence.promotedAtUtc',
    errors
  );
  const expiresAt = instant(
    receipt.expiresAtUtc,
    'suite promotion evidence.expiresAtUtc',
    errors
  );
  const latestProviderPromotion = expected.latestProviderPromotionAtUtc
    ? instant(
      expected.latestProviderPromotionAtUtc,
      'suite promotion expected latestProviderPromotionAtUtc',
      errors
    )
    : null;
  if (promotedAt && latestProviderPromotion && promotedAt.getTime() < latestProviderPromotion.getTime()) {
    errors.push('suite promotion evidence.promotedAtUtc must not predate provider promotion');
  }
  if (promotedAt && expiresAt && promotedAt.getTime() >= expiresAt.getTime()) {
    errors.push('suite promotion evidence.promotedAtUtc must predate expiresAtUtc');
  }
  return { errors, promotedAt };
}
