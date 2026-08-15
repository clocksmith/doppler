const ACTIVATION_SCHEMA = 'doppler.runtime-promotion-activation-evidence/v1';
const DECISION_SCHEMA = 'doppler.runtime-promotion-decision-evidence/v1';
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const REVISION_PATTERN = /^[0-9a-f]{40}$/;
const SCOPE_FIELDS = Object.freeze([
  'modelId',
  'artifactVariantId',
  'executionId',
  'providerId',
  'environmentFingerprint',
  'workloadId',
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

function digest(value, label, errors) {
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

function scope(value, label, errors) {
  exactKeys(value, SCOPE_FIELDS, label, errors);
  const normalized = Object.fromEntries(SCOPE_FIELDS.map((field) => [
    field,
    text(value?.[field], `${label}.${field}`, errors),
  ]));
  digest(normalized.artifactVariantId, `${label}.artifactVariantId`, errors);
  digest(normalized.executionId, `${label}.executionId`, errors);
  return normalized;
}

function match(actual, expected, label, errors) {
  if (expected != null && actual != null && actual !== expected) {
    errors.push(`${label} ${actual} does not match expected ${expected}`);
  }
}

function matchScope(actual, expected, label, errors) {
  if (!expected) return;
  for (const field of SCOPE_FIELDS) match(actual[field], expected[field], `${label}.${field}`, errors);
}

export function validateRuntimePromotionActivationEvidence(receipt, expected = {}) {
  const errors = [];
  const fields = [
    'schema', 'promotionId', 'candidateId', 'candidateHash', 'activatedAtUtc', 'scope',
    'authority', 'reviewer', 'reviewerRevision', 'statement',
  ];
  if (!exactKeys(receipt, fields, 'activation evidence', errors)) {
    return { errors, activatedAt: null, scope: null };
  }
  if (receipt.schema !== ACTIVATION_SCHEMA) {
    errors.push(`activation evidence.schema must be ${ACTIVATION_SCHEMA}`);
  }
  const promotionId = text(receipt.promotionId, 'activation evidence.promotionId', errors);
  const candidateId = text(receipt.candidateId, 'activation evidence.candidateId', errors);
  const candidateHash = digest(receipt.candidateHash, 'activation evidence.candidateHash', errors);
  const activatedAt = instant(receipt.activatedAtUtc, 'activation evidence.activatedAtUtc', errors);
  const resolvedScope = scope(receipt.scope, 'activation evidence.scope', errors);
  if (receipt.authority !== 'human') errors.push('activation evidence.authority must be human');
  text(receipt.reviewer, 'activation evidence.reviewer', errors);
  revision(receipt.reviewerRevision, 'activation evidence.reviewerRevision', errors);
  text(receipt.statement, 'activation evidence.statement', errors);
  match(promotionId, expected.promotionId, 'activation promotionId', errors);
  match(candidateId, expected.candidateId, 'activation candidateId', errors);
  match(candidateHash, expected.candidateHash, 'activation candidateHash', errors);
  if (expected.activatedAtUtc && activatedAt) {
    match(activatedAt.toISOString(), expected.activatedAtUtc, 'activation timestamp', errors);
  }
  matchScope(resolvedScope, expected.scope, 'activation scope', errors);
  return { errors, activatedAt, scope: resolvedScope, promotionId, candidateId, candidateHash };
}

export function validateRuntimePromotionDecisionEvidence(receipt, expected = {}) {
  const errors = [];
  const fields = [
    'schema', 'promotionId', 'candidateId', 'candidateHash', 'scope', 'status',
    'decidedAtUtc', 'reason', 'revocationRecordId', 'authority', 'reviewer',
    'reviewerRevision', 'statement',
  ];
  if (!exactKeys(receipt, fields, 'decision evidence', errors)) {
    return { errors, decidedAt: null, scope: null };
  }
  if (receipt.schema !== DECISION_SCHEMA) {
    errors.push(`decision evidence.schema must be ${DECISION_SCHEMA}`);
  }
  const promotionId = text(receipt.promotionId, 'decision evidence.promotionId', errors);
  const candidateId = text(receipt.candidateId, 'decision evidence.candidateId', errors);
  const candidateHash = digest(receipt.candidateHash, 'decision evidence.candidateHash', errors);
  const resolvedScope = scope(receipt.scope, 'decision evidence.scope', errors);
  const status = text(receipt.status, 'decision evidence.status', errors);
  if (status && !['retain', 'revoke'].includes(status)) {
    errors.push('decision evidence.status must be retain or revoke');
  }
  const decidedAt = instant(receipt.decidedAtUtc, 'decision evidence.decidedAtUtc', errors);
  const reason = text(receipt.reason, 'decision evidence.reason', errors);
  const revocationRecordId = text(
    receipt.revocationRecordId,
    'decision evidence.revocationRecordId',
    errors,
    true
  );
  if (status === 'retain' && revocationRecordId !== null) {
    errors.push('retain decision evidence must not name a revocation record');
  }
  if (status === 'revoke' && !revocationRecordId) {
    errors.push('revoke decision evidence requires a revocation record');
  }
  if (receipt.authority !== 'human') errors.push('decision evidence.authority must be human');
  text(receipt.reviewer, 'decision evidence.reviewer', errors);
  revision(receipt.reviewerRevision, 'decision evidence.reviewerRevision', errors);
  text(receipt.statement, 'decision evidence.statement', errors);
  match(promotionId, expected.promotionId, 'decision promotionId', errors);
  match(candidateId, expected.candidateId, 'decision candidateId', errors);
  match(candidateHash, expected.candidateHash, 'decision candidateHash', errors);
  match(status, expected.status, 'decision status', errors);
  match(reason, expected.reason, 'decision reason', errors);
  match(revocationRecordId, expected.revocationRecordId, 'decision revocationRecordId', errors);
  if (expected.decidedAtUtc && decidedAt) {
    match(decidedAt.toISOString(), expected.decidedAtUtc, 'decision timestamp', errors);
  }
  matchScope(resolvedScope, expected.scope, 'decision scope', errors);
  return {
    errors,
    decidedAt,
    scope: resolvedScope,
    promotionId,
    candidateId,
    candidateHash,
    status,
    reason,
    revocationRecordId,
  };
}
