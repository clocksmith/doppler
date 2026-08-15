import {
  canonicalizeJson,
  computeCanonicalJsonSha256,
} from './canonical-json.js';

const EXTERNAL_SCHEMA = 'doppler.runtime-ownership-execution-evidence/v1';
const DOPPLER_RECEIPT_SCHEMA = 'doppler_provider_receipt_v1';
const RESOLUTION_SCHEMA = 'doppler.resolution-identity/v1';
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const ROLES = new Set(['source', 'incumbent']);
const WORKLOADS = new Set(['generation', 'embedding', 'reranking']);
const STATUSES = new Set(['passed', 'failed']);

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

function nullableSha256(value, label, errors) {
  if (value === null) return null;
  return requiredSha256(value, label, errors);
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

export function canonicalizeRuntimeOwnershipEvidence(value) {
  return canonicalizeJson(value);
}

export function computeRuntimeOwnershipEvidenceId(receipt) {
  return computeCanonicalJsonSha256(receipt);
}

export function validateRuntimeOwnershipExecutionEvidence(receipt, expected = {}) {
  const errors = [];
  const reasons = [];
  const fields = [
    'schema',
    'role',
    'providerId',
    'artifactId',
    'artifactRevision',
    'workload',
    'logicalModelId',
    'runtime',
    'invocation',
    'result',
  ];
  if (!exactKeys(receipt, fields, 'execution evidence', errors)) {
    return { errors, reasons: ['execution-evidence-invalid'], evidenceId: null, status: null };
  }
  if (receipt.schema !== EXTERNAL_SCHEMA) {
    errors.push(`execution evidence.schema must be ${EXTERNAL_SCHEMA}`);
  }
  const role = requiredText(receipt.role, 'execution evidence.role', errors);
  if (role && !ROLES.has(role)) errors.push('execution evidence.role is not recognized');
  const providerId = requiredText(receipt.providerId, 'execution evidence.providerId', errors);
  const artifactId = requiredText(receipt.artifactId, 'execution evidence.artifactId', errors);
  requiredText(receipt.artifactRevision, 'execution evidence.artifactRevision', errors);
  const workload = requiredText(receipt.workload, 'execution evidence.workload', errors);
  if (workload && !WORKLOADS.has(workload)) {
    errors.push('execution evidence.workload is not recognized');
  }
  const logicalModelId = requiredText(
    receipt.logicalModelId,
    'execution evidence.logicalModelId',
    errors
  );
  matchExpected(role, expected.role, 'execution evidence role', errors);
  matchExpected(providerId, expected.providerId, 'execution evidence providerId', errors);
  matchExpected(artifactId, expected.artifactId, 'execution evidence artifactId', errors);
  matchExpected(workload, expected.workload, 'execution evidence workload', errors);
  matchExpected(
    logicalModelId,
    expected.logicalModelId,
    'execution evidence logicalModelId',
    errors
  );
  if (exactKeys(
    receipt.runtime,
    ['name', 'version', 'backendId', 'environmentFingerprint'],
    'execution evidence.runtime',
    errors
  )) {
    requiredText(receipt.runtime.name, 'execution evidence.runtime.name', errors);
    requiredText(receipt.runtime.version, 'execution evidence.runtime.version', errors);
    requiredText(receipt.runtime.backendId, 'execution evidence.runtime.backendId', errors);
    requiredSha256(
      receipt.runtime.environmentFingerprint,
      'execution evidence.runtime.environmentFingerprint',
      errors
    );
  }
  if (exactKeys(
    receipt.invocation,
    ['configurationDigest'],
    'execution evidence.invocation',
    errors
  )) {
    requiredSha256(
      receipt.invocation.configurationDigest,
      'execution evidence.invocation.configurationDigest',
      errors
    );
  }
  let status = null;
  let completedAt = null;
  if (exactKeys(
    receipt.result,
    ['status', 'outputDigest', 'startedAtUtc', 'completedAtUtc'],
    'execution evidence.result',
    errors
  )) {
    status = requiredText(receipt.result.status, 'execution evidence.result.status', errors);
    if (status && !STATUSES.has(status)) {
      errors.push('execution evidence.result.status is not recognized');
    }
    const outputDigest = nullableSha256(
      receipt.result.outputDigest,
      'execution evidence.result.outputDigest',
      errors
    );
    if (status === 'passed' && !outputDigest) {
      errors.push('passed execution evidence requires result.outputDigest');
    }
    const startedAt = isoInstant(
      receipt.result.startedAtUtc,
      'execution evidence.result.startedAtUtc',
      errors
    );
    completedAt = isoInstant(
      receipt.result.completedAtUtc,
      'execution evidence.result.completedAtUtc',
      errors
    );
    if (startedAt && completedAt && completedAt.getTime() < startedAt.getTime()) {
      errors.push('execution evidence completion predates its start');
    }
  }
  if (status !== 'passed') reasons.push(`${role || expected.role || 'external'}-execution-not-passed`);
  return {
    errors,
    reasons,
    evidenceId: errors.length === 0 ? computeRuntimeOwnershipEvidenceId(receipt) : null,
    status,
    completedAt,
  };
}

export function validateDopplerRuntimeOwnershipReceipt(receipt, expected = {}) {
  const errors = [];
  const reasons = [];
  if (!isPlainObject(receipt)) {
    return {
      errors: ['Doppler execution receipt must be an object'],
      reasons: ['doppler-execution-receipt-invalid'],
      resolution: null,
    };
  }
  if (receipt.receiptVersion !== DOPPLER_RECEIPT_SCHEMA) {
    errors.push(`Doppler execution receipt must use ${DOPPLER_RECEIPT_SCHEMA}`);
  }
  requiredText(receipt.receiptId, 'Doppler execution receipt.receiptId', errors);
  const timestamp = isoInstant(receipt.timestamp, 'Doppler execution receipt.timestamp', errors);
  if (receipt.source !== 'local') reasons.push('doppler-execution-not-local');
  if (receipt.fallbackDecision?.executed === true) reasons.push('doppler-execution-used-fallback');
  if (receipt.failure !== null) reasons.push('doppler-execution-failed');
  if (!isPlainObject(receipt.device)) reasons.push('doppler-environment-missing');
  if (receipt.resolutionStatus !== 'resolved' || !isPlainObject(receipt.resolution)) {
    reasons.push('doppler-resolution-identity-missing');
    return { errors, reasons, resolution: null, timestamp };
  }
  const resolution = receipt.resolution;
  if (resolution.schema !== RESOLUTION_SCHEMA) {
    errors.push(`Doppler execution resolution must use ${RESOLUTION_SCHEMA}`);
  }
  const logicalModelId = requiredText(
    resolution.logicalModelId,
    'Doppler execution resolution.logicalModelId',
    errors
  );
  const artifactId = requiredSha256(
    resolution.resolvedArtifactVariantId,
    'Doppler execution resolution.resolvedArtifactVariantId',
    errors
  );
  const executionId = requiredSha256(
    resolution.resolvedExecutionId,
    'Doppler execution resolution.resolvedExecutionId',
    errors
  );
  matchExpected(logicalModelId, expected.logicalModelId, 'Doppler logicalModelId', errors);
  matchExpected(artifactId, expected.resolvedArtifactVariantId, 'Doppler artifact identity', errors);
  matchExpected(executionId, expected.resolvedExecutionId, 'Doppler execution identity', errors);
  if (receipt.model?.hash != null) {
    const modelHash = requiredSha256(receipt.model.hash, 'Doppler execution model.hash', errors);
    matchExpected(modelHash, artifactId, 'Doppler model hash', errors);
  }
  return {
    errors,
    reasons,
    resolution: artifactId && executionId && logicalModelId
      ? {
        logicalModelId,
        resolvedArtifactVariantId: artifactId,
        resolvedExecutionId: executionId,
      }
      : null,
    timestamp,
  };
}
