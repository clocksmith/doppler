import { cloneJsonValue } from '../utils/clone-json.js';

export const KERNEL_NEGATIVE_RESULTS_SCHEMA = 'doppler.kernel-negative-results/v1';

const DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/;

function requireObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`kernel negative results: ${label} must be an object`);
  }
  return value;
}

function requireString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`kernel negative results: ${label} must be a non-empty string`);
  }
  return value;
}

function requireDigest(value, label) {
  if (typeof value !== 'string' || !DIGEST_PATTERN.test(value)) {
    throw new Error(`kernel negative results: ${label} must be sha256:<64 lowercase hex>`);
  }
  return value;
}

function requireFinite(value, label) {
  if (!Number.isFinite(value)) {
    throw new Error(`kernel negative results: ${label} must be finite`);
  }
  return value;
}

function validateEntry(entry, index, ids) {
  const label = `entries[${index}]`;
  requireObject(entry, label);
  const id = requireString(entry.id, `${label}.id`);
  if (ids.has(id)) {
    throw new Error(`kernel negative results: duplicate id "${id}"`);
  }
  ids.add(id);
  requireString(entry.recordedAtUtc, `${label}.recordedAtUtc`);
  requireString(entry.hypothesis, `${label}.hypothesis`);

  const scope = requireObject(entry.scope, `${label}.scope`);
  for (const field of ['modelId', 'candidate', 'surface', 'runtime', 'phase']) {
    requireString(scope[field], `${label}.scope.${field}`);
  }
  for (const field of ['adapterDigest', 'executionGraphDigest', 'artifactDigest']) {
    requireDigest(scope[field], `${label}.scope.${field}`);
  }
  requireObject(scope.adapter, `${label}.scope.adapter`);
  for (const field of ['promptTokenCount', 'generatedTokenCount']) {
    if (!Number.isInteger(scope[field]) || scope[field] <= 0) {
      throw new Error(`kernel negative results: ${label}.scope.${field} must be positive`);
    }
  }

  const correctness = requireObject(entry.correctness, `${label}.correctness`);
  requireDigest(correctness.generatedTokenIdsHash, `${label}.correctness.generatedTokenIdsHash`);
  requireDigest(correctness.generatedTextHash, `${label}.correctness.generatedTextHash`);
  if (correctness.exactTokenParity !== true) {
    throw new Error(`kernel negative results: ${label} must record an exact parity result`);
  }
  if (correctness.promotionEligible !== false) {
    throw new Error(`kernel negative results: ${label} cannot be promotion eligible`);
  }

  const measurement = requireObject(entry.measurement, `${label}.measurement`);
  if (!Array.isArray(measurement.order) || measurement.order.length < 2) {
    throw new Error(`kernel negative results: ${label}.measurement.order is required`);
  }
  for (const field of [
    'baselineMeanDecodeTokensPerSec',
    'candidateMeanDecodeTokensPerSec',
    'throughputDeltaPercent',
    'baselineMeanDecodeMs',
    'candidateMeanDecodeMs',
    'decodeTimeDeltaPercent',
    'baselineMeanRecordMs',
    'candidateMeanRecordMs',
    'recordTimeDeltaPercent',
  ]) {
    requireFinite(measurement[field], `${label}.measurement.${field}`);
  }
  for (const [name, digest] of Object.entries(
    requireObject(measurement.rawEnvelopeSha256, `${label}.measurement.rawEnvelopeSha256`)
  )) {
    requireDigest(digest, `${label}.measurement.rawEnvelopeSha256.${name}`);
  }

  const decision = requireObject(entry.decision, `${label}.decision`);
  if (decision.status !== 'rejected' || decision.runtimeMutationApplied !== false) {
    throw new Error(
      `kernel negative results: ${label} must be rejected with runtimeMutationApplied=false`
    );
  }
  requireString(decision.reason, `${label}.decision.reason`);
  if (!Array.isArray(decision.retryOnlyWhen) || decision.retryOnlyWhen.length === 0) {
    throw new Error(`kernel negative results: ${label}.decision.retryOnlyWhen is required`);
  }
}

export function validateKernelNegativeResults(input) {
  const registry = cloneJsonValue(requireObject(input, 'registry'));
  if (
    registry.schema !== KERNEL_NEGATIVE_RESULTS_SCHEMA
    || registry.$schema !== KERNEL_NEGATIVE_RESULTS_SCHEMA
  ) {
    throw new Error(
      `kernel negative results: schema and $schema must be "${KERNEL_NEGATIVE_RESULTS_SCHEMA}"`
    );
  }
  if (!Array.isArray(registry.entries)) {
    throw new Error('kernel negative results: entries must be an array');
  }
  const ids = new Set();
  registry.entries.forEach((entry, index) => validateEntry(entry, index, ids));
  return registry;
}

export function findKernelNegativeResults(input, scope = {}) {
  const registry = validateKernelNegativeResults(input);
  const filters = ['modelId', 'candidate', 'adapterDigest', 'phase'];
  return registry.entries.filter((entry) => (
    filters.every((field) => scope[field] == null || entry.scope[field] === scope[field])
  ));
}
