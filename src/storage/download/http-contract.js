import { ERROR_CODES, createDopplerError } from '../../errors/doppler-error.js';
import { DEFAULT_DISTRIBUTION_CONFIG } from '../../config/schema/distribution.schema.js';
export const DISTRIBUTION_SOURCE_HTTP = 'http';

export function normalizeRequiredInteger(value, label, { allowZero = false, fallback = null } = {}) {
  if (value === undefined || value === null) {
    if (fallback !== null) {
      return fallback;
    }
    throw new Error(`${label} is required.`);
  }
  const parsed = Number(value);
  const min = allowZero ? 0 : 1;
  if (!Number.isInteger(parsed) || parsed < min) {
    throw new Error(
      `${label} must be a ${allowZero ? 'non-negative' : 'positive'} integer when provided.`
    );
  }
  return parsed;
}

export function createShardSizeMismatchError(message, details = {}) {
  const error = createDopplerError(
    ERROR_CODES.DISTRIBUTION_SHARD_SIZE_MISMATCH,
    message
  );
  Object.assign(error, details);
  return error;
}
const DISTRIBUTION_SOURCE_CACHE = 'cache';
const DISTRIBUTION_SOURCE_P2P = 'p2p';
const DISTRIBUTION_SOURCES = DEFAULT_DISTRIBUTION_CONFIG.sourceOrder;
const DISTRIBUTION_DELIVERY_METRICS_SCHEMA_VERSION = 1;
export function createSourceCounter() {
  return {
    cache: 0,
    p2p: 0,
    http: 0,
  };
}

export function createLatencySummary(durations) {
  const values = durations.filter((value) => Number.isFinite(value));
  if (values.length === 0) {
    return {
      count: 0,
      min: null,
      max: null,
      avg: null,
    };
  }
  let sum = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    sum += value;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return {
    count: values.length,
    min,
    max,
    avg: sum / values.length,
  };
}

export function createDeliveryMetrics(order, result, attempts, totalDurationMs) {
  const sourceAttempts = createSourceCounter();
  const retries = createSourceCounter();
  const failureCodes = {};
  const p2pDurations = [];
  const httpDurations = [];
  let storageWriteMs = Number.isFinite(result?.writeDurationMs) ? result.writeDurationMs : null;
  let attemptCount = 0;
  const attemptsBySource = createSourceCounter();

  for (const attempt of attempts) {
    if (attempt?.status !== 'success' && attempt?.status !== 'failed') {
      continue;
    }
    attemptCount += 1;
    const source = attempt?.source;
    if (source === DISTRIBUTION_SOURCE_CACHE || source === DISTRIBUTION_SOURCE_P2P || source === DISTRIBUTION_SOURCE_HTTP) {
      sourceAttempts[source] += 1;
      attemptsBySource[source] += 1;
      if (source === DISTRIBUTION_SOURCE_P2P && Number.isFinite(attempt.durationMs)) {
        p2pDurations.push(attempt.durationMs);
      }
      if (source === DISTRIBUTION_SOURCE_HTTP && Number.isFinite(attempt.durationMs)) {
        httpDurations.push(attempt.durationMs);
      }
    }
    if (attempt.status === 'failed') {
      const code = typeof attempt.code === 'string' && attempt.code
        ? attempt.code
        : 'unknown';
      failureCodes[code] = (failureCodes[code] ?? 0) + 1;
    }
    if (storageWriteMs == null && Number.isFinite(attempt.writeDurationMs)) {
      storageWriteMs = attempt.writeDurationMs;
    }
  }

  for (const source of [DISTRIBUTION_SOURCE_CACHE, DISTRIBUTION_SOURCE_P2P, DISTRIBUTION_SOURCE_HTTP]) {
    retries[source] = Math.max(0, attemptsBySource[source] - 1);
  }

  return {
    schemaVersion: DISTRIBUTION_DELIVERY_METRICS_SCHEMA_VERSION,
    totalDurationMs: Number.isFinite(totalDurationMs) ? totalDurationMs : 0,
    sourceOrder: Array.isArray(order) ? [...order] : [...DISTRIBUTION_SOURCES],
    successSource: result?.source ?? null,
    attemptCount,
    sourceAttempts,
    retries,
    failureCodes,
    p2pRttMs: createLatencySummary(p2pDurations),
    httpRttMs: createLatencySummary(httpDurations),
    storageWriteMs,
  };
}

export function normalizeManifestVersionSet(value) {
  if (value === undefined || value === null) return null;
  const normalized = String(value).trim();
  return normalized || null;
}

export function assertExpectedHash(resultHash, expectedHash, shardIndex) {
  if (!expectedHash) return;
  if (!resultHash) {
    const error = createDopplerError(
      ERROR_CODES.DISTRIBUTION_SHARD_HASH_MISMATCH,
      `Shard ${shardIndex} missing hash result`
    );
    error.code = 'hash_missing';
    throw error;
  }
  if (resultHash !== expectedHash) {
    const error = createDopplerError(
      ERROR_CODES.DISTRIBUTION_SHARD_HASH_MISMATCH,
      `Hash mismatch for shard ${shardIndex}: expected ${expectedHash}, got ${resultHash}`
    );
    error.code = 'hash_mismatch';
    error.expectedHash = expectedHash;
    error.actualHash = resultHash;
    throw error;
  }
}

export function assertExpectedSize(bytes, expectedSize, shardIndex) {
  if (!Number.isFinite(expectedSize)) return;
  const expected = Math.floor(expectedSize);
  const actual = Number.isFinite(bytes) ? Math.floor(bytes) : -1;
  if (expected < 0 || actual < 0) return;
  if (actual !== expected) {
    const error = createDopplerError(
      ERROR_CODES.DISTRIBUTION_SHARD_SIZE_MISMATCH,
      `Size mismatch for shard ${shardIndex}: expected ${expected}, got ${actual}`
    );
    error.code = 'size_mismatch';
    error.expectedSize = expected;
    error.actualSize = actual;
    throw error;
  }
}

export function assertExpectedManifestVersionSet(resultVersionSet, expectedVersionSet, shardIndex, source) {
  const expected = normalizeManifestVersionSet(expectedVersionSet);
  if (!expected) return;
  const actual = normalizeManifestVersionSet(resultVersionSet);
  if (!actual) {
    const error = createDopplerError(
      ERROR_CODES.DISTRIBUTION_SHARD_MANIFEST_VERSION_SET_MISMATCH,
      `Shard ${shardIndex} source "${source}" missing manifestVersionSet while antiRollback.requireManifestVersionSet=true.`
    );
    error.code = 'manifest_version_set_missing';
    error.expectedManifestVersionSet = expected;
    error.actualManifestVersionSet = actual;
    throw error;
  }
  if (actual !== expected) {
    const error = createDopplerError(
      ERROR_CODES.DISTRIBUTION_SHARD_MANIFEST_VERSION_SET_MISMATCH,
      `Shard ${shardIndex} source "${source}" manifestVersionSet mismatch: expected ${expected}, got ${actual}`
    );
    error.code = 'manifest_version_set_mismatch';
    error.expectedManifestVersionSet = expected;
    error.actualManifestVersionSet = actual;
    throw error;
  }
}

export function createAbortError(label = 'operation aborted') {
  const error = new Error(label);
  error.name = 'AbortError';
  return error;
}
