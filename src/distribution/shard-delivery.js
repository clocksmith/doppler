import { log } from '../debug/index.js';
import {
  computeHash,
  createStreamingHasher,
  createShardWriter,
  deleteShard,
  getShardStoredSize,
  loadShard as loadShardFromStore,
  shardExists,
  streamShardRange,
} from '../storage/shard-manager.js';
import { ERROR_CODES, createDopplerError } from '../errors/doppler-error.js';
import { DEFAULT_DISTRIBUTION_CONFIG } from '../config/schema/distribution.schema.js';
import {
  P2P_TRANSPORT_CONTRACT_VERSION,
  P2P_TRANSPORT_ERROR_CODES,
  assertSupportedP2PTransportContract,
  createP2PTransportError,
  normalizeP2PTransportError,
  normalizeP2PTransportResult,
  isP2PTransportRetryable,
} from './p2p-transport-contract.js';

const DISTRIBUTION_SOURCE_CACHE = 'cache';
const DISTRIBUTION_SOURCE_P2P = 'p2p';
const DISTRIBUTION_SOURCE_HTTP = 'http';
const DISTRIBUTION_DECISION_TRACE_SCHEMA_VERSION = 1;

const DISTRIBUTION_SOURCES = Object.freeze(
  [...DEFAULT_DISTRIBUTION_CONFIG.sourceOrder]
);
const DEFAULT_SOURCE_MATRIX = Object.freeze({
  cache: { ...DEFAULT_DISTRIBUTION_CONFIG.sourceMatrix.cache },
  p2p: { ...DEFAULT_DISTRIBUTION_CONFIG.sourceMatrix.p2p },
  http: { ...DEFAULT_DISTRIBUTION_CONFIG.sourceMatrix.http },
});

const DEFAULT_P2P_TIMEOUT_MS = DEFAULT_DISTRIBUTION_CONFIG.p2p.timeoutMs;
const DEFAULT_P2P_MAX_RETRIES = DEFAULT_DISTRIBUTION_CONFIG.p2p.maxRetries;
const DEFAULT_P2P_RETRY_DELAY_MS = DEFAULT_DISTRIBUTION_CONFIG.p2p.retryDelayMs;

const inFlightDeliveries = new Map();

function normalizeDistributionSourceOrder(rawSources = []) {
  if (!Array.isArray(rawSources)) {
    return [...DISTRIBUTION_SOURCES];
  }

  const normalized = [];
  const seen = new Set();

  for (const value of rawSources) {
    const source = String(value || '').trim().toLowerCase();
    if (!DISTRIBUTION_SOURCES.includes(source)) continue;
    if (seen.has(source)) continue;
    seen.add(source);
    normalized.push(source);
  }

  return normalized.length > 0 ? normalized : [...DISTRIBUTION_SOURCES];
}

function normalizeInteger(value, fallback, allowZero = false) {
  const parsed = Number(value);
  const min = allowZero ? 0 : 1;
  return Number.isFinite(parsed) && parsed >= min && Number.isInteger(parsed)
    ? parsed
    : fallback;
}

function normalizeContentEncodings(value) {
  if (!value) return [];
  return value
    .split(',')
    .map((entry) => entry.trim().toLowerCase())
    .filter(Boolean);
}

function normalizeManifestVersionSet(value) {
  if (value === undefined || value === null) return null;
  const normalized = String(value).trim();
  return normalized || null;
}

function createShardSizeMismatchError(message, details = {}) {
  const error = createDopplerError(
    ERROR_CODES.DISTRIBUTION_SHARD_SIZE_MISMATCH,
    message
  );
  Object.assign(error, details);
  return error;
}

function parseContentLengthHeader(response, shardIndex) {
  const raw = response?.headers?.get?.('content-length');
  if (raw == null || raw === '') return null;
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw createShardSizeMismatchError(
      `Invalid content-length header for shard ${shardIndex}: ${raw}`,
      {
        code: 'http_content_length_invalid',
        headerValue: raw,
      }
    );
  }
  return parsed;
}

function parseContentRangeHeader(response, shardIndex) {
  const raw = response?.headers?.get?.('content-range');
  if (raw == null || raw.trim() === '') return null;
  const match = /^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/iu.exec(raw.trim());
  if (!match) {
    throw createShardSizeMismatchError(
      `Invalid content-range header for shard ${shardIndex}: ${raw}`,
      {
        code: 'http_content_range_invalid',
        headerValue: raw,
      }
    );
  }
  const start = Number(match[1]);
  const end = Number(match[2]);
  const total = match[3] === '*' ? null : Number(match[3]);
  if (!Number.isInteger(start) || !Number.isInteger(end) || end < start) {
    throw createShardSizeMismatchError(
      `Invalid content-range byte span for shard ${shardIndex}: ${raw}`,
      {
        code: 'http_content_range_invalid_span',
        headerValue: raw,
      }
    );
  }
  if (total != null && (!Number.isInteger(total) || total <= 0 || total <= end)) {
    throw createShardSizeMismatchError(
      `Invalid content-range total size for shard ${shardIndex}: ${raw}`,
      {
        code: 'http_content_range_invalid_total',
        headerValue: raw,
      }
    );
  }
  return {
    start,
    end,
    total,
    length: end - start + 1,
  };
}

function assertHttpResponseBoundaryHeaders(response, shardIndex, contentLength, contentRange) {
  if (response.status === 206 && !contentRange) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} returned HTTP 206 without content-range header.`,
      {
        code: 'http_content_range_missing',
      }
    );
  }
  if (contentRange && response.status !== 206) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} returned content-range header with unexpected HTTP ${response.status}.`,
      {
        code: 'http_content_range_unexpected_status',
        status: response.status,
      }
    );
  }
  if (
    contentLength != null
    && contentRange
    && contentLength !== contentRange.length
  ) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} content-length/content-range mismatch: content-length=${contentLength}, range-length=${contentRange.length}.`,
      {
        code: 'http_header_length_mismatch',
        contentLength,
        contentRangeLength: contentRange.length,
      }
    );
  }
}

function assertHttpResumeAlignment(
  response,
  shardIndex,
  resumeOffset,
  contentRange
) {
  if (!Number.isInteger(resumeOffset) || resumeOffset <= 0) {
    return { resetState: false };
  }
  if (response.status === 200) {
    return { resetState: true };
  }
  if (response.status !== 206 || !contentRange) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} resume response mismatch: expected HTTP 206 with content-range for offset ${resumeOffset}, got HTTP ${response.status}.`,
      {
        code: 'http_resume_response_mismatch',
        status: response.status,
        resumeOffset,
      }
    );
  }
  if (contentRange.start !== resumeOffset) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} resume content-range start mismatch: expected ${resumeOffset}, got ${contentRange.start}.`,
      {
        code: 'http_resume_offset_mismatch',
        resumeOffset,
        contentRangeStart: contentRange.start,
      }
    );
  }
  return { resetState: false };
}

function assertHttpPayloadBoundary(shardIndex, bytesReceived, contentLength, contentRange, expectedSize) {
  if (contentLength != null && bytesReceived !== contentLength) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} content-length mismatch: expected ${contentLength}, received ${bytesReceived}.`,
      {
        code: 'http_content_length_mismatch',
        contentLength,
        bytesReceived,
      }
    );
  }
  if (contentRange && bytesReceived !== contentRange.length) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} content-range mismatch: expected ${contentRange.length} bytes, received ${bytesReceived}.`,
      {
        code: 'http_content_range_length_mismatch',
        contentRangeLength: contentRange.length,
        bytesReceived,
      }
    );
  }
  if (contentRange?.total != null && Number.isFinite(expectedSize)) {
    const normalizedExpectedSize = Math.floor(expectedSize);
    if (normalizedExpectedSize >= 0 && contentRange.total !== normalizedExpectedSize) {
      throw createShardSizeMismatchError(
        `Shard ${shardIndex} content-range total mismatch: expected ${normalizedExpectedSize}, got ${contentRange.total}.`,
        {
          code: 'http_content_range_total_mismatch',
          expectedSize: normalizedExpectedSize,
          contentRangeTotal: contentRange.total,
        }
      );
    }
  }
}

function assertP2PPayloadRangeStart(
  shardIndex,
  rangeStart,
  expectedStart
) {
  if (rangeStart == null) {
    return;
  }
  if (!Number.isInteger(rangeStart) || rangeStart < 0) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} p2p payload rangeStart must be a non-negative integer.`,
      {
        code: 'p2p_range_start_invalid',
        rangeStart,
      }
    );
  }
  if (rangeStart !== expectedStart) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} p2p resume range mismatch: expected start ${expectedStart}, got ${rangeStart}.`,
      {
        code: 'p2p_resume_offset_mismatch',
        expectedStart,
        rangeStart,
      }
    );
  }
}

function assertP2PTotalSize(shardIndex, totalSize, expectedSize) {
  if (totalSize == null || !Number.isFinite(expectedSize)) {
    return;
  }
  const normalizedExpectedSize = Math.floor(expectedSize);
  if (totalSize !== normalizedExpectedSize) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} p2p totalSize mismatch: expected ${normalizedExpectedSize}, got ${totalSize}.`,
      {
        code: 'p2p_total_size_mismatch',
        expectedSize: normalizedExpectedSize,
        totalSize,
      }
    );
  }
}

function assertRequiredContentEncoding(response, requiredEncoding, context) {
  if (!requiredEncoding) return;
  const required = requiredEncoding.trim().toLowerCase();
  if (!required) return;
  const found = normalizeContentEncodings(response.headers.get('content-encoding'));
  if (!found.includes(required)) {
    const foundValue = found.length > 0 ? found.join(', ') : 'none';
    throw new Error(`Missing required content-encoding "${required}" for ${context} (found: ${foundValue})`);
  }
}

function buildShardUrl(baseUrl, shardInfo) {
  const base = String(baseUrl || '').replace(/\/$/, '');
  const filename = String(shardInfo?.filename || '').replace(/^\/+/, '');
  return `${base}/${filename}`;
}

function bytesToHex(bytes) {
  return Array.from(bytes)
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

function normalizeP2PConfig(config = {}) {
  const enabled = config?.enabled === true;
  const rawTimeoutMs = config?.timeoutMs;
  const rawMaxRetries = config?.maxRetries;
  const rawRetryDelayMs = config?.retryDelayMs;

  let transport = config?.transport;
  if (typeof transport !== 'function') {
    transport = null;
  }

  const contractVersion = assertSupportedP2PTransportContract(
    config?.contractVersion ?? P2P_TRANSPORT_CONTRACT_VERSION
  );

  return {
    enabled,
    timeoutMs: normalizeInteger(rawTimeoutMs, DEFAULT_P2P_TIMEOUT_MS),
    maxRetries: normalizeInteger(rawMaxRetries, DEFAULT_P2P_MAX_RETRIES, true),
    retryDelayMs: normalizeInteger(rawRetryDelayMs, DEFAULT_P2P_RETRY_DELAY_MS, true),
    transport,
    contractVersion,
  };
}

function normalizeAntiRollbackConfig(config = {}) {
  const antiRollback = config?.antiRollback && typeof config.antiRollback === 'object'
    ? config.antiRollback
    : {};
  return {
    enabled: antiRollback.enabled !== false,
    requireExpectedHash: antiRollback.requireExpectedHash !== false,
    requireExpectedSize: antiRollback.requireExpectedSize === true,
    requireManifestVersionSet: antiRollback.requireManifestVersionSet !== false,
  };
}

function normalizeDecisionTraceConfig(config = {}) {
  const sourceDecision = config?.sourceDecision && typeof config.sourceDecision === 'object'
    ? config.sourceDecision
    : {};
  const trace = sourceDecision.trace && typeof sourceDecision.trace === 'object'
    ? sourceDecision.trace
    : {};
  return {
    deterministic: sourceDecision.deterministic !== false,
    enabled: trace.enabled === true,
    includeSkippedSources: trace.includeSkippedSources !== false,
  };
}

function normalizeSourceMatrix(config = {}) {
  const matrix = config?.sourceMatrix && typeof config.sourceMatrix === 'object'
    ? config.sourceMatrix
    : {};
  const defaultMatrix = DEFAULT_SOURCE_MATRIX;
  const normalized = {};
  for (const source of DISTRIBUTION_SOURCES) {
    const entry = matrix[source] && typeof matrix[source] === 'object'
      ? matrix[source]
      : {};
    normalized[source] = {
      onHit: entry.onHit === 'return' ? 'return' : defaultMatrix[source].onHit,
      onMiss: entry.onMiss === 'terminal' ? 'terminal' : 'next',
      onFailure: entry.onFailure === 'terminal' ? 'terminal' : 'next',
    };
  }
  return normalized;
}

function createDecisionTrace(order, plan, shardIndex, deterministic, expectedManifestVersionSet) {
  return {
    schemaVersion: DISTRIBUTION_DECISION_TRACE_SCHEMA_VERSION,
    deterministic: deterministic === true,
    shardIndex,
    expectedManifestVersionSet: normalizeManifestVersionSet(expectedManifestVersionSet),
    sourceOrder: [...order],
    plan: plan.map((entry) => ({
      source: entry.source,
      enabled: entry.enabled,
      reason: entry.reason,
    })),
    attempts: [],
  };
}

function appendDecisionTraceAttempt(trace, entry) {
  if (!trace) return;
  trace.attempts.push({
    source: entry.source,
    status: entry.status,
    reason: entry.reason ?? null,
    code: entry.code ?? null,
    message: entry.message ?? null,
    durationMs: Number.isFinite(entry.durationMs) ? entry.durationMs : null,
    bytes: Number.isFinite(entry.bytes) ? entry.bytes : null,
    hash: typeof entry.hash === 'string' ? entry.hash : null,
    path: typeof entry.path === 'string' ? entry.path : null,
    manifestVersionSet: normalizeManifestVersionSet(entry.manifestVersionSet),
  });
}

function attachDecisionTrace(result, trace) {
  if (!trace) return result;
  return {
    ...result,
    decisionTrace: trace,
  };
}

function assertExpectedHash(resultHash, expectedHash, shardIndex) {
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

function assertExpectedSize(bytes, expectedSize, shardIndex) {
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

function assertExpectedManifestVersionSet(resultVersionSet, expectedVersionSet, shardIndex, source) {
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

function parseDownloadOptions(options = {}) {
  return {
    algorithm: options.algorithm,
    onProgress: options.onProgress ?? null,
    signal: options.signal,
    requiredEncoding: options.requiredEncoding ?? null,
    writeToStore: options.writeToStore ?? false,
    expectedHash: options.expectedHash ?? null,
    expectedSize: Number.isFinite(options.expectedSize) ? Math.floor(options.expectedSize) : null,
    expectedManifestVersionSet: normalizeManifestVersionSet(options.expectedManifestVersionSet),
    maxRetries: options.maxRetries,
    initialRetryDelayMs: options.initialRetryDelayMs,
    maxRetryDelayMs: options.maxRetryDelayMs,
  };
}

function createDeliveryKey(baseUrl, shardIndex, options, order, sourceMatrix) {
  return [
    String(baseUrl || ''),
    String(shardIndex),
    String(options.algorithm || ''),
    String(options.expectedHash || ''),
    String(options.expectedSize ?? ''),
    String(options.expectedManifestVersionSet ?? ''),
    JSON.stringify(sourceMatrix || null),
    String(options.writeToStore === true),
    order.join(','),
  ].join('|');
}

function createAbortError(label = 'operation aborted') {
  const error = new Error(label);
  error.name = 'AbortError';
  return error;
}

function awaitWithSignal(promise, signal, label) {
  if (!signal) return promise;
  if (signal.aborted) {
    return Promise.reject(createAbortError(label));
  }
  return new Promise((resolve, reject) => {
    const onAbort = () => reject(createAbortError(label));
    signal.addEventListener('abort', onAbort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener('abort', onAbort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener('abort', onAbort);
        reject(error);
      }
    );
  });
}

async function withTimeout(promise, timeoutMs, label = 'operation') {
  if (!timeoutMs || timeoutMs <= 0) {
    return promise;
  }

  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      const error = new Error(`${label} timed out after ${timeoutMs}ms`);
      error.name = 'TimeoutError';
      reject(error);
    }, timeoutMs);
  });

  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

export function resolveShardDeliveryPlan(options = {}) {
  const order = normalizeDistributionSourceOrder(options.sourceOrder);
  const plan = [];
  for (const source of order) {
    if (source === DISTRIBUTION_SOURCE_CACHE) {
      const enabled = options.enableSourceCache !== false;
      plan.push({
        source,
        enabled,
        reason: enabled ? 'enabled' : 'cache_disabled',
      });
      continue;
    }
    if (source === DISTRIBUTION_SOURCE_P2P) {
      const enabled = options.p2pEnabled === true && options.p2pTransportAvailable === true;
      let reason = 'enabled';
      if (options.p2pEnabled !== true) {
        reason = 'p2p_disabled';
      } else if (options.p2pTransportAvailable !== true) {
        reason = 'p2p_transport_unconfigured';
      }
      plan.push({ source, enabled, reason });
      continue;
    }
    if (source === DISTRIBUTION_SOURCE_HTTP) {
      const enabled = options.httpEnabled !== false;
      plan.push({
        source,
        enabled,
        reason: enabled ? 'enabled' : 'http_disabled',
      });
      continue;
    }
  }
  return { order, plan };
}

async function seedHasherFromStoredPrefix(hasher, shardIndex, expectedPrefixBytes) {
  if (!Number.isInteger(expectedPrefixBytes) || expectedPrefixBytes <= 0) {
    return;
  }
  let hashedBytes = 0;
  for await (const chunk of streamShardRange(shardIndex, 0, expectedPrefixBytes)) {
    if (!chunk?.byteLength) continue;
    const remaining = expectedPrefixBytes - hashedBytes;
    if (remaining <= 0) break;
    const next = chunk.byteLength > remaining
      ? chunk.subarray(0, remaining)
      : chunk;
    hasher.update(next);
    hashedBytes += next.byteLength;
    if (hashedBytes >= expectedPrefixBytes) break;
  }
  if (hashedBytes !== expectedPrefixBytes) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} stored resume prefix mismatch: expected ${expectedPrefixBytes} bytes, read ${hashedBytes}.`,
      {
        code: 'resume_state_prefix_mismatch',
        expectedPrefixBytes,
        actualPrefixBytes: hashedBytes,
      }
    );
  }
}

async function resolvePersistedResumeOffset(writeToStore, shardIndex, expectedSize) {
  if (!writeToStore) return 0;
  const storedSize = await getShardStoredSize(shardIndex);
  const resumeOffset = Number.isFinite(storedSize)
    ? Math.max(0, Math.floor(storedSize))
    : 0;
  if (resumeOffset <= 0) return 0;
  if (Number.isFinite(expectedSize)) {
    const normalizedExpected = Math.max(0, Math.floor(expectedSize));
    if (resumeOffset > normalizedExpected) {
      throw createShardSizeMismatchError(
        `Shard ${shardIndex} stored resume bytes exceed expected size: stored=${resumeOffset}, expected=${normalizedExpected}.`,
        {
          code: 'resume_state_oversize',
          storedBytes: resumeOffset,
          expectedSize: normalizedExpected,
        }
      );
    }
    if (resumeOffset === normalizedExpected) {
      return 0;
    }
  }
  return resumeOffset;
}

async function createHttpTransferState(writeToStore, shardIndex, algorithm, resumeOffset = 0) {
  const normalizedResumeOffset = Number.isInteger(resumeOffset) && resumeOffset > 0
    ? resumeOffset
    : 0;
  const hasher = await createStreamingHasher(algorithm);
  if (normalizedResumeOffset > 0) {
    await seedHasherFromStoredPrefix(hasher, shardIndex, normalizedResumeOffset);
  }
  return {
    hasher,
    chunks: writeToStore ? null : [],
    writer: writeToStore
      ? await createShardWriter(shardIndex, {
        append: normalizedResumeOffset > 0,
        expectedOffset: normalizedResumeOffset,
      })
      : null,
    writerClosed: false,
    receivedBytes: normalizedResumeOffset,
  };
}

async function resetHttpTransferState(state, writeToStore, shardIndex, algorithm) {
  await state.writer?.abort?.();
  state.hasher = await createStreamingHasher(algorithm);
  state.chunks = writeToStore ? null : [];
  state.writer = writeToStore ? await createShardWriter(shardIndex) : null;
  state.writerClosed = false;
  state.receivedBytes = 0;
}

async function appendHttpTransferChunk(state, chunk) {
  const bytes = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
  state.hasher.update(bytes);
  if (state.writer) {
    await state.writer.write(bytes);
  } else if (state.chunks) {
    state.chunks.push(bytes.slice(0));
  }
  state.receivedBytes += bytes.byteLength;
}

async function finalizeHttpTransferState(state, startTime, shardIndex) {
  const hashBytes = await state.hasher.finalize();
  const hash = bytesToHex(hashBytes);
  if (state.writer) {
    await state.writer.close();
    state.writerClosed = true;
    const elapsed = (performance.now() - startTime) / 1000;
    const speed = elapsed > 0 ? state.receivedBytes / elapsed : 0;
    const speedDisplay = `${(speed / (1024 * 1024)).toFixed(2)}MB/s`;
    log.verbose(
      'Distribution',
      `Shard ${shardIndex}: http stream (${state.receivedBytes} bytes, ${elapsed.toFixed(2)}s, ${speedDisplay})`
    );
    return {
      buffer: null,
      bytes: state.receivedBytes,
      hash,
      wrote: true,
      source: DISTRIBUTION_SOURCE_HTTP,
      path: 'http-stream-store',
    };
  }

  const buffer = !state.chunks || state.chunks.length === 0
    ? new ArrayBuffer(0)
    : await new Blob(state.chunks).arrayBuffer();
  return {
    buffer,
    bytes: buffer.byteLength,
    hash,
    wrote: false,
    source: DISTRIBUTION_SOURCE_HTTP,
    path: 'http-stream-buffer',
  };
}

async function abortHttpTransferState(state) {
  if (state.writer && !state.writerClosed) {
    await state.writer.abort?.();
    state.writerClosed = true;
  }
}

async function persistHttpTransferState(state) {
  if (!state.writer || state.writerClosed) {
    return;
  }
  if (state.receivedBytes > 0) {
    await state.writer.close();
    state.writerClosed = true;
    return;
  }
  await state.writer.abort?.();
  state.writerClosed = true;
}

async function clearPersistedShardState(shardIndex) {
  const deleted = await deleteShard(shardIndex);
  if (deleted) {
    return;
  }
  const writer = await createShardWriter(shardIndex, {
    append: false,
    expectedOffset: 0,
  });
  await writer.abort?.();
}

async function downloadShardFromHttp(baseUrl, shardInfo, shardIndex, options = {}) {
  const {
    signal,
    algorithm,
    onProgress,
    requiredEncoding,
    writeToStore = false,
  } = options;

  if (!algorithm) {
    throw new Error('Missing hash algorithm for shard download.');
  }

  const startTime = performance.now();
  const url = buildShardUrl(baseUrl, shardInfo);
  let lastError;
  const maxRetries = normalizeInteger(options.maxRetries, 3, true);
  const initialRetryDelayMs = normalizeInteger(options.initialRetryDelayMs, 1000);
  const maxRetryDelayMs = normalizeInteger(options.maxRetryDelayMs, 30000);
  const progressTotalBytes = Number.isFinite(options.expectedSize)
    ? Math.floor(options.expectedSize)
    : (Number.isFinite(shardInfo?.size) ? Math.floor(shardInfo.size) : 0);
  let retryDelay = initialRetryDelayMs;
  const disablePersistedResume = options.__disablePersistedResume === true;
  let resumeOffset = 0;
  if (!disablePersistedResume) {
    try {
      resumeOffset = await resolvePersistedResumeOffset(
        writeToStore,
        shardIndex,
        options.expectedSize
      );
    } catch (error) {
      if (writeToStore && error?.code === 'resume_state_oversize') {
        await clearPersistedShardState(shardIndex);
        resumeOffset = 0;
      } else {
        throw error;
      }
    }
  }
  const startedWithResume = resumeOffset > 0;
  let transferState;
  try {
    transferState = await createHttpTransferState(
      writeToStore,
      shardIndex,
      algorithm,
      resumeOffset
    );
  } catch (error) {
    if (writeToStore && error?.code === 'resume_state_prefix_mismatch') {
      await clearPersistedShardState(shardIndex);
      resumeOffset = 0;
      transferState = await createHttpTransferState(
        writeToStore,
        shardIndex,
        algorithm,
        0
      );
    } else {
      throw error;
    }
  }

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    try {
      const resumeOffset = transferState.receivedBytes;
      const requestHeaders = resumeOffset > 0
        ? { range: `bytes=${resumeOffset}-` }
        : undefined;
      const response = await fetch(url, { signal, headers: requestHeaders });
      if (!response.ok) {
        const error = new Error(`HTTP ${response.status}: ${response.statusText}`);
        error.status = response.status;
        throw error;
      }

      assertRequiredContentEncoding(response, requiredEncoding, `shard ${shardIndex}`);
      const contentLength = parseContentLengthHeader(response, shardIndex);
      const contentRange = parseContentRangeHeader(response, shardIndex);
      assertHttpResponseBoundaryHeaders(response, shardIndex, contentLength, contentRange);
      const { resetState } = assertHttpResumeAlignment(
        response,
        shardIndex,
        resumeOffset,
        contentRange
      );
      if (resetState) {
        await resetHttpTransferState(transferState, writeToStore, shardIndex, algorithm);
      }

      if (!response.body) {
        const buffer = await response.arrayBuffer();
        assertHttpPayloadBoundary(
          shardIndex,
          buffer.byteLength,
          contentLength,
          contentRange,
          options.expectedSize
        );
        await appendHttpTransferChunk(transferState, new Uint8Array(buffer));
        const total = progressTotalBytes > 0 ? progressTotalBytes : transferState.receivedBytes;
        const percent = total > 0
          ? Math.min(100, Math.floor((transferState.receivedBytes / total) * 100))
          : 100;
        onProgress?.({
          shardIndex,
          receivedBytes: transferState.receivedBytes,
          totalBytes: total,
          percent,
        });

        const finalized = await finalizeHttpTransferState(transferState, startTime, shardIndex);
        const result = {
          ...finalized,
          path: finalized.wrote ? finalized.path : 'http-blob',
          manifestVersionSet: options.expectedManifestVersionSet ?? null,
        };
        if (
          writeToStore
          && startedWithResume
          && options.__resumeRecoveryAttempted !== true
          && options.expectedHash
          && result.hash !== options.expectedHash
        ) {
          await clearPersistedShardState(shardIndex);
          return downloadShardFromHttp(baseUrl, shardInfo, shardIndex, {
            ...options,
            __disablePersistedResume: true,
            __resumeRecoveryAttempted: true,
          });
        }
        return result;
      }

      const reader = response.body.getReader();
      let attemptBytes = 0;

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          if (value?.length) {
            await appendHttpTransferChunk(transferState, value);
            attemptBytes += value.length;
          }

          const total = progressTotalBytes > 0 ? progressTotalBytes : transferState.receivedBytes;
          onProgress?.({
            shardIndex,
            receivedBytes: transferState.receivedBytes,
            totalBytes: total,
            percent: total > 0 ? (transferState.receivedBytes / total) * 100 : 0,
          });
        }

        assertHttpPayloadBoundary(
          shardIndex,
          attemptBytes,
          contentLength,
          contentRange,
          options.expectedSize
        );
        const finalized = await finalizeHttpTransferState(transferState, startTime, shardIndex);
        const result = {
          ...finalized,
          manifestVersionSet: options.expectedManifestVersionSet ?? null,
        };
        if (
          writeToStore
          && startedWithResume
          && options.__resumeRecoveryAttempted !== true
          && options.expectedHash
          && result.hash !== options.expectedHash
        ) {
          await clearPersistedShardState(shardIndex);
          return downloadShardFromHttp(baseUrl, shardInfo, shardIndex, {
            ...options,
            __disablePersistedResume: true,
            __resumeRecoveryAttempted: true,
          });
        }
        return result;
      } catch (error) {
        throw error;
      }
    } catch (error) {
      lastError = error;

      if (error?.name === 'AbortError') {
        if (writeToStore) {
          await persistHttpTransferState(transferState);
        } else {
          await abortHttpTransferState(transferState);
        }
        throw error;
      }

      if (Number.isInteger(error?.status) && error.status >= 400 && error.status < 500 && error.status !== 429) {
        await abortHttpTransferState(transferState);
        throw error;
      }
      if (typeof error?.code === 'string' && error.code.startsWith('http_')) {
        await abortHttpTransferState(transferState);
        throw error;
      }

      if (attempt < maxRetries) {
        await new Promise((resolve) => setTimeout(resolve, retryDelay));
        retryDelay = Math.min(retryDelay * 2, maxRetryDelayMs);
        continue;
      }

      if (writeToStore) {
        await persistHttpTransferState(transferState);
      } else {
        await abortHttpTransferState(transferState);
      }
    }
  }

  if (writeToStore) {
    await persistHttpTransferState(transferState);
  } else {
    await abortHttpTransferState(transferState);
  }
  throw lastError;
}

async function downloadShardFromP2P(shardIndex, shardInfo, p2pConfig, options = {}) {
  const transport = p2pConfig.transport;
  if (!p2pConfig.enabled || typeof transport !== 'function') {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.unconfigured,
      'P2P transport is not configured',
      { shardIndex }
    );
  }

  const writeToStore = options.writeToStore === true;
  const algorithm = options.algorithm;
  if (writeToStore && !algorithm) {
    throw new Error(`Missing hash algorithm for shard ${shardIndex} p2p transfer.`);
  }

  const expectedSize = Number.isFinite(options.expectedSize)
    ? Math.floor(options.expectedSize)
    : null;
  let seededResumeOffset = 0;
  let transferState = null;
  if (writeToStore) {
    try {
      seededResumeOffset = await resolvePersistedResumeOffset(
        true,
        shardIndex,
        expectedSize
      );
    } catch (error) {
      if (error?.code === 'resume_state_oversize') {
        await clearPersistedShardState(shardIndex);
        seededResumeOffset = 0;
      } else {
        throw error;
      }
    }
    try {
      transferState = await createHttpTransferState(
        true,
        shardIndex,
        algorithm,
        seededResumeOffset
      );
    } catch (error) {
      if (error?.code === 'resume_state_prefix_mismatch') {
        await clearPersistedShardState(shardIndex);
        seededResumeOffset = 0;
        transferState = await createHttpTransferState(true, shardIndex, algorithm, 0);
      } else {
        throw error;
      }
    }
  }

  const startTime = performance.now();
  let lastError = null;
  const maxRetries = Math.max(0, p2pConfig.maxRetries);
  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    try {
      const requestResumeOffset = transferState?.receivedBytes ?? 0;
      const transportResult = await withTimeout(
        transport({
          shardIndex,
          shardInfo,
          signal: options.signal,
          source: DISTRIBUTION_SOURCE_P2P,
          timeoutMs: p2pConfig.timeoutMs,
          contractVersion: p2pConfig.contractVersion,
          attempt,
          maxRetries,
          resumeOffset: requestResumeOffset,
          expectedHash: options.expectedHash ?? null,
          expectedSize: options.expectedSize ?? null,
          expectedManifestVersionSet: options.expectedManifestVersionSet ?? null,
        }),
        p2pConfig.timeoutMs,
        `P2P shard ${shardIndex}`
      );
      const payload = normalizeP2PTransportResult(
        transportResult,
        `P2P transport result for shard ${shardIndex}`
      );
      if (!payload) {
        throw createP2PTransportError(
          P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
          `P2P transport returned empty payload for shard ${shardIndex}`,
          { shardIndex }
        );
      }

      const payloadRangeStart = payload.rangeStart;
      const payloadTotalSize = payload.totalSize;
      assertP2PTotalSize(shardIndex, payloadTotalSize, expectedSize);

      const onProgress = options.onProgress ?? null;
      return {
        ...(await (async () => {
          if (!writeToStore) {
            assertP2PPayloadRangeStart(shardIndex, payloadRangeStart, 0);
            onProgress?.({
              shardIndex,
              receivedBytes: payload.data.byteLength,
              totalBytes: expectedSize ?? payloadTotalSize ?? payload.data.byteLength,
              percent: 100,
            });
            return {
              buffer: payload.data,
              bytes: payload.data.byteLength,
              source: DISTRIBUTION_SOURCE_P2P,
              path: 'p2p-transport',
              wrote: false,
            };
          }

          let effectiveRangeStart = payloadRangeStart;
          if (effectiveRangeStart == null) {
            effectiveRangeStart = requestResumeOffset;
          }
          if (requestResumeOffset > 0 && effectiveRangeStart === 0) {
            await resetHttpTransferState(transferState, true, shardIndex, algorithm);
          } else {
            assertP2PPayloadRangeStart(
              shardIndex,
              effectiveRangeStart,
              transferState.receivedBytes
            );
          }
          await appendHttpTransferChunk(transferState, new Uint8Array(payload.data));
          onProgress?.({
            shardIndex,
            receivedBytes: transferState.receivedBytes,
            totalBytes: expectedSize ?? payloadTotalSize ?? transferState.receivedBytes,
            percent: 100,
          });
          const finalized = await finalizeHttpTransferState(transferState, startTime, shardIndex);
          if (Number.isFinite(expectedSize)) {
            assertExpectedSize(finalized.bytes, expectedSize, shardIndex);
          } else if (Number.isInteger(payloadTotalSize)) {
            assertExpectedSize(finalized.bytes, payloadTotalSize, shardIndex);
          }
          return {
            ...finalized,
            source: DISTRIBUTION_SOURCE_P2P,
            path: 'p2p-stream-store',
          };
        })()),
        manifestVersionSet: normalizeManifestVersionSet(
          payload.manifestVersionSet ?? options.expectedManifestVersionSet
        ),
      };
    } catch (error) {
      if (typeof error?.code === 'string' && error.code.startsWith('p2p_')) {
        if (writeToStore) {
          await clearPersistedShardState(shardIndex);
        }
        throw error;
      }

      const normalized = normalizeP2PTransportError(error, {
        shardIndex,
        attempt,
        maxRetries,
        label: `P2P shard ${shardIndex}`,
      });
      lastError = normalized;
      if (normalized?.code === P2P_TRANSPORT_ERROR_CODES.aborted) {
        if (writeToStore) {
          await persistHttpTransferState(transferState);
        }
        const abortError = createAbortError(normalized.message || 'P2P transport aborted');
        throw abortError;
      }
      if (attempt < maxRetries && isP2PTransportRetryable(normalized)) {
        await new Promise((resolve) => setTimeout(resolve, p2pConfig.retryDelayMs));
        continue;
      }
      if (writeToStore) {
        await persistHttpTransferState(transferState);
      }
      throw normalized;
    }
  }

  if (writeToStore) {
    await persistHttpTransferState(transferState);
  }
  throw lastError;
}

async function executeDeliveryPlan(
  baseUrl,
  shardIndex,
  shardInfo,
  plan,
  p2p,
  options,
  trace,
  decisionTraceConfig,
  sourceMatrix
) {
  let lastError = null;
  const enabledSources = plan.filter((entry) => entry.enabled);

  for (const step of plan) {
    if (!step.enabled) {
      if (decisionTraceConfig.includeSkippedSources === true) {
        appendDecisionTraceAttempt(trace, {
          source: step.source,
          status: 'skipped',
          reason: step.reason,
        });
      }
      continue;
    }

    const attemptStart = performance.now();
    try {
      let result = null;
      if (step.source === DISTRIBUTION_SOURCE_CACHE) {
        if (!(await shardExists(shardIndex))) {
          const cacheMiss = new Error(`Shard ${shardIndex} missing from local cache`);
          cacheMiss.code = 'cache_miss';
          throw cacheMiss;
        }
        const buffer = await loadShardFromStore(shardIndex, { verify: false });
        result = {
          buffer,
          bytes: buffer.byteLength,
          hash: await computeHash(buffer, options.algorithm),
          wrote: false,
          source: DISTRIBUTION_SOURCE_CACHE,
          path: 'cache',
          manifestVersionSet: options.expectedManifestVersionSet ?? null,
        };
      } else if (step.source === DISTRIBUTION_SOURCE_P2P) {
        result = await downloadShardFromP2P(shardIndex, shardInfo, p2p, options);
        if (!result.hash) {
          if (!(result.buffer instanceof ArrayBuffer)) {
            throw new Error(`Shard ${shardIndex} p2p result missing hash and buffer.`);
          }
          result.hash = await computeHash(result.buffer, options.algorithm);
        }
      } else if (step.source === DISTRIBUTION_SOURCE_HTTP) {
        result = await downloadShardFromHttp(baseUrl, shardInfo, shardIndex, { ...options });
      }

      assertExpectedManifestVersionSet(
        result.manifestVersionSet,
        options.expectedManifestVersionSet,
        shardIndex,
        step.source
      );
      assertExpectedHash(result.hash, options.expectedHash, shardIndex);
      assertExpectedSize(result.bytes, options.expectedSize, shardIndex);

      appendDecisionTraceAttempt(trace, {
        source: step.source,
        status: 'success',
        durationMs: performance.now() - attemptStart,
        bytes: result.bytes,
        hash: result.hash,
        path: result.path,
        manifestVersionSet: result.manifestVersionSet,
      });
      return result;
    } catch (error) {
      if (error?.name === 'AbortError') {
        throw error;
      }
      lastError = error;
      appendDecisionTraceAttempt(trace, {
        source: step.source,
        status: 'failed',
        reason: step.reason,
        code: error?.code || null,
        message: error?.message || String(error),
        durationMs: performance.now() - attemptStart,
      });
      const enabledIndex = enabledSources.findIndex((entry) => entry.source === step.source);
      const isLastEnabled = enabledIndex === enabledSources.length - 1;
      const transitionType = (
        error?.code === 'cache_miss'
        || error?.code === 'p2p_unconfigured'
        || error?.code === P2P_TRANSPORT_ERROR_CODES.unconfigured
        || error?.code === P2P_TRANSPORT_ERROR_CODES.unavailable
      )
        ? 'onMiss'
        : 'onFailure';
      const transition = sourceMatrix?.[step.source]?.[transitionType] || 'next';
      if (isLastEnabled || transition === 'terminal') {
        log.warn('Distribution', `All shard delivery sources failed for shard ${shardIndex}: ${error.message}`);
        throw error;
      }
      log.warn('Distribution', `Shard ${shardIndex} source "${step.source}" failed (${error.code || 'error'}): ${error.message}`);
      continue;
    }
  }

  throw lastError || new Error(`No shard delivery source available for shard ${shardIndex}`);
}

export async function downloadShard(
  baseUrl,
  shardIndex,
  shardInfo,
  options = {}
) {
  const {
    sourceOrder,
    distributionConfig = {},
    distribution = {},
    maxRetries,
    initialRetryDelayMs,
    maxRetryDelayMs,
    requiredEncoding,
    algorithm,
    signal,
    onProgress = null,
    writeToStore = false,
    enableSourceCache = true,
    p2pTransport,
    expectedSize,
  } = options;

  if (!algorithm) {
    throw new Error('Missing hash algorithm for shard download verification.');
  }

  const activeConfig = {
    ...(distributionConfig || {}),
    ...distribution,
    sourceOrder: sourceOrder || distributionConfig?.sourceOrder || distributionConfig?.sources,
  };

  const antiRollback = normalizeAntiRollbackConfig(activeConfig);
  const decisionTraceConfig = normalizeDecisionTraceConfig(activeConfig);
  const sourceMatrix = normalizeSourceMatrix(activeConfig);
  const order = normalizeDistributionSourceOrder(activeConfig.sourceOrder);

  const p2p = normalizeP2PConfig({
    ...activeConfig.p2p,
    transport: activeConfig?.p2p?.transport || p2pTransport,
  });

  const downloadOptions = parseDownloadOptions({
    ...options,
    algorithm,
    onProgress,
    signal,
    requiredEncoding: requiredEncoding ?? activeConfig.requiredContentEncoding ?? null,
    expectedHash: options.expectedHash ?? shardInfo?.hash ?? activeConfig.expectedHash ?? null,
    expectedSize: expectedSize ?? shardInfo?.size ?? null,
    expectedManifestVersionSet: options.expectedManifestVersionSet ?? null,
    writeToStore,
    maxRetries: maxRetries ?? activeConfig.maxRetries,
    initialRetryDelayMs: initialRetryDelayMs ?? activeConfig.initialRetryDelayMs,
    maxRetryDelayMs: maxRetryDelayMs ?? activeConfig.maxRetryDelayMs,
  });

  if (antiRollback.enabled && antiRollback.requireExpectedHash && !downloadOptions.expectedHash) {
    throw createDopplerError(
      ERROR_CODES.DISTRIBUTION_SHARD_HASH_MISMATCH,
      `Missing expected hash for shard ${shardIndex} while antiRollback.requireExpectedHash=true.`
    );
  }

  if (
    antiRollback.enabled
    && antiRollback.requireExpectedSize
    && !Number.isFinite(downloadOptions.expectedSize)
  ) {
    throw createDopplerError(
      ERROR_CODES.DISTRIBUTION_SHARD_SIZE_MISMATCH,
      `Missing expected size for shard ${shardIndex} while antiRollback.requireExpectedSize=true.`
    );
  }

  if (
    antiRollback.enabled
    && antiRollback.requireManifestVersionSet
    && !downloadOptions.expectedManifestVersionSet
  ) {
    throw createDopplerError(
      ERROR_CODES.DISTRIBUTION_SHARD_MANIFEST_VERSION_SET_MISMATCH,
      `Missing expected manifestVersionSet for shard ${shardIndex} while antiRollback.requireManifestVersionSet=true.`
    );
  }

  const planResult = resolveShardDeliveryPlan({
    sourceOrder: order,
    enableSourceCache,
    p2pEnabled: p2p.enabled,
    p2pTransportAvailable: typeof p2p.transport === 'function',
    httpEnabled: true,
  });

  const trace = decisionTraceConfig.enabled
    ? createDecisionTrace(
      order,
      planResult.plan,
      shardIndex,
      decisionTraceConfig.deterministic,
      downloadOptions.expectedManifestVersionSet
    )
    : null;

  const dedupeKey = createDeliveryKey(baseUrl, shardIndex, downloadOptions, order, sourceMatrix);
  if (inFlightDeliveries.has(dedupeKey)) {
    return await awaitWithSignal(
      inFlightDeliveries.get(dedupeKey),
      signal,
      `Shard ${shardIndex} delivery aborted`
    );
  }

  const deliveryPromise = (async () => {
    const result = await executeDeliveryPlan(
      baseUrl,
      shardIndex,
      shardInfo,
      planResult.plan,
      p2p,
      downloadOptions,
      trace,
      decisionTraceConfig,
      sourceMatrix
    );
    return attachDecisionTrace(result, trace);
  })();

  inFlightDeliveries.set(dedupeKey, deliveryPromise);
  try {
    return await awaitWithSignal(
      deliveryPromise,
      signal,
      `Shard ${shardIndex} delivery aborted`
    );
  } finally {
    inFlightDeliveries.delete(dedupeKey);
  }
}

export function getSourceOrder(config = {}) {
  return normalizeDistributionSourceOrder(config.sourceOrder || config.sources || DISTRIBUTION_SOURCES);
}

export function getInFlightShardDeliveryCount() {
  return inFlightDeliveries.size;
}
