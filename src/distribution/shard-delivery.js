import { log } from '../debug/index.js';
import {
  computeHash,
  createStreamingHasher,
  createShardWriter,
  loadShard as loadShardFromStore,
  shardExists,
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
  let retryDelay = initialRetryDelayMs;

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    try {
      const response = await fetch(url, { signal });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      assertRequiredContentEncoding(response, requiredEncoding, `shard ${shardIndex}`);

      if (!response.body) {
        const buffer = await response.arrayBuffer();
        const hash = await computeHash(buffer, algorithm);
        const percent = shardInfo?.size
          ? Math.min(100, Math.floor((buffer.byteLength / shardInfo.size) * 100))
          : 100;
        onProgress?.({
          shardIndex,
          receivedBytes: buffer.byteLength,
          totalBytes: shardInfo.size ?? buffer.byteLength,
          percent,
        });

        return {
          buffer,
          bytes: buffer.byteLength,
          hash,
          wrote: false,
          source: DISTRIBUTION_SOURCE_HTTP,
          path: 'http-blob',
          manifestVersionSet: options.expectedManifestVersionSet ?? null,
        };
      }

      const totalBytes = shardInfo?.size ?? 0;
      const reader = response.body.getReader();
      const hasher = await createStreamingHasher(algorithm);
      const chunks = writeToStore ? null : [];
      const writer = writeToStore ? await createShardWriter(shardIndex) : null;
      let receivedBytes = 0;

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          if (value?.length) {
            hasher.update(value);
            if (writer) {
              await writer.write(value);
            } else {
              chunks.push(value);
            }
            receivedBytes += value.length;
          }

          onProgress?.({
            shardIndex,
            receivedBytes,
            totalBytes,
            percent: totalBytes > 0 ? (receivedBytes / totalBytes) * 100 : 0,
          });
        }

        const hashBytes = await hasher.finalize();
        const hash = bytesToHex(hashBytes);

        if (writer) {
          await writer.close();
          const elapsed = (performance.now() - startTime) / 1000;
          const speed = elapsed > 0 ? receivedBytes / elapsed : 0;
          const speedDisplay = `${(speed / (1024 * 1024)).toFixed(2)}MB/s`;
          log.verbose('Distribution', `Shard ${shardIndex}: http stream (${receivedBytes} bytes, ${elapsed.toFixed(2)}s, ${speedDisplay})`);
          return {
            buffer: null,
            bytes: receivedBytes,
            hash,
            wrote: true,
            source: DISTRIBUTION_SOURCE_HTTP,
            path: 'http-stream-store',
            manifestVersionSet: options.expectedManifestVersionSet ?? null,
          };
        }

        const buffer = new Blob(chunks).size === 0
          ? new ArrayBuffer(0)
          : await new Blob(chunks).arrayBuffer();

        return {
          buffer,
          bytes: buffer.byteLength,
          hash,
          wrote: false,
          source: DISTRIBUTION_SOURCE_HTTP,
          path: 'http-stream-buffer',
          manifestVersionSet: options.expectedManifestVersionSet ?? null,
        };
      } catch (error) {
        await writer?.abort?.();
        throw error;
      }
    } catch (error) {
      lastError = error;

      if (error?.name === 'AbortError') {
        throw error;
      }

      if (String(error?.message || '').includes('HTTP 4') && !String(error?.message || '').includes('HTTP 429')) {
        throw error;
      }

      if (attempt < maxRetries) {
        await new Promise((resolve) => setTimeout(resolve, retryDelay));
        retryDelay = Math.min(retryDelay * 2, maxRetryDelayMs);
        continue;
      }
    }
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

  let lastError = null;
  const maxRetries = Math.max(0, p2pConfig.maxRetries);
  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    try {
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

      return {
        buffer: payload.data,
        bytes: payload.data.byteLength,
        source: DISTRIBUTION_SOURCE_P2P,
        path: 'p2p-transport',
        wrote: false,
        manifestVersionSet: normalizeManifestVersionSet(
          payload.manifestVersionSet ?? options.expectedManifestVersionSet
        ),
      };
    } catch (error) {
      const normalized = normalizeP2PTransportError(error, {
        shardIndex,
        attempt,
        maxRetries,
        label: `P2P shard ${shardIndex}`,
      });
      lastError = normalized;
      if (normalized?.code === P2P_TRANSPORT_ERROR_CODES.aborted) {
        const abortError = createAbortError(normalized.message || 'P2P transport aborted');
        throw abortError;
      }
      if (attempt < maxRetries && isP2PTransportRetryable(normalized)) {
        await new Promise((resolve) => setTimeout(resolve, p2pConfig.retryDelayMs));
        continue;
      }
      throw normalized;
    }
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
        result.hash = await computeHash(result.buffer, options.algorithm);
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
