import { log } from '../debug/index.js';
import { getExpectedShardHash } from '../formats/rdrr/index.js';
import { computeHash, loadShard as loadShardFromStore, shardExists } from './shard-manager.js';
import { ERROR_CODES, createDopplerError } from '../errors/doppler-error.js';
import { DEFAULT_DISTRIBUTION_CONFIG } from '../config/schema/distribution.schema.js';
import { downloadShardFromHttp, createHttpShardTransport } from './download/http-transport.js';
import { createDeliveryMetrics, assertExpectedHash, assertExpectedSize,
  assertExpectedManifestVersionSet, createAbortError, normalizeManifestVersionSet } from './download/http-contract.js';
import { getDistributionConfig } from './download-types.js';
const DISTRIBUTION_SOURCE_CACHE = 'cache';
const DISTRIBUTION_SOURCE_HTTP = 'http';
const DISTRIBUTION_SOURCE_P2P = 'p2p';
const DISTRIBUTION_SOURCES = Object.freeze([...DEFAULT_DISTRIBUTION_CONFIG.sourceOrder]);

const DISTRIBUTION_DECISION_TRACE_SCHEMA_VERSION = 1;

const DISTRIBUTION_DELIVERY_METRICS_EVENT_SCHEMA_VERSION = 1;

const DEFAULT_SOURCE_MATRIX = Object.freeze({
  cache: { ...DEFAULT_DISTRIBUTION_CONFIG.sourceMatrix.cache },
  p2p: { ...DEFAULT_DISTRIBUTION_CONFIG.sourceMatrix.p2p },
  http: { ...DEFAULT_DISTRIBUTION_CONFIG.sourceMatrix.http },
});

const inFlightDeliveries = new Map();
const deliveryOwners = new WeakMap();
let nextDeliveryOwner = 1;

function deliveryOwnerKey(options, peer) {
  return [options.store, options.signal, peer.transport].filter(Boolean).map((owner) => {
    if (!deliveryOwners.has(owner)) deliveryOwners.set(owner, nextDeliveryOwner++);
    return deliveryOwners.get(owner);
  }).join(':');
}

function normalizeDistributionSourceOrder(rawSources = []) {
  if (rawSources === undefined || rawSources === null) {
    return [...DISTRIBUTION_SOURCES];
  }
  if (!Array.isArray(rawSources)) {
    throw new Error('distribution.sourceOrder must be an array when provided.');
  }

  const normalized = [];
  const seen = new Set();

  for (const value of rawSources) {
    const source = String(value || '').trim().toLowerCase();
    if (!DISTRIBUTION_SOURCES.includes(source)) {
      throw new Error(`distribution.sourceOrder contains unsupported source "${source || value}".`);
    }
    if (seen.has(source)) continue;
    seen.add(source);
    normalized.push(source);
  }

  if (normalized.length === 0) {
    throw new Error('distribution.sourceOrder must include at least one supported source.');
  }
  return normalized;
}

function normalizeInteger(value, fallback, allowZero = false) {
  const parsed = Number(value);
  const min = allowZero ? 0 : 1;
  return Number.isFinite(parsed) && parsed >= min && Number.isInteger(parsed)
    ? parsed
    : fallback;
}

function normalizeSamplingRate(value, fallback = 1, label = 'distribution.sourceDecision.trace.samplingRate') {
  if (value === undefined || value === null) {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${label} must be a finite number between 0 and 1 when provided.`);
  }
  if (parsed < 0 || parsed > 1) {
    throw new Error(`${label} must be between 0 and 1 when provided.`);
  }
  return parsed;
}

function hashStringToUnitInterval(value) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}

function shouldEmitDecisionTrace(config, shardIndex, expectedManifestVersionSet, sourceOrder) {
  if (config.enabled !== true) {
    return false;
  }
  const samplingRate = normalizeSamplingRate(config.samplingRate, 1);
  if (samplingRate >= 1) {
    return true;
  }
  if (samplingRate <= 0) {
    return false;
  }
  if (config.deterministic !== false) {
    const seed = [
      String(shardIndex),
      normalizeManifestVersionSet(expectedManifestVersionSet) ?? '',
      Array.isArray(sourceOrder) ? sourceOrder.join(',') : '',
    ].join('|');
    return hashStringToUnitInterval(seed) < samplingRate;
  }
  return Math.random() < samplingRate;
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
    samplingRate: normalizeSamplingRate(trace.samplingRate, 1),
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

function appendAttemptLogAttempt(logEntries, entry) {
  if (!Array.isArray(logEntries)) return;
  logEntries.push({
    source: entry.source,
    status: entry.status,
    code: entry.code ?? null,
    durationMs: Number.isFinite(entry.durationMs) ? entry.durationMs : null,
    writeDurationMs: Number.isFinite(entry.writeDurationMs) ? entry.writeDurationMs : null,
  });
}

async function emitDeliveryMetricsHook(hook, payload) {
  if (typeof hook !== 'function') {
    return;
  }
  try {
    await hook(payload);
  } catch (error) {
    log.warn(
      'Distribution',
      `delivery metrics hook failed: ${error?.message || String(error)}`
    );
  }
}

function parseDownloadOptions(options = {}) {
  return {
    store: options.store,
    algorithm: options.algorithm,
    onProgress: options.onProgress ?? null,
    onDeliveryMetrics: options.onDeliveryMetrics ?? null,
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

async function executeDeliveryPlan(
  baseUrl,
  shardIndex,
  shardInfo,
  plan,
  p2p,
  options,
  trace,
  decisionTraceConfig,
  sourceMatrix,
  attemptLog
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
      appendAttemptLogAttempt(attemptLog, {
        source: step.source,
        status: 'skipped',
      });
      continue;
    }

    const attemptStart = performance.now();
    try {
      let result = null;
      if (step.source === DISTRIBUTION_SOURCE_CACHE) {
        if (!(await (options.store?.shardExists ?? shardExists)(shardIndex))) {
          const cacheMiss = new Error(`Shard ${shardIndex} missing from local cache`);
          cacheMiss.code = 'cache_miss';
          throw cacheMiss;
        }
        const buffer = await (options.store?.loadShard ?? loadShardFromStore)(shardIndex, { verify: false });
        result = {
          buffer,
          bytes: buffer.byteLength,
          hash: await computeHash(buffer, options.algorithm),
          wrote: false,
          source: DISTRIBUTION_SOURCE_CACHE,
          path: 'cache',
          manifestVersionSet: options.expectedManifestVersionSet ?? null,
          writeDurationMs: null,
        };
      } else if (step.source === DISTRIBUTION_SOURCE_P2P) {
        result = await p2p.download(shardIndex, shardInfo, options);
        if (!result.hash) {
          if (!(result.buffer instanceof ArrayBuffer)) {
            throw new Error(`Shard ${shardIndex} p2p result missing hash and buffer.`);
          }
          result.hash = await computeHash(result.buffer, options.algorithm);
        }
      } else if (step.source === DISTRIBUTION_SOURCE_HTTP) {
        const download = options.store
          ? createHttpShardTransport(options.store).downloadShardFromHttp : downloadShardFromHttp;
        result = await download(baseUrl, shardInfo, shardIndex, { ...options });
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
      appendAttemptLogAttempt(attemptLog, {
        source: step.source,
        status: 'success',
        durationMs: performance.now() - attemptStart,
        writeDurationMs: result.writeDurationMs,
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
      appendAttemptLogAttempt(attemptLog, {
        source: step.source,
        status: 'failed',
        code: error?.code || null,
        durationMs: performance.now() - attemptStart,
      });
      const enabledIndex = enabledSources.findIndex((entry) => entry.source === step.source);
      const isLastEnabled = enabledIndex === enabledSources.length - 1;
      const transitionType = (
        error?.code === 'cache_miss'
        || error?.code === 'p2p_unconfigured'
        || p2p.isMiss?.(error) === true
      )
        ? 'onMiss'
        : 'onFailure';
      const transition = sourceMatrix?.[step.source]?.[transitionType] || 'next';
      if (isLastEnabled || transition === 'terminal') {
        log.warn('Distribution', `All shard delivery sources failed for shard ${shardIndex}: ${error.message}`);
        throw error;
      }
      log.debug('Distribution', `Shard ${shardIndex} source "${step.source}" failed (${error.code || 'error'}): ${error.message}`);
      continue;
    }
  }

  throw lastError || new Error(`No shard delivery source available for shard ${shardIndex}`);
}

export async function downloadDistributedShard(
  baseUrl,
  shardIndex,
  shardInfo,
  options = {}
) {
  if (options.transport) {
    return options.transport(baseUrl, shardIndex, shardInfo, { ...options, transport: undefined });
  }
  const {
    sourceOrder,
    distributionConfig = getDistributionConfig(),
    distribution = {},
    maxRetries,
    initialRetryDelayMs,
    maxRetryDelayMs,
    requiredEncoding,
    algorithm,
    signal,
    onProgress = null,
    onDeliveryMetrics = null,
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

  const peerConfig = {
    ...activeConfig.p2p,
    transport: activeConfig?.p2p?.transport || p2pTransport,
  };
  if ((peerConfig.enabled || peerConfig.transport) && !options.peerDelivery) {
    throw new Error('Peer distribution requires an explicit host-supplied transport.');
  }
  const p2p = options.peerDelivery?.(peerConfig) ?? { enabled: false, transport: null };

  const downloadOptions = parseDownloadOptions({
    ...options,
    algorithm,
    onProgress,
    onDeliveryMetrics,
    signal,
    requiredEncoding: requiredEncoding ?? activeConfig.requiredContentEncoding ?? null,
    expectedHash:
      options.expectedHash
      ?? getExpectedShardHash(shardInfo, algorithm)
      ?? activeConfig.expectedHash
      ?? null,
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
    && shouldEmitDecisionTrace(
      decisionTraceConfig,
      shardIndex,
      downloadOptions.expectedManifestVersionSet,
      order
    )
    ? createDecisionTrace(
      order,
      planResult.plan,
      shardIndex,
      decisionTraceConfig.deterministic,
      downloadOptions.expectedManifestVersionSet
    )
    : null;

  const dedupeKey = `${deliveryOwnerKey(options, p2p)}:${createDeliveryKey(baseUrl, shardIndex, downloadOptions, order, sourceMatrix)}`;
  if (inFlightDeliveries.has(dedupeKey)) {
    return await awaitWithSignal(
      inFlightDeliveries.get(dedupeKey),
      signal,
      `Shard ${shardIndex} delivery aborted`
    );
  }

  const deliveryPromise = (async () => {
    const deliveryStart = performance.now();
    const attemptLog = [];
    const result = await executeDeliveryPlan(
      baseUrl,
      shardIndex,
      shardInfo,
      planResult.plan,
      p2p,
      downloadOptions,
      trace,
      decisionTraceConfig,
      sourceMatrix,
      attemptLog
    );
    const metrics = createDeliveryMetrics(
      order,
      result,
      attemptLog,
      performance.now() - deliveryStart
    );
    const resultWithMetrics = {
      ...result,
      deliveryMetrics: metrics,
    };
    await emitDeliveryMetricsHook(downloadOptions.onDeliveryMetrics, {
      schemaVersion: DISTRIBUTION_DELIVERY_METRICS_EVENT_SCHEMA_VERSION,
      shardIndex,
      source: result.source ?? null,
      path: result.path ?? null,
      expectedManifestVersionSet: downloadOptions.expectedManifestVersionSet ?? null,
      deliveryMetrics: metrics,
      decisionTrace: trace ?? null,
    });
    return attachDecisionTrace(resultWithMetrics, trace);
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
