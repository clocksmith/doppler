import { log } from '../../debug/index.js';
import { getExpectedShardHash } from '../../formats/rdrr/index.js';
import {
  computeHash,
  createStreamingHasher,
  createShardWriter,
  deleteShard,
  getShardStoredSize,
  loadShard as loadShardFromStore,
  shardExists,
  streamShardRange,
} from '../../storage/shard-manager.js';
import { ERROR_CODES, createDopplerError } from '../../errors/doppler-error.js';
import { DEFAULT_DISTRIBUTION_CONFIG } from '../../config/schema/distribution.schema.js';
import {
  P2P_TRANSPORT_CONTRACT_VERSION,
  P2P_TRANSPORT_ERROR_CODES,
  assertSupportedP2PTransportContract,
  createP2PTransportError,
  normalizeP2PTransportError,
  normalizeP2PTransportResult,
  isP2PTransportRetryable,
} from './p2p-transport-contract.js';
import {
  normalizeP2PControlPlaneConfig,
  resolveP2PSessionToken,
  evaluateP2PPolicyDecision,
} from './p2p-control-plane.js';
import { createBrowserWebRTCDataPlaneTransport } from './p2p-webrtc-browser.js';

const DISTRIBUTION_SOURCE_CACHE = 'cache';
const DISTRIBUTION_SOURCE_P2P = 'p2p';
const DISTRIBUTION_SOURCE_HTTP = 'http';
const DISTRIBUTION_DECISION_TRACE_SCHEMA_VERSION = 1;
const DISTRIBUTION_DELIVERY_METRICS_SCHEMA_VERSION = 1;
const DISTRIBUTION_DELIVERY_METRICS_EVENT_SCHEMA_VERSION = 1;

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
const DEFAULT_P2P_RATE_LIMIT_PER_MINUTE = DEFAULT_DISTRIBUTION_CONFIG.p2p.abuse.rateLimitPerMinute;
const DEFAULT_P2P_MAX_CONSECUTIVE_FAILURES = DEFAULT_DISTRIBUTION_CONFIG.p2p.abuse.maxConsecutiveFailures;
const DEFAULT_P2P_QUARANTINE_MS = DEFAULT_DISTRIBUTION_CONFIG.p2p.abuse.quarantineMs;
const DEFAULT_P2P_CONTROL_PLANE_TOKEN_REFRESH_SKEW_MS = DEFAULT_DISTRIBUTION_CONFIG.p2p.controlPlane.tokenRefreshSkewMs;

const inFlightDeliveries = new Map();
const p2pTransportPolicyState = new WeakMap();

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

function normalizeRequiredInteger(value, label, { allowZero = false, fallback = null } = {}) {
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

function normalizeOptionalToken(value) {
  if (value === undefined || value === null) {
    return null;
  }
  const normalized = String(value).trim();
  return normalized || null;
}

function normalizeOptionalTimestamp(value) {
  if (value === undefined || value === null) {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null;
  }
  return Math.floor(parsed);
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

function assertP2PPayloadBoundary(
  shardIndex,
  rangeStart,
  payloadBytes,
  totalSize,
  writeToStore
) {
  if (totalSize == null) {
    return;
  }
  if (!Number.isInteger(rangeStart) || rangeStart < 0) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} p2p payload rangeStart must be a non-negative integer for boundary checks.`,
      {
        code: 'p2p_payload_range_start_invalid',
        rangeStart,
      }
    );
  }
  if (!Number.isInteger(payloadBytes) || payloadBytes < 0) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} p2p payload size must be a non-negative integer for boundary checks.`,
      {
        code: 'p2p_payload_size_invalid',
        payloadBytes,
      }
    );
  }
  if (rangeStart + payloadBytes > totalSize) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} p2p payload exceeds total size: start=${rangeStart}, bytes=${payloadBytes}, total=${totalSize}.`,
      {
        code: 'p2p_payload_exceeds_total',
        rangeStart,
        payloadBytes,
        totalSize,
      }
    );
  }
  if (!writeToStore && rangeStart === 0 && payloadBytes !== totalSize) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} p2p payload size mismatch: expected ${totalSize}, got ${payloadBytes}.`,
      {
        code: 'p2p_payload_size_mismatch',
        payloadBytes,
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
  const rawSecurity = config?.security && typeof config.security === 'object'
    ? config.security
    : {};
  const rawAbuse = config?.abuse && typeof config.abuse === 'object'
    ? config.abuse
    : {};
  const rawControlPlane = config?.controlPlane && typeof config.controlPlane === 'object'
    ? config.controlPlane
    : {};
  const rawWebRTC = config?.webrtc && typeof config.webrtc === 'object'
    ? config.webrtc
    : {};

  let transport = config?.transport;
  if (typeof transport !== 'function' && rawWebRTC.enabled === true) {
    transport = createBrowserWebRTCDataPlaneTransport(rawWebRTC);
  }
  if (typeof transport !== 'function') {
    transport = null;
  }

  const contractVersion = assertSupportedP2PTransportContract(
    config?.contractVersion ?? P2P_TRANSPORT_CONTRACT_VERSION
  );

  return {
    enabled,
    timeoutMs: normalizeRequiredInteger(
      rawTimeoutMs,
      'distribution.p2p.timeoutMs',
      { fallback: DEFAULT_P2P_TIMEOUT_MS }
    ),
    maxRetries: normalizeRequiredInteger(
      rawMaxRetries,
      'distribution.p2p.maxRetries',
      { allowZero: true, fallback: DEFAULT_P2P_MAX_RETRIES }
    ),
    retryDelayMs: normalizeRequiredInteger(
      rawRetryDelayMs,
      'distribution.p2p.retryDelayMs',
      { allowZero: true, fallback: DEFAULT_P2P_RETRY_DELAY_MS }
    ),
    transport,
    contractVersion,
    controlPlane: normalizeP2PControlPlaneConfig({
      ...DEFAULT_DISTRIBUTION_CONFIG.p2p.controlPlane,
      ...rawControlPlane,
      tokenRefreshSkewMs: rawControlPlane.tokenRefreshSkewMs
        ?? DEFAULT_P2P_CONTROL_PLANE_TOKEN_REFRESH_SKEW_MS,
    }),
    security: {
      requireSessionToken: rawSecurity.requireSessionToken === true,
      sessionToken: normalizeOptionalToken(rawSecurity.sessionToken),
      tokenExpiresAtMs: normalizeOptionalTimestamp(rawSecurity.tokenExpiresAtMs),
    },
    abuse: {
      rateLimitPerMinute: normalizeRequiredInteger(
        rawAbuse.rateLimitPerMinute,
        'distribution.p2p.abuse.rateLimitPerMinute',
        { allowZero: true, fallback: DEFAULT_P2P_RATE_LIMIT_PER_MINUTE }
      ),
      maxConsecutiveFailures: normalizeRequiredInteger(
        rawAbuse.maxConsecutiveFailures,
        'distribution.p2p.abuse.maxConsecutiveFailures',
        { fallback: DEFAULT_P2P_MAX_CONSECUTIVE_FAILURES }
      ),
      quarantineMs: normalizeRequiredInteger(
        rawAbuse.quarantineMs,
        'distribution.p2p.abuse.quarantineMs',
        { allowZero: true, fallback: DEFAULT_P2P_QUARANTINE_MS }
      ),
    },
  };
}

function getP2PTransportPolicyState(transport) {
  if (typeof transport !== 'function') {
    return null;
  }
  let state = p2pTransportPolicyState.get(transport);
  if (!state) {
    state = {
      requestTimestamps: [],
      consecutiveFailures: 0,
      quarantinedUntilMs: 0,
    };
    p2pTransportPolicyState.set(transport, state);
  }
  return state;
}

function createP2PPolicyDeniedError(message, details = {}) {
  return createP2PTransportError(
    P2P_TRANSPORT_ERROR_CODES.policyDenied,
    message,
    details,
    false
  );
}

function isSessionTokenExpiredOrExpiring(tokenExpiresAtMs, nowMs = Date.now(), skewMs = 0) {
  if (!Number.isFinite(tokenExpiresAtMs)) {
    return false;
  }
  const threshold = nowMs + Math.max(0, Math.floor(skewMs));
  return threshold >= tokenExpiresAtMs;
}

function applyControlPlaneSessionUpdate(p2pConfig, sessionUpdate) {
  if (!sessionUpdate || !p2pConfig?.security) {
    return;
  }
  if (sessionUpdate.hasSessionToken === true) {
    p2pConfig.security.sessionToken = normalizeOptionalToken(sessionUpdate.sessionToken);
  }
  if (sessionUpdate.hasTokenExpiresAtMs === true) {
    p2pConfig.security.tokenExpiresAtMs = normalizeOptionalTimestamp(sessionUpdate.tokenExpiresAtMs);
  }
}

async function refreshP2PSessionTokenFromControlPlane(p2pConfig, context, nowMs = Date.now()) {
  const controlPlane = p2pConfig?.controlPlane;
  if (!controlPlane?.enabled || typeof controlPlane.tokenProvider !== 'function') {
    return;
  }

  const requiresSessionToken = p2pConfig?.security?.requireSessionToken === true;
  const token = p2pConfig?.security?.sessionToken ?? null;
  const tokenExpiresAtMs = p2pConfig?.security?.tokenExpiresAtMs ?? null;
  let reason = null;

  if (requiresSessionToken && !token) {
    reason = 'missing';
  } else if (isSessionTokenExpiredOrExpiring(tokenExpiresAtMs, nowMs, 0)) {
    reason = 'expired';
  } else if (
    isSessionTokenExpiredOrExpiring(tokenExpiresAtMs, nowMs, controlPlane.tokenRefreshSkewMs)
  ) {
    reason = 'refresh';
  }

  if (!reason) {
    return;
  }

  const sessionUpdate = await resolveP2PSessionToken(controlPlane, {
    ...context,
    reason,
    nowMs,
    currentSessionToken: token,
    currentTokenExpiresAtMs: tokenExpiresAtMs,
  });
  applyControlPlaneSessionUpdate(p2pConfig, sessionUpdate);

  if (requiresSessionToken && !p2pConfig.security.sessionToken) {
    throw createP2PPolicyDeniedError(
      `P2P shard ${context?.shardIndex} requires a session token from control plane.`,
      {
        shardIndex: context?.shardIndex ?? null,
        policyReason: 'session_token_missing_after_refresh',
      }
    );
  }
}

async function enforceP2PControlPlanePolicy(p2pConfig, context, nowMs = Date.now()) {
  const controlPlane = p2pConfig?.controlPlane;
  if (!controlPlane?.enabled || typeof controlPlane.policyEvaluator !== 'function') {
    return;
  }
  const decision = await evaluateP2PPolicyDecision(controlPlane, {
    ...context,
    nowMs,
    currentSessionToken: p2pConfig?.security?.sessionToken ?? null,
    currentTokenExpiresAtMs: p2pConfig?.security?.tokenExpiresAtMs ?? null,
  });
  applyControlPlaneSessionUpdate(p2pConfig, decision.sessionUpdate);
  if (decision.allow !== false) {
    return;
  }
  throw createP2PPolicyDeniedError(
    `P2P shard ${context?.shardIndex} denied by control-plane policy.`,
    {
      shardIndex: context?.shardIndex ?? null,
      policyReason: decision.reason ?? 'policy_denied_control_plane',
      controlPlaneMetadata: decision.metadata ?? null,
    }
  );
}

function enforceP2PSecurityAndAbusePolicy(p2pConfig, state, shardIndex, nowMs = Date.now()) {
  const security = p2pConfig?.security ?? {};
  const abuse = p2pConfig?.abuse ?? {};

  if (security.requireSessionToken === true && !security.sessionToken) {
    throw createP2PPolicyDeniedError(
      `P2P shard ${shardIndex} requires a session token.`,
      {
        shardIndex,
        policyReason: 'session_token_missing',
      }
    );
  }
  if (
    Number.isFinite(security.tokenExpiresAtMs)
    && nowMs >= security.tokenExpiresAtMs
  ) {
    throw createP2PPolicyDeniedError(
      `P2P shard ${shardIndex} session token expired.`,
      {
        shardIndex,
        policyReason: 'session_token_expired',
        tokenExpiresAtMs: security.tokenExpiresAtMs,
      }
    );
  }

  if (!state) {
    return;
  }

  if (Number.isFinite(state.quarantinedUntilMs) && nowMs < state.quarantinedUntilMs) {
    throw createP2PPolicyDeniedError(
      `P2P shard ${shardIndex} transport is quarantined.`,
      {
        shardIndex,
        policyReason: 'transport_quarantined',
        quarantinedUntilMs: state.quarantinedUntilMs,
      }
    );
  }

  const limit = Number.isFinite(abuse.rateLimitPerMinute)
    ? Math.max(0, Math.floor(abuse.rateLimitPerMinute))
    : 0;
  if (limit > 0) {
    const cutoff = nowMs - 60000;
    state.requestTimestamps = state.requestTimestamps.filter((stamp) => stamp >= cutoff);
    if (state.requestTimestamps.length >= limit) {
      throw createP2PPolicyDeniedError(
        `P2P shard ${shardIndex} transport rate limit exceeded.`,
        {
          shardIndex,
          policyReason: 'rate_limited',
          rateLimitPerMinute: limit,
        }
      );
    }
    state.requestTimestamps.push(nowMs);
  }
}

function markP2PTransportSuccess(state) {
  if (!state) {
    return;
  }
  state.consecutiveFailures = 0;
  state.quarantinedUntilMs = 0;
}

function markP2PTransportFailure(p2pConfig, state, normalizedError, nowMs = Date.now()) {
  if (!state) {
    return;
  }
  if (!normalizedError || normalizedError.code === P2P_TRANSPORT_ERROR_CODES.aborted) {
    return;
  }
  const maxFailures = Number.isFinite(p2pConfig?.abuse?.maxConsecutiveFailures)
    ? Math.max(1, Math.floor(p2pConfig.abuse.maxConsecutiveFailures))
    : DEFAULT_P2P_MAX_CONSECUTIVE_FAILURES;
  const quarantineMs = Number.isFinite(p2pConfig?.abuse?.quarantineMs)
    ? Math.max(0, Math.floor(p2pConfig.abuse.quarantineMs))
    : DEFAULT_P2P_QUARANTINE_MS;
  state.consecutiveFailures += 1;
  if (quarantineMs > 0 && state.consecutiveFailures >= maxFailures) {
    state.quarantinedUntilMs = nowMs + quarantineMs;
  }
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

function createSourceCounter() {
  return {
    cache: 0,
    p2p: 0,
    http: 0,
  };
}

function createLatencySummary(durations) {
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

function createDeliveryMetrics(order, result, attempts, totalDurationMs) {
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
  try {
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
  } catch (error) {
    throw createShardSizeMismatchError(
      `Shard ${shardIndex} stored resume prefix unreadable: ${error.message}`,
      {
        code: 'resume_state_prefix_mismatch',
        expectedPrefixBytes,
        actualPrefixBytes: hashedBytes,
      }
    );
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
      return resumeOffset;
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
    writeDurationMs: 0,
  };
}

async function resetHttpTransferState(state, writeToStore, shardIndex, algorithm) {
  await state.writer?.abort?.();
  state.hasher = await createStreamingHasher(algorithm);
  state.chunks = writeToStore ? null : [];
  state.writer = writeToStore ? await createShardWriter(shardIndex) : null;
  state.writerClosed = false;
  state.receivedBytes = 0;
  state.writeDurationMs = 0;
}

async function appendHttpTransferChunk(state, chunk) {
  const bytes = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
  state.hasher.update(bytes);
  if (state.writer) {
    const writeStart = performance.now();
    await state.writer.write(bytes);
    state.writeDurationMs += performance.now() - writeStart;
  } else if (state.chunks) {
    state.chunks.push(bytes.slice(0));
  }
  state.receivedBytes += bytes.byteLength;
}

function hasCompleteExpectedHttpTransfer(state, expectedSize) {
  if (!Number.isFinite(expectedSize)) {
    return false;
  }
  const expected = Math.max(0, Math.floor(expectedSize));
  return Number.isInteger(state?.receivedBytes) && state.receivedBytes === expected;
}

function createHttpStreamReadError(
  error,
  shardIndex,
  attempt,
  maxRetries,
  requestedRangeHeader,
  receivedBytes,
  expectedSize
) {
  const rangeDetail = requestedRangeHeader ? ` range=${requestedRangeHeader}` : '';
  const expectedDetail = Number.isFinite(expectedSize) ? ` expected=${Math.floor(expectedSize)}` : '';
  const message = error?.message || String(error);
  const wrapped = new Error(
    `Shard ${shardIndex} HTTP stream read failed on attempt ${attempt + 1}/${maxRetries + 1}${rangeDetail} received=${receivedBytes}${expectedDetail}: ${message}`,
    error instanceof Error ? { cause: error } : undefined
  );
  wrapped.name = error?.name || 'Error';
  wrapped.code = 'stream_read_failed';
  wrapped.shardIndex = shardIndex;
  wrapped.attempt = attempt;
  wrapped.maxRetries = maxRetries;
  wrapped.requestedRange = requestedRangeHeader;
  wrapped.receivedBytes = receivedBytes;
  wrapped.expectedSize = Number.isFinite(expectedSize) ? Math.floor(expectedSize) : null;
  return wrapped;
}

async function finalizeHttpTransferState(state, startTime, shardIndex) {
  if (state.finalizedResult) {
    return state.finalizedResult;
  }
  const hashBytes = await state.hasher.finalize();
  const hash = bytesToHex(hashBytes);
  if (state.writer) {
    const closeStart = performance.now();
    await state.writer.close();
    state.writerClosed = true;
    state.writeDurationMs += performance.now() - closeStart;
    const elapsed = (performance.now() - startTime) / 1000;
    const speed = elapsed > 0 ? state.receivedBytes / elapsed : 0;
    const speedDisplay = `${(speed / (1024 * 1024)).toFixed(2)}MB/s`;
    log.verbose(
      'Distribution',
      `Shard ${shardIndex}: http stream (${state.receivedBytes} bytes, ${elapsed.toFixed(2)}s, ${speedDisplay})`
    );
    state.finalizedResult = {
      buffer: null,
      bytes: state.receivedBytes,
      hash,
      wrote: true,
      source: DISTRIBUTION_SOURCE_HTTP,
      path: 'http-stream-store',
      writeDurationMs: state.writeDurationMs,
    };
    return state.finalizedResult;
  }

  const buffer = !state.chunks || state.chunks.length === 0
    ? new ArrayBuffer(0)
    : await new Blob(state.chunks).arrayBuffer();
  state.finalizedResult = {
    buffer,
    bytes: buffer.byteLength,
    hash,
    wrote: false,
    source: DISTRIBUTION_SOURCE_HTTP,
    path: 'http-stream-buffer',
    writeDurationMs: null,
  };
  return state.finalizedResult;
}

async function finalizeHttpTransferStateAtRejectedEof(
  state,
  startTime,
  shardIndex,
  algorithm,
  writeToStore
) {
  try {
    return await finalizeHttpTransferState(state, startTime, shardIndex);
  } catch (error) {
    if (!String(error?.message || '').includes('BLAKE3 finalize called with no chunks')) {
      throw error;
    }
    if (!writeToStore) {
      throw error;
    }
    if (state.writer && !state.writerClosed) {
      const closeStart = performance.now();
      await state.writer.close();
      state.writerClosed = true;
      state.writeDurationMs += performance.now() - closeStart;
    }
    const buffer = await loadShardFromStore(shardIndex, { verify: false });
    return {
      buffer: null,
      bytes: buffer.byteLength,
      hash: await computeHash(buffer, algorithm),
      wrote: true,
      source: DISTRIBUTION_SOURCE_HTTP,
      path: 'http-stream-store',
      writeDurationMs: state.writeDurationMs,
    };
  }
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
    const closeStart = performance.now();
    await state.writer.close();
    state.writerClosed = true;
    state.writeDurationMs += performance.now() - closeStart;
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

async function recoverHttpRejectedResumeRange(
  baseUrl,
  shardInfo,
  shardIndex,
  options,
  transferState,
  writeToStore
) {
  await abortHttpTransferState(transferState);
  if (writeToStore) {
    await clearPersistedShardState(shardIndex);
  }
  return downloadShardFromHttp(baseUrl, shardInfo, shardIndex, {
    ...options,
    __disablePersistedResume: true,
    __resumeRangeRecoveryCount: (options.__resumeRangeRecoveryCount ?? 0) + 1,
  });
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
  const maxRetries = normalizeRequiredInteger(
    options.maxRetries,
    'download.maxRetries',
    { allowZero: true, fallback: 3 }
  );
  const initialRetryDelayMs = normalizeRequiredInteger(
    options.initialRetryDelayMs,
    'download.initialRetryDelayMs',
    { allowZero: true, fallback: 1000 }
  );
  const maxRetryDelayMs = normalizeRequiredInteger(
    options.maxRetryDelayMs,
    'download.maxRetryDelayMs',
    { allowZero: true, fallback: 30000 }
  );
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
    let requestedResumeOffset = 0;
    let requestedRangeHeader = null;
    try {
      requestedResumeOffset = transferState.receivedBytes;
      requestedRangeHeader = requestedResumeOffset > 0
        ? `bytes=${requestedResumeOffset}-`
        : null;
      const requestHeaders = requestedRangeHeader
        ? { range: requestedRangeHeader }
        : undefined;
      const response = await fetch(url, {
        signal,
        headers: requestHeaders,
        cache: 'no-store',
      });
      if (!response.ok) {
        const rangeDetail = requestedRangeHeader ? ` range=${requestedRangeHeader}` : '';
        const error = new Error(
          `Shard ${shardIndex} HTTP ${response.status}: ${response.statusText}${rangeDetail}`
        );
        error.status = response.status;
        error.shardIndex = shardIndex;
        error.requestedRange = requestedRangeHeader;
        throw error;
      }

      assertRequiredContentEncoding(response, requiredEncoding, `shard ${shardIndex}`);
      const contentLength = parseContentLengthHeader(response, shardIndex);
      const contentRange = parseContentRangeHeader(response, shardIndex);
      assertHttpResponseBoundaryHeaders(response, shardIndex, contentLength, contentRange);
      const { resetState } = assertHttpResumeAlignment(
        response,
        shardIndex,
        requestedResumeOffset,
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
        if (hasCompleteExpectedHttpTransfer(transferState, options.expectedSize)) {
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
        }
        throw createHttpStreamReadError(
          error,
          shardIndex,
          attempt,
          maxRetries,
          requestedRangeHeader,
          transferState.receivedBytes,
          options.expectedSize
        );
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

      if (
        error?.status === 416
        && requestedResumeOffset > 0
        && Number.isFinite(options.expectedSize)
        && requestedResumeOffset === Math.floor(options.expectedSize)
      ) {
        const finalized = await finalizeHttpTransferStateAtRejectedEof(
          transferState,
          startTime,
          shardIndex,
          algorithm,
          writeToStore
        );
        return {
          ...finalized,
          manifestVersionSet: options.expectedManifestVersionSet ?? null,
        };
      }

      if (
        error?.status === 416
        && requestedResumeOffset > 0
        && (options.__resumeRangeRecoveryCount ?? 0) <= maxRetries
      ) {
        return recoverHttpRejectedResumeRange(
          baseUrl,
          shardInfo,
          shardIndex,
          options,
          transferState,
          writeToStore
        );
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
  const transportState = getP2PTransportPolicyState(transport);

  const writeToStore = options.writeToStore === true;
  const algorithm = options.algorithm;
  if (writeToStore && !algorithm) {
    throw new Error(`Missing hash algorithm for shard ${shardIndex} p2p transfer.`);
  }

  const expectedSize = Number.isFinite(options.expectedSize)
    ? Math.floor(options.expectedSize)
    : null;
  const disablePersistedResume = options.__disablePersistedResume === true;
  let seededResumeOffset = 0;
  let transferState = null;
  if (writeToStore) {
    if (!disablePersistedResume) {
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
  const startedWithResume = writeToStore && seededResumeOffset > 0;

  const startTime = performance.now();
  let lastError = null;
  const maxRetries = Math.max(0, p2pConfig.maxRetries);
  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    try {
      const requestResumeOffset = transferState?.receivedBytes ?? 0;
      const nowMs = Date.now();
      const attemptContext = {
        shardIndex,
        attempt,
        maxRetries,
        resumeOffset: requestResumeOffset,
        expectedHash: options.expectedHash ?? null,
        expectedSize: options.expectedSize ?? null,
        expectedManifestVersionSet: options.expectedManifestVersionSet ?? null,
      };
      await refreshP2PSessionTokenFromControlPlane(
        p2pConfig,
        attemptContext,
        nowMs
      );
      await enforceP2PControlPlanePolicy(
        p2pConfig,
        attemptContext,
        nowMs
      );
      enforceP2PSecurityAndAbusePolicy(
        p2pConfig,
        transportState,
        shardIndex,
        nowMs
      );
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
      const transferResult = await (async () => {
        if (!writeToStore) {
          assertP2PPayloadRangeStart(shardIndex, payloadRangeStart, 0);
          assertP2PPayloadBoundary(
            shardIndex,
            0,
            payload.data.byteLength,
            payloadTotalSize,
            false
          );
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
            writeDurationMs: null,
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
        assertP2PPayloadBoundary(
          shardIndex,
          effectiveRangeStart,
          payload.data.byteLength,
          payloadTotalSize,
          true
        );
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
      })();
      const result = {
        ...transferResult,
        manifestVersionSet: normalizeManifestVersionSet(
          payload.manifestVersionSet ?? options.expectedManifestVersionSet
        ),
      };
      if (!result.hash && result.buffer instanceof ArrayBuffer) {
        result.hash = await computeHash(result.buffer, options.algorithm);
      }
      if (writeToStore) {
        try {
          assertExpectedManifestVersionSet(
            result.manifestVersionSet,
            options.expectedManifestVersionSet,
            shardIndex,
            DISTRIBUTION_SOURCE_P2P
          );
          if (Number.isFinite(expectedSize)) {
            assertExpectedSize(result.bytes, expectedSize, shardIndex);
          }
          if (options.expectedHash) {
            assertExpectedHash(result.hash, options.expectedHash, shardIndex);
          }
        } catch (verificationError) {
          await clearPersistedShardState(shardIndex);
          if (
            startedWithResume
            && options.__resumeRecoveryAttempted !== true
            && options.expectedHash
            && verificationError?.code === 'hash_mismatch'
          ) {
            return downloadShardFromP2P(shardIndex, shardInfo, p2pConfig, {
              ...options,
              __disablePersistedResume: true,
              __resumeRecoveryAttempted: true,
            });
          }
          throw verificationError;
        }
      }
      markP2PTransportSuccess(transportState);
      return result;
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
      markP2PTransportFailure(
        p2pConfig,
        transportState,
        normalized,
        Date.now()
      );
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
          writeDurationMs: null,
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
      log.debug('Distribution', `Shard ${shardIndex} source "${step.source}" failed (${error.code || 'error'}): ${error.message}`);
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

  const p2p = normalizeP2PConfig({
    ...activeConfig.p2p,
    transport: activeConfig?.p2p?.transport || p2pTransport,
  });

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

  const dedupeKey = createDeliveryKey(baseUrl, shardIndex, downloadOptions, order, sourceMatrix);
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
