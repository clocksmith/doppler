import { log } from '../../../debug/index.js';
import { getExpectedShardHash } from '../../../formats/rdrr/index.js';
import {
  computeHash,
  createStreamingHasher,
  createShardWriter,
  deleteShard,
  getShardStoredSize,
  loadShard as loadShardFromStore,
  shardExists,
  streamShardRange,
} from '../../../storage/shard-manager.js';
import { ERROR_CODES, createDopplerError } from '../../../errors/doppler-error.js';
import { DEFAULT_DISTRIBUTION_CONFIG } from '../../../config/schema/distribution.schema.js';
import {
  P2P_TRANSPORT_CONTRACT_VERSION,
  P2P_TRANSPORT_ERROR_CODES,
  assertSupportedP2PTransportContract,
  createP2PTransportError,
  normalizeP2PTransportError,
  normalizeP2PTransportResult,
  isP2PTransportRetryable,
} from '../p2p-transport-contract.js';
import {
  normalizeP2PControlPlaneConfig,
  resolveP2PSessionToken,
  evaluateP2PPolicyDecision,
} from '../p2p-control-plane.js';
import { createBrowserWebRTCDataPlaneTransport } from '../p2p-webrtc-browser.js';

export const DISTRIBUTION_SOURCE_CACHE = 'cache';

export const DISTRIBUTION_SOURCE_P2P = 'p2p';

import { DISTRIBUTION_SOURCE_HTTP } from '../../../storage/download/http-contract.js';
export { DISTRIBUTION_SOURCE_HTTP };

export const DISTRIBUTION_DELIVERY_METRICS_SCHEMA_VERSION = 1;

export const DISTRIBUTION_SOURCES = Object.freeze(
  [...DEFAULT_DISTRIBUTION_CONFIG.sourceOrder]
);

export const DEFAULT_P2P_TIMEOUT_MS = DEFAULT_DISTRIBUTION_CONFIG.p2p.timeoutMs;

export const DEFAULT_P2P_MAX_RETRIES = DEFAULT_DISTRIBUTION_CONFIG.p2p.maxRetries;

export const DEFAULT_P2P_RETRY_DELAY_MS = DEFAULT_DISTRIBUTION_CONFIG.p2p.retryDelayMs;

export const DEFAULT_P2P_RATE_LIMIT_PER_MINUTE = DEFAULT_DISTRIBUTION_CONFIG.p2p.abuse.rateLimitPerMinute;

export const DEFAULT_P2P_MAX_CONSECUTIVE_FAILURES = DEFAULT_DISTRIBUTION_CONFIG.p2p.abuse.maxConsecutiveFailures;

export const DEFAULT_P2P_QUARANTINE_MS = DEFAULT_DISTRIBUTION_CONFIG.p2p.abuse.quarantineMs;

export const DEFAULT_P2P_CONTROL_PLANE_TOKEN_REFRESH_SKEW_MS = DEFAULT_DISTRIBUTION_CONFIG.p2p.controlPlane.tokenRefreshSkewMs;

import { normalizeRequiredInteger, createShardSizeMismatchError } from '../../../storage/download/http-contract.js';
export { normalizeRequiredInteger, createShardSizeMismatchError };

export function normalizeOptionalToken(value) {
  if (value === undefined || value === null) {
    return null;
  }
  const normalized = String(value).trim();
  return normalized || null;
}

export function normalizeOptionalTimestamp(value) {
  if (value === undefined || value === null) {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null;
  }
  return Math.floor(parsed);
}


export function assertP2PPayloadBoundary(
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

export function normalizeP2PConfig(config = {}) {
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

export function createP2PPolicyDeniedError(message, details = {}) {
  return createP2PTransportError(
    P2P_TRANSPORT_ERROR_CODES.policyDenied,
    message,
    details,
    false
  );
}

export function enforceP2PSecurityAndAbusePolicy(p2pConfig, state, shardIndex, nowMs = Date.now()) {
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



import { createSourceCounter, createLatencySummary, createDeliveryMetrics } from '../../../storage/download/http-contract.js';
export { createSourceCounter, createLatencySummary, createDeliveryMetrics };
