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
import { DEFAULT_P2P_MAX_CONSECUTIVE_FAILURES, DEFAULT_P2P_QUARANTINE_MS, DISTRIBUTION_SOURCES, DISTRIBUTION_SOURCE_CACHE, DISTRIBUTION_SOURCE_HTTP, DISTRIBUTION_SOURCE_P2P, assertP2PPayloadBoundary, createDeliveryMetrics, createP2PPolicyDeniedError, createShardSizeMismatchError, enforceP2PSecurityAndAbusePolicy, normalizeOptionalTimestamp, normalizeOptionalToken, normalizeP2PConfig, normalizeRequiredInteger } from './retry.js';
import { appendHttpTransferChunk, clearPersistedShardState, createHttpTransferState, downloadShardFromHttp, finalizeHttpTransferState, persistHttpTransferState, resetHttpTransferState, resolvePersistedResumeOffset } from './transport.js';

export const p2pTransportPolicyState = new WeakMap();

export function normalizeManifestVersionSet(value) {
  if (value === undefined || value === null) return null;
  const normalized = String(value).trim();
  return normalized || null;
}

export function assertP2PPayloadRangeStart(
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

export function assertP2PTotalSize(shardIndex, totalSize, expectedSize) {
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

export function getP2PTransportPolicyState(transport) {
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

export function isSessionTokenExpiredOrExpiring(tokenExpiresAtMs, nowMs = Date.now(), skewMs = 0) {
  if (!Number.isFinite(tokenExpiresAtMs)) {
    return false;
  }
  const threshold = nowMs + Math.max(0, Math.floor(skewMs));
  return threshold >= tokenExpiresAtMs;
}

export function applyControlPlaneSessionUpdate(p2pConfig, sessionUpdate) {
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

export async function refreshP2PSessionTokenFromControlPlane(p2pConfig, context, nowMs = Date.now()) {
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

export async function enforceP2PControlPlanePolicy(p2pConfig, context, nowMs = Date.now()) {
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

export function markP2PTransportSuccess(state) {
  if (!state) {
    return;
  }
  state.consecutiveFailures = 0;
  state.quarantinedUntilMs = 0;
}

export function markP2PTransportFailure(p2pConfig, state, normalizedError, nowMs = Date.now()) {
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

export async function withTimeout(promise, timeoutMs, label = 'operation') {
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

export async function downloadShardFromP2P(shardIndex, shardInfo, p2pConfig, options = {}) {
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
