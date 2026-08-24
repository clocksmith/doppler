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

export function normalizeContentEncodings(value) {
  if (!value) return [];
  return value
    .split(',')
    .map((entry) => entry.trim().toLowerCase())
    .filter(Boolean);
}

export function parseContentLengthHeader(response, shardIndex) {
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

export function parseContentRangeHeader(response, shardIndex) {
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

export function assertHttpResponseBoundaryHeaders(response, shardIndex, contentLength, contentRange) {
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

export function assertHttpResumeAlignment(
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

export function assertHttpPayloadBoundary(shardIndex, bytesReceived, contentLength, contentRange, expectedSize) {
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

export function assertRequiredContentEncoding(response, requiredEncoding, context) {
  if (!requiredEncoding) return;
  const required = requiredEncoding.trim().toLowerCase();
  if (!required) return;
  const found = normalizeContentEncodings(response.headers.get('content-encoding'));
  if (!found.includes(required)) {
    const foundValue = found.length > 0 ? found.join(', ') : 'none';
    throw new Error(`Missing required content-encoding "${required}" for ${context} (found: ${foundValue})`);
  }
}

export function buildShardUrl(baseUrl, shardInfo) {
  const base = String(baseUrl || '').replace(/\/$/, '');
  const filename = String(shardInfo?.filename || '').replace(/^\/+/, '');
  return `${base}/${filename}`;
}

export function bytesToHex(bytes) {
  return Array.from(bytes)
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

export async function seedHasherFromStoredPrefix(hasher, shardIndex, expectedPrefixBytes) {
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

export async function resolvePersistedResumeOffset(writeToStore, shardIndex, expectedSize) {
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

export async function createHttpTransferState(writeToStore, shardIndex, algorithm, resumeOffset = 0) {
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

export async function resetHttpTransferState(state, writeToStore, shardIndex, algorithm) {
  await state.writer?.abort?.();
  state.hasher = await createStreamingHasher(algorithm);
  state.chunks = writeToStore ? null : [];
  state.writer = writeToStore ? await createShardWriter(shardIndex) : null;
  state.writerClosed = false;
  state.receivedBytes = 0;
  state.writeDurationMs = 0;
}

export async function appendHttpTransferChunk(state, chunk) {
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

export function hasCompleteExpectedHttpTransfer(state, expectedSize) {
  if (!Number.isFinite(expectedSize)) {
    return false;
  }
  const expected = Math.max(0, Math.floor(expectedSize));
  return Number.isInteger(state?.receivedBytes) && state.receivedBytes === expected;
}

export function createHttpStreamReadError(
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

export async function finalizeHttpTransferState(state, startTime, shardIndex) {
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

export async function finalizeHttpTransferStateAtRejectedEof(
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

export async function abortHttpTransferState(state) {
  if (state.writer && !state.writerClosed) {
    await state.writer.abort?.();
    state.writerClosed = true;
  }
}

export async function persistHttpTransferState(state) {
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

export async function clearPersistedShardState(shardIndex) {
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

export async function recoverHttpRejectedResumeRange(
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

export async function downloadShardFromHttp(baseUrl, shardInfo, shardIndex, options = {}) {
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
