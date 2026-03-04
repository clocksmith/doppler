import { ERROR_CODES, createDopplerError } from '../errors/doppler-error.js';

export const P2P_TRANSPORT_CONTRACT_VERSION = 1;
export const P2P_TRANSPORT_RESULT_SCHEMA_VERSION = 1;

export const P2P_TRANSPORT_ERROR_CODES = Object.freeze({
  unconfigured: ERROR_CODES.DISTRIBUTION_P2P_TRANSPORT_UNCONFIGURED,
  unavailable: ERROR_CODES.DISTRIBUTION_P2P_TRANSPORT_UNAVAILABLE,
  timeout: ERROR_CODES.DISTRIBUTION_P2P_TRANSPORT_TIMEOUT,
  aborted: ERROR_CODES.DISTRIBUTION_P2P_TRANSPORT_ABORTED,
  integrityMismatch: ERROR_CODES.DISTRIBUTION_P2P_TRANSPORT_INTEGRITY_MISMATCH,
  policyDenied: ERROR_CODES.DISTRIBUTION_P2P_TRANSPORT_POLICY_DENIED,
  internal: ERROR_CODES.DISTRIBUTION_P2P_TRANSPORT_INTERNAL,
  payloadInvalid: ERROR_CODES.DISTRIBUTION_P2P_TRANSPORT_PAYLOAD_INVALID,
  contractUnsupported: ERROR_CODES.DISTRIBUTION_P2P_CONTRACT_UNSUPPORTED,
});

const RETRYABLE_CODES = new Set([
  P2P_TRANSPORT_ERROR_CODES.timeout,
  P2P_TRANSPORT_ERROR_CODES.internal,
]);

function toArrayBuffer(value, label = 'payload') {
  if (value instanceof ArrayBuffer) return value;
  if (value instanceof Uint8Array) {
    return value.byteOffset === 0 && value.byteLength === value.buffer.byteLength
      ? value.buffer
      : value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  }
  throw createP2PTransportError(
    P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
    `${label} must return ArrayBuffer or Uint8Array.`,
    { label }
  );
}

function normalizeOptionalString(value, label) {
  if (value === undefined || value === null) return null;
  if (typeof value !== 'string') {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label} must be a string when provided.`,
      { label }
    );
  }
  const trimmed = value.trim();
  return trimmed || null;
}

function normalizeOptionalNonNegativeInteger(value, label) {
  if (value === undefined || value === null) return null;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label} must be a non-negative integer when provided.`,
      { label }
    );
  }
  return parsed;
}

function normalizeTransportErrorCode(rawCode) {
  if (typeof rawCode !== 'string') return null;
  const normalized = rawCode.trim().toLowerCase().replace(/[\s_-]+/g, '');
  if (!normalized) return null;
  if (normalized === 'timeout' || normalized === 'timedout') {
    return P2P_TRANSPORT_ERROR_CODES.timeout;
  }
  if (
    normalized === 'unavailable'
    || normalized === 'notfound'
    || normalized === 'miss'
    || normalized === 'peermiss'
  ) {
    return P2P_TRANSPORT_ERROR_CODES.unavailable;
  }
  if (normalized === 'aborted' || normalized === 'abort' || normalized === 'cancelled') {
    return P2P_TRANSPORT_ERROR_CODES.aborted;
  }
  if (
    normalized === 'integritymismatch'
    || normalized === 'hashmismatch'
    || normalized === 'checksummismatch'
  ) {
    return P2P_TRANSPORT_ERROR_CODES.integrityMismatch;
  }
  if (normalized === 'policydenied' || normalized === 'forbidden' || normalized === 'denied') {
    return P2P_TRANSPORT_ERROR_CODES.policyDenied;
  }
  if (normalized === 'payloadinvalid' || normalized === 'invalidpayload' || normalized === 'badpayload') {
    return P2P_TRANSPORT_ERROR_CODES.payloadInvalid;
  }
  if (normalized === 'internal' || normalized === 'error' || normalized === 'internalerror') {
    return P2P_TRANSPORT_ERROR_CODES.internal;
  }
  if (normalized === 'unconfigured') {
    return P2P_TRANSPORT_ERROR_CODES.unconfigured;
  }
  if (normalized === 'contractunsupported') {
    return P2P_TRANSPORT_ERROR_CODES.contractUnsupported;
  }
  return null;
}

function normalizeTransportErrorMessage(message) {
  if (typeof message !== 'string') return null;
  const normalized = message.trim().toLowerCase().replace(/[\s_-]+/g, '');
  if (!normalized) return null;
  if (
    normalized.includes('timeout')
    || normalized.includes('timedout')
    || normalized.includes('deadlineexceeded')
  ) {
    return P2P_TRANSPORT_ERROR_CODES.timeout;
  }
  if (
    normalized.includes('peermiss')
    || normalized.includes('notfound')
    || normalized.includes('unavailable')
    || normalized.includes('nomatch')
    || normalized.includes('noprovider')
  ) {
    return P2P_TRANSPORT_ERROR_CODES.unavailable;
  }
  if (
    normalized.includes('abort')
    || normalized.includes('cancelled')
    || normalized.includes('canceled')
  ) {
    return P2P_TRANSPORT_ERROR_CODES.aborted;
  }
  if (
    normalized.includes('hashmismatch')
    || normalized.includes('checksummismatch')
    || normalized.includes('integrity')
  ) {
    return P2P_TRANSPORT_ERROR_CODES.integrityMismatch;
  }
  if (
    normalized.includes('forbidden')
    || normalized.includes('denied')
    || normalized.includes('unauthorized')
    || normalized.includes('policy')
  ) {
    return P2P_TRANSPORT_ERROR_CODES.policyDenied;
  }
  if (
    normalized.includes('badpayload')
    || normalized.includes('payloadinvalid')
    || normalized.includes('invalidpayload')
  ) {
    return P2P_TRANSPORT_ERROR_CODES.payloadInvalid;
  }
  return null;
}

export function createP2PTransportError(code, message, details = null, retryable = false) {
  const error = createDopplerError(code, message);
  error.retryable = retryable === true;
  if (details && typeof details === 'object') {
    error.details = details;
  }
  return error;
}

export function assertSupportedP2PTransportContract(version) {
  const parsed = Number(version ?? P2P_TRANSPORT_CONTRACT_VERSION);
  if (!Number.isInteger(parsed) || parsed !== P2P_TRANSPORT_CONTRACT_VERSION) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.contractUnsupported,
      `Unsupported p2p.transport contractVersion "${version}". Supported: ${P2P_TRANSPORT_CONTRACT_VERSION}.`,
      { contractVersion: version }
    );
  }
  return parsed;
}

export function normalizeP2PTransportResult(value, label = 'p2p transport payload') {
  if (value == null) return null;

  if (value instanceof ArrayBuffer || value instanceof Uint8Array) {
    return {
      schemaVersion: P2P_TRANSPORT_RESULT_SCHEMA_VERSION,
      data: toArrayBuffer(value, label),
      manifestVersionSet: null,
      manifestHash: null,
      rangeStart: null,
      totalSize: null,
    };
  }

  if (typeof value === 'object') {
    if (value.miss === true || value.notFound === true) {
      throw createP2PTransportError(
        P2P_TRANSPORT_ERROR_CODES.unavailable,
        `${label} unavailable`,
        { label },
        false
      );
    }

    if (value.error != null) {
      throw normalizeP2PTransportError(value.error, { label });
    }

    const payload = value.data ?? value.buffer ?? null;
    if (payload == null) {
      return null;
    }

    return {
      schemaVersion: P2P_TRANSPORT_RESULT_SCHEMA_VERSION,
      data: toArrayBuffer(payload, `${label}.data`),
      manifestVersionSet: normalizeOptionalString(
        value.manifestVersionSet,
        `${label}.manifestVersionSet`
      ),
      manifestHash: normalizeOptionalString(
        value.manifestHash,
        `${label}.manifestHash`
      ),
      rangeStart: normalizeOptionalNonNegativeInteger(
        value.rangeStart ?? value.offset,
        `${label}.rangeStart`
      ),
      totalSize: normalizeOptionalNonNegativeInteger(
        value.totalSize,
        `${label}.totalSize`
      ),
    };
  }

  return null;
}

export function normalizeP2PTransportPayload(value, label = 'p2p transport payload') {
  const result = normalizeP2PTransportResult(value, label);
  return result ? result.data : null;
}

export function normalizeP2PTransportError(error, context = {}) {
  if (error?.code && String(error.code).startsWith('DOPPLER_')) {
    if (RETRYABLE_CODES.has(error.code)) {
      error.retryable = true;
    }
    return error;
  }

  const mappedCode = normalizeTransportErrorCode(error?.code);
  if (mappedCode) {
    return createP2PTransportError(
      mappedCode,
      String(error?.message || 'P2P transport failed'),
      {
        ...context,
        causeName: error?.name || null,
      },
      RETRYABLE_CODES.has(mappedCode)
    );
  }

  const messageCode = normalizeTransportErrorMessage(error?.message);
  if (messageCode) {
    return createP2PTransportError(
      messageCode,
      String(error?.message || 'P2P transport failed'),
      {
        ...context,
        causeName: error?.name || null,
      },
      RETRYABLE_CODES.has(messageCode)
    );
  }

  if (error?.name === 'AbortError') {
    return createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.aborted,
      context.label ? `${context.label} aborted` : 'P2P transport aborted',
      context,
      false
    );
  }

  if (error?.name === 'TimeoutError') {
    return createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.timeout,
      context.label ? `${context.label} timed out` : 'P2P transport timed out',
      context,
      true
    );
  }

  const message = String(error?.message || 'P2P transport failed');
  return createP2PTransportError(
    P2P_TRANSPORT_ERROR_CODES.internal,
    message,
    {
      ...context,
      causeName: error?.name || null,
    },
    true
  );
}

export function isP2PTransportRetryable(error) {
  if (!error) return false;
  if (error.retryable === true) return true;
  return RETRYABLE_CODES.has(error.code);
}
