import { isPlainObject } from '../utils/plain-object.js';

export const TOOLING_ENVELOPE_SCHEMA_VERSION = 1;
export const TOOLING_ERROR_CODE_FALLBACK = 'tooling_error';

const CONTEXT_DETAIL_KEYS = Object.freeze([
  'command',
  'suite',
  'workloadType',
  'modelId',
  'requestedSuite',
  'allowedSuites',
  'fromSurface',
  'toSurface',
  'surface',
]);

function asNonEmptyString(value) {
  if (value == null) return null;
  const normalized = String(value).trim();
  return normalized.length > 0 ? normalized : null;
}

function mergeDetailRecords(...sources) {
  const merged = {};
  for (const source of sources) {
    if (!isPlainObject(source)) continue;
    for (const [key, value] of Object.entries(source)) {
      if (value === undefined) continue;
      merged[key] = value;
    }
  }
  return Object.keys(merged).length > 0 ? merged : null;
}

function pickKnownDetails(source) {
  if (!isPlainObject(source)) return null;
  const picked = {};
  for (const key of CONTEXT_DETAIL_KEYS) {
    if (source[key] !== undefined) {
      picked[key] = source[key];
    }
  }
  return Object.keys(picked).length > 0 ? picked : null;
}

function pickContextDetails(context) {
  if (!isPlainObject(context)) return null;
  const requestDetails = pickKnownDetails(context.request);
  const surface = asNonEmptyString(context.surface);
  return mergeDetailRecords(
    surface ? { surface } : null,
    requestDetails
  );
}

function resolveErrorMessage(error) {
  if (error && typeof error.message === 'string' && error.message.trim().length > 0) {
    return error.message;
  }
  return String(error || 'Unknown tooling command error');
}

function resolveErrorCode(error) {
  const explicit = asNonEmptyString(error?.code);
  if (explicit) return explicit;
  if (error?.name === 'AbortError') return 'aborted';
  return TOOLING_ERROR_CODE_FALLBACK;
}

function resolveRetryable(error) {
  if (typeof error?.retryable === 'boolean') {
    return error.retryable;
  }
  const code = asNonEmptyString(error?.code);
  if (code === 'ETIMEDOUT' || code === 'ECONNRESET' || code === 'EAI_AGAIN') {
    return true;
  }
  return null;
}

export class ToolingCommandError extends Error {
  constructor(message, options = {}) {
    const text = asNonEmptyString(message) || 'Unknown tooling command error';
    super(text, options.cause !== undefined ? { cause: options.cause } : undefined);
    this.name = 'ToolingCommandError';
    this.code = asNonEmptyString(options.code) || TOOLING_ERROR_CODE_FALLBACK;
    this.details = isPlainObject(options.details) ? options.details : null;
    this.retryable = typeof options.retryable === 'boolean' ? options.retryable : null;
  }
}

export function normalizeToToolingCommandError(error, context = {}) {
  if (error instanceof ToolingCommandError) {
    const contextDetails = pickContextDetails(context);
    if (!error.details && contextDetails) {
      error.details = contextDetails;
    } else if (error.details && contextDetails) {
      error.details = mergeDetailRecords(error.details, contextDetails);
    }
    if (error.retryable == null) {
      error.retryable = resolveRetryable(error);
    }
    return error;
  }

  const cause = error instanceof Error ? error : undefined;
  const details = mergeDetailRecords(
    isPlainObject(error?.details) ? error.details : null,
    pickKnownDetails(error),
    pickContextDetails(context)
  );

  const normalized = new ToolingCommandError(resolveErrorMessage(error), {
    cause,
    code: resolveErrorCode(error),
    details,
    retryable: resolveRetryable(error),
  });
  return normalized;
}

export function createToolingSuccessEnvelope({
  surface,
  request,
  result,
  meta = null,
}) {
  const normalizedSurface = asNonEmptyString(surface);
  if (!normalizedSurface) {
    throw new Error('tooling envelope: surface is required for success responses.');
  }
  if (!isPlainObject(request)) {
    throw new Error('tooling envelope: request must be an object for success responses.');
  }

  return {
    ok: true,
    schemaVersion: TOOLING_ENVELOPE_SCHEMA_VERSION,
    surface: normalizedSurface,
    request,
    result,
    ...(meta == null ? {} : { meta }),
  };
}

export function createToolingErrorEnvelope(error, context = {}) {
  const normalized = normalizeToToolingCommandError(error, context);
  const surface = asNonEmptyString(context?.surface);
  const request = isPlainObject(context?.request) ? context.request : null;
  return {
    ok: false,
    schemaVersion: TOOLING_ENVELOPE_SCHEMA_VERSION,
    surface,
    request,
    error: {
      code: normalized.code,
      message: normalized.message,
      details: normalized.details ?? null,
      retryable: normalized.retryable ?? null,
    },
  };
}

export function isToolingSuccessEnvelope(value) {
  if (!isPlainObject(value)) return false;
  return (
    value.ok === true
    && Number.isInteger(value.schemaVersion)
    && value.schemaVersion === TOOLING_ENVELOPE_SCHEMA_VERSION
    && typeof value.surface === 'string'
    && isPlainObject(value.request)
    && Object.prototype.hasOwnProperty.call(value, 'result')
  );
}

export function isToolingErrorEnvelope(value) {
  if (!isPlainObject(value)) return false;
  if (value.ok !== false) return false;
  if (!Number.isInteger(value.schemaVersion) || value.schemaVersion !== TOOLING_ENVELOPE_SCHEMA_VERSION) {
    return false;
  }
  if (value.surface !== null && typeof value.surface !== 'string') {
    return false;
  }
  if (value.request !== null && !isPlainObject(value.request)) {
    return false;
  }
  if (!isPlainObject(value.error)) return false;
  if (typeof value.error.code !== 'string' || !value.error.code.trim()) return false;
  if (typeof value.error.message !== 'string' || !value.error.message.trim()) return false;
  if (value.error.details !== null && !isPlainObject(value.error.details)) return false;
  if (value.error.retryable !== null && typeof value.error.retryable !== 'boolean') return false;
  return true;
}
