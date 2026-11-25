import { ERROR_CODES } from '../errors/doppler-error.js';
import { canonicalizeLegacyEnumValue } from '../config/legacy-enum-values.js';

// Canonical fine-grained failure class vocabulary. Single taxonomy shared
// between the classifier, the receipt, and the policy fallbackOn list.
// Standardized across governed diagnose routing and error receipts. Values
// are kebab-case; snake_case spellings from before the rename are read through
// normalizeFailureClass (for example in a policy fallbackOn list).
export const FAILURE_CLASSES = Object.freeze({
  GPU_DEVICE_LOST: 'gpu-device-lost',
  GPU_OOM: 'gpu-oom',
  GPU_TIMEOUT: 'gpu-timeout',
  GPU_UNSUPPORTED: 'gpu-unsupported',
  GPU_UNAVAILABLE: 'gpu-unavailable',
  MODEL_LOAD_FAILED: 'model-load-failed',
  POLICY_DENIED: 'policy-denied',
  RUNTIME_INTERNAL: 'runtime-internal',
  FALLBACK_FAILED: 'fallback-failed',
  UNKNOWN: 'unknown',
});

// Canonical mapping from error codes to fine-grained classes.
const ERROR_CODE_TO_CLASS = {
  [ERROR_CODES.GPU_OOM]: FAILURE_CLASSES.GPU_OOM,
  [ERROR_CODES.GPU_DEVICE_LOST]: FAILURE_CLASSES.GPU_DEVICE_LOST,
  [ERROR_CODES.GPU_DEVICE_FAILED]: FAILURE_CLASSES.GPU_DEVICE_LOST,
  [ERROR_CODES.GPU_TIMEOUT]: FAILURE_CLASSES.GPU_TIMEOUT,
  [ERROR_CODES.GPU_UNSUPPORTED_ADAPTER]: FAILURE_CLASSES.GPU_UNSUPPORTED,
  [ERROR_CODES.GPU_UNAVAILABLE]: FAILURE_CLASSES.GPU_UNAVAILABLE,
  [ERROR_CODES.LOADER_MANIFEST_INVALID]: FAILURE_CLASSES.MODEL_LOAD_FAILED,
  [ERROR_CODES.LOADER_SHARD_INDEX_INVALID]: FAILURE_CLASSES.MODEL_LOAD_FAILED,
  [ERROR_CODES.PROVIDER_POLICY_DENIED]: FAILURE_CLASSES.POLICY_DENIED,
  [ERROR_CODES.PROVIDER_LOCAL_FAILED]: FAILURE_CLASSES.RUNTIME_INTERNAL,
  [ERROR_CODES.PROVIDER_FALLBACK_FAILED]: FAILURE_CLASSES.FALLBACK_FAILED,
  [ERROR_CODES.PROVIDER_FALLBACK_NOT_CONFIGURED]: FAILURE_CLASSES.FALLBACK_FAILED,
  [ERROR_CODES.PROVIDER_NETWORK_FAILED]: FAILURE_CLASSES.FALLBACK_FAILED,
};

// Regex fallback for thrown errors that do not carry a Doppler error code.
// Ordered most-specific first.
const MESSAGE_PATTERNS = [
  { pattern: /device.?lost/i, cls: FAILURE_CLASSES.GPU_DEVICE_LOST, code: ERROR_CODES.GPU_DEVICE_LOST },
  { pattern: /out of memory|\boom\b/i, cls: FAILURE_CLASSES.GPU_OOM, code: ERROR_CODES.GPU_OOM },
  { pattern: /timeout/i, cls: FAILURE_CLASSES.GPU_TIMEOUT, code: ERROR_CODES.GPU_TIMEOUT },
  { pattern: /webgpu.*unavailable|navigator\.gpu|no adapter/i, cls: FAILURE_CLASSES.GPU_UNAVAILABLE, code: ERROR_CODES.GPU_UNAVAILABLE },
  { pattern: /unsupported.*adapter|unsupported.*feature/i, cls: FAILURE_CLASSES.GPU_UNSUPPORTED, code: ERROR_CODES.GPU_UNSUPPORTED_ADAPTER },
  { pattern: /manifest|shard|load.*model|failed to fetch/i, cls: FAILURE_CLASSES.MODEL_LOAD_FAILED, code: ERROR_CODES.LOADER_MANIFEST_INVALID },
  { pattern: /policy.?denied|not permitted/i, cls: FAILURE_CLASSES.POLICY_DENIED, code: ERROR_CODES.PROVIDER_POLICY_DENIED },
];

export function normalizeFailureClass(value) {
  return canonicalizeLegacyEnumValue(value);
}

function classifyFromCode(errorCode) {
  if (!errorCode) return null;
  return ERROR_CODE_TO_CLASS[errorCode] || null;
}

function classifyFromMessage(message) {
  for (const entry of MESSAGE_PATTERNS) {
    if (entry.pattern.test(message)) {
      return { cls: entry.cls, code: entry.code };
    }
  }
  return null;
}

function inferStage(error, message) {
  const lower = message.toLowerCase();
  if (lower.includes('prefill')) return 'prefill';
  if (lower.includes('decode') || lower.includes('decoding')) return 'decode';
  if (lower.includes('load') || lower.includes('manifest') || lower.includes('shard')) return 'load';
  if (lower.includes('policy')) return 'policy';
  if (error && error.__dopplerFaultStage) return String(error.__dopplerFaultStage);
  return 'unknown';
}

function inferSurface(message, providedSurface) {
  if (providedSurface) return providedSurface;
  const lower = message.toLowerCase();
  if (lower.includes('webgpu') || lower.includes('gpu') || lower.includes('adapter')) return 'webgpu';
  if (lower.includes('openai') || lower.includes('fallback') || lower.includes('fetch')) return 'openai-compat';
  return 'unknown';
}

export function classifyProviderFailure(error, context = {}) {
  const message = error instanceof Error ? error.message : String(error);
  const errorCode = error && typeof error === 'object' && typeof error.code === 'string' ? error.code : null;

  const codeClass = classifyFromCode(errorCode);
  let failureClass = codeClass;
  let failureCode = errorCode || ERROR_CODES.PROVIDER_LOCAL_FAILED;

  if (!failureClass) {
    const match = classifyFromMessage(message);
    if (match) {
      failureClass = match.cls;
      failureCode = match.code;
    } else {
      failureClass = FAILURE_CLASSES.UNKNOWN;
    }
  }

  const isSimulated = Boolean(error && typeof error === 'object' && error.__dopplerFaultInjected);

  return {
    failureClass,
    failureCode,
    stage: context.stage || inferStage(error, message),
    surface: inferSurface(message, context.surface),
    device: context.device ?? null,
    modelId: context.modelId ?? null,
    runtimeProfile: context.runtimeProfile ?? null,
    kernelPathId: context.kernelPathId ?? null,
    isSimulated,
    message,
  };
}
