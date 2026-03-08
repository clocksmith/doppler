const SUPPORTED_TJS_DTYPES = Object.freeze(['fp16', 'q4', 'q4f16']);
const DEFAULT_CACHE_MODE = 'warm';
const DEFAULT_TJS_VERSION = '4';
const UNKNOWN_LABEL = 'unknown';
const EMPTY_STRING = '';
const HF_CACHE_TOKEN_FILE = '.cache/huggingface/token';

export function normalizePreferredDtype(dtype) {
  const normalized = String(dtype || 'fp16').trim().toLowerCase();
  return SUPPORTED_TJS_DTYPES.includes(normalized) ? normalized : 'fp16';
}

export function buildStrictWebgpuExecution(preferredDtype = 'fp16') {
  const requestedDtype = normalizePreferredDtype(preferredDtype);
  return Object.freeze({
    requestedDtype,
    effectiveDtype: requestedDtype,
    executionProviderMode: 'webgpu-only',
    effectiveOrtProxy: false,
    fallbackUsed: false,
    ortProxyFallbackUsed: false,
    executionProviderFallbackUsed: false,
  });
}

export function requiresPersistentBrowserContext(cacheMode, loadMode) {
  return cacheMode === 'warm' || loadMode === 'opfs';
}

export function persistentContextFailureMessage(cacheMode, loadMode) {
  if (loadMode === 'opfs') {
    return 'loadMode=opfs requires persistent browser context; persistent launch failed.';
  }
  if (cacheMode === 'warm') {
    return 'cacheMode=warm requires persistent browser context; persistent launch failed.';
  }
  return 'persistent browser context required by benchmark contract; persistent launch failed.';
}

export { SUPPORTED_TJS_DTYPES };
export {
  DEFAULT_CACHE_MODE,
  DEFAULT_TJS_VERSION,
  UNKNOWN_LABEL,
  EMPTY_STRING,
  HF_CACHE_TOKEN_FILE,
};
