import assert from 'node:assert/strict';

import {
  DEFAULT_CACHE_MODE,
  DEFAULT_TJS_VERSION,
  EMPTY_STRING,
  HF_CACHE_TOKEN_FILE,
  UNKNOWN_LABEL,
  buildStrictWebgpuExecution,
  normalizePreferredDtype,
  persistentContextFailureMessage,
  requiresPersistentBrowserContext,
} from '../../benchmarks/runners/transformersjs-contract.js';

assert.equal(DEFAULT_CACHE_MODE, 'warm');
assert.equal(DEFAULT_TJS_VERSION, '4');
assert.equal(UNKNOWN_LABEL, 'unknown');
assert.equal(EMPTY_STRING, '');
assert.equal(HF_CACHE_TOKEN_FILE, '.cache/huggingface/token');

assert.equal(normalizePreferredDtype('FP16'), 'fp16');
assert.equal(normalizePreferredDtype('q4f16'), 'q4f16');
assert.equal(normalizePreferredDtype('bogus'), 'fp16');

assert.equal(requiresPersistentBrowserContext('warm', 'http'), true);
assert.equal(requiresPersistentBrowserContext('cold', 'opfs'), true);
assert.equal(requiresPersistentBrowserContext('cold', 'http'), false);

assert.equal(
  persistentContextFailureMessage('warm', 'http'),
  'cacheMode=warm requires persistent browser context; persistent launch failed.'
);
assert.equal(
  persistentContextFailureMessage('cold', 'opfs'),
  'loadMode=opfs requires persistent browser context; persistent launch failed.'
);

assert.deepEqual(buildStrictWebgpuExecution('q4'), {
  requestedDtype: 'q4',
  effectiveDtype: 'q4',
  executionProviderMode: 'webgpu-only',
  effectiveOrtProxy: false,
  fallbackUsed: false,
  ortProxyFallbackUsed: false,
  executionProviderFallbackUsed: false,
});

console.log('transformersjs-contract.test: ok');
