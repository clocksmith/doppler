import assert from 'node:assert/strict';

import { createDopplerProvider } from '../../src/client/provider.js';
import { ERROR_CODES } from '../../src/errors/doppler-error.js';

// --- Config validation ---

assert.throws(
  () => createDopplerProvider(null),
  /requires a config object/,
  'rejects null config'
);

assert.throws(
  () => createDopplerProvider({ policy: { mode: 'bogus' } }),
  /Invalid policy mode/,
  'rejects unknown policy mode'
);

assert.throws(
  () => createDopplerProvider({ policy: { mode: 'prefer-local' } }),
  /requires local\.model/,
  'rejects prefer-local without local.model'
);

assert.throws(
  () => createDopplerProvider({ policy: { mode: 'local-only' } }),
  /requires local\.model/,
  'rejects local-only without local.model'
);

assert.throws(
  () => createDopplerProvider({
    policy: { mode: 'cloud-only' },
    fallback: { model: 'gpt-4o' },
  }),
  /fallback requires fallback\.provider/,
  'rejects fallback without provider field'
);

assert.throws(
  () => createDopplerProvider({
    policy: { mode: 'cloud-only' },
    fallback: { provider: 'openai' },
  }),
  /fallback requires fallback\.model/,
  'rejects fallback without model field'
);

assert.throws(
  () => createDopplerProvider({ policy: { mode: 'fallback-only' } }),
  /Invalid policy mode/,
  'rejects deprecated fallback-only mode (removed)'
);

// --- Valid configs produce a provider ---

const providerLocal = createDopplerProvider({
  local: { model: 'gemma-3-270m' },
  policy: { mode: 'local-only' },
});
assert.equal(typeof providerLocal.generate, 'function', 'provider exposes generate');
assert.equal(typeof providerLocal.unload, 'function', 'provider exposes unload');

const providerCloud = createDopplerProvider({
  policy: { mode: 'cloud-only' },
  fallback: { provider: 'openai', model: 'gpt-4o', apiKey: 'test-key' },
});
assert.equal(typeof providerCloud.generate, 'function', 'cloud-only provider exposes generate');

const providerPreferCloud = createDopplerProvider({
  local: { model: 'gemma-3-270m' },
  fallback: { provider: 'openai', model: 'gpt-4o', apiKey: 'test-key' },
  policy: { mode: 'prefer-cloud' },
});
assert.equal(typeof providerPreferCloud.generate, 'function', 'prefer-cloud provider exposes generate');

const providerPrefer = createDopplerProvider({
  local: { model: 'gemma-3-270m' },
  fallback: { provider: 'openai', model: 'gpt-4o' },
  policy: { mode: 'prefer-local' },
  diagnostics: { receipts: true },
});
assert.equal(typeof providerPrefer.generate, 'function', 'prefer-local provider exposes generate');

// --- Default policy mode ---

const providerDefault = createDopplerProvider({
  local: { model: 'gemma-3-270m' },
});
assert.equal(typeof providerDefault.generate, 'function', 'default policy mode is accepted');

// --- Error codes exist ---

assert.equal(typeof ERROR_CODES.GPU_OOM, 'string', 'GPU_OOM error code exists');
assert.equal(typeof ERROR_CODES.GPU_DEVICE_LOST, 'string', 'GPU_DEVICE_LOST error code exists');
assert.equal(typeof ERROR_CODES.GPU_TIMEOUT, 'string', 'GPU_TIMEOUT error code exists');
assert.equal(typeof ERROR_CODES.GPU_UNSUPPORTED_ADAPTER, 'string', 'GPU_UNSUPPORTED_ADAPTER error code exists');
assert.equal(typeof ERROR_CODES.PROVIDER_LOCAL_FAILED, 'string', 'PROVIDER_LOCAL_FAILED error code exists');
assert.equal(typeof ERROR_CODES.PROVIDER_FALLBACK_FAILED, 'string', 'PROVIDER_FALLBACK_FAILED error code exists');
assert.equal(typeof ERROR_CODES.PROVIDER_FALLBACK_NOT_CONFIGURED, 'string', 'PROVIDER_FALLBACK_NOT_CONFIGURED error code exists');
assert.equal(typeof ERROR_CODES.PROVIDER_POLICY_DENIED, 'string', 'PROVIDER_POLICY_DENIED error code exists');

// --- cloud-only with a valid fallback constructs cleanly ---

const providerCloudOnly = createDopplerProvider({
  policy: { mode: 'cloud-only' },
  fallback: { provider: 'openai', model: 'gpt-4o' },
});
assert.equal(typeof providerCloudOnly.generate, 'function');

console.log('doppler-provider-contract.test: ok');
