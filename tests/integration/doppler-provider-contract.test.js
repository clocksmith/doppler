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

const resolution = {
  schema: 'doppler.resolution-identity/v1',
  logicalModelId: 'fixture-logical',
  resolvedArtifactVariantId: `sha256:${'a'.repeat(64)}`,
  resolvedExecutionId: `sha256:${'b'.repeat(64)}`,
};
const providerWithEvidence = createDopplerProvider({
  local: {
    handle: {
      loaded: true,
      modelId: 'fixture-resolved',
      resolvedArtifactVariantId: resolution.resolvedArtifactVariantId,
      manifest: { modelId: 'fixture-resolved' },
      deviceInfo: null,
      async generateText() {
        throw new Error('generateText must not replace evidence-backed generation');
      },
      async generateWithEvidence() {
        return { outputText: 'evidence-backed', resolution };
      },
      async unload() {},
    },
  },
  policy: { mode: 'local-only' },
});
const evidenceResult = await providerWithEvidence.generate('hello');
assert.equal(evidenceResult.text, 'evidence-backed');
assert.equal(evidenceResult.receipt.resolutionStatus, 'resolved');
assert.deepEqual(evidenceResult.receipt.resolution, resolution);

// --- fallbackOn failure classes: kebab-case, legacy snake_case still read ---

function oomHandle() {
  return {
    loaded: true,
    modelId: 'fixture-oom',
    manifest: { modelId: 'fixture-oom' },
    deviceInfo: null,
    async generateText() {
      throw new Error('WebGPU out of memory during prefill');
    },
    async unload() {},
  };
}

const originalFetch = globalThis.fetch;
globalThis.fetch = async () => new Response(JSON.stringify({
  choices: [{ message: { content: 'from-fallback' } }],
}), { status: 200, headers: { 'Content-Type': 'application/json' } });
try {
  for (const fallbackOn of [['gpu-oom'], ['gpu_oom']]) {
    const provider = createDopplerProvider({
      local: { handle: oomHandle() },
      policy: { mode: 'prefer-local', fallbackOn },
      fallback: { provider: 'openai', model: 'gpt-4o', baseUrl: 'http://127.0.0.1:9' },
    });
    const result = await provider.generate('hello');
    assert.equal(result.inferenceSource, 'fallback', `fallbackOn ${fallbackOn[0]} admits gpu-oom`);
    assert.equal(result.receipt.failure.failureClass, 'gpu-oom');
    assert.equal(result.receipt.fallbackDecision.reason, 'gpu-oom');
  }

  const deniedProvider = createDopplerProvider({
    local: { handle: oomHandle() },
    policy: { mode: 'prefer-local', fallbackOn: ['gpu_timeout'] },
    fallback: { provider: 'openai', model: 'gpt-4o', baseUrl: 'http://127.0.0.1:9' },
  });
  await assert.rejects(
    () => deniedProvider.generate('hello'),
    (error) => error.code === ERROR_CODES.PROVIDER_POLICY_DENIED && /gpu-oom/.test(error.message)
  );
} finally {
  globalThis.fetch = originalFetch;
}

console.log('doppler-provider-contract.test: ok');
