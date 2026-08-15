import assert from 'node:assert/strict';
import { test } from 'node:test';

import { createScopedModelSession } from '../../src/client/runtime/scoped-session.js';

function makeHandle(overrides = {}) {
  let unloaded = 0;
  const evidence = {
    outputText: 'hello',
    tokenIds: [1, 2],
    resolution: {
      schema: 'doppler.resolution-identity/v1',
      logicalModelId: 'fixture-logical',
      resolvedArtifactVariantId: `sha256:${'a'.repeat(64)}`,
      resolvedExecutionId: `sha256:${'b'.repeat(64)}`,
    },
    runtimeProfile: {
      model: {
        modelId: 'fixture-model',
        manifestHash: 'sha256:manifest',
      },
    },
    runtimeProfileHash: 'sha256:runtime',
    backendIdentityHash: 'sha256:backend',
    backendIdentity: {
      executionPlanId: 'primary',
      kernelPathId: 'fixture-path',
    },
    stats: null,
  };
  const handle = {
    modelId: 'fixture-model',
    manifestHash: 'sha256:manifest',
    manifest: { modelId: 'fixture-model' },
    loaded: true,
    supportsEmbedding: true,
    supportsSequence: false,
    activeLoRA: null,
    deviceInfo: { vendor: 'fixture' },
    advanced: { tokenizeText: () => [1] },
    async generateWithEvidence() {
      return evidence;
    },
    async *generate() {
      yield 'hel';
      yield 'lo';
    },
    async embed() {
      return { embedding: [1, 2] };
    },
    async encodeSequence() {
      return { embedding: [3, 4] };
    },
    async loadLoRA() {},
    async unloadLoRA() {},
    resetGenerationState() {},
    async unload() {
      unloaded += 1;
      handle.loaded = false;
    },
    get unloadCount() {
      return unloaded;
    },
    inspect: {
      async generate(_prompt, options = {}) {
        const modifiesExecution = options.policyId !== 'demo/always-on';
        return {
          schema: 'doppler.model-inspection-receipt/v1',
          policy: {
            id: options.policyId,
            modifiesExecution,
          },
          fingerprint: {
            schema: 'doppler.comparison-fingerprint/v1',
            fullDigest: 'sha256:fingerprint',
          },
          outputText: 'hello',
          generatedTokenIds: [1, 2],
          wallTimingMs: 1,
          performanceRepresentative: !modifiesExecution,
          tokens: modifiesExecution ? [{ tokenId: 1, surprisal: 0.5 }] : [],
          quality: modifiesExecution ? { words: [] } : null,
          generationEvidence: evidence,
        };
      },
    },
    ...overrides,
  };
  return handle;
}

test('scoped session exposes explicit capabilities and stable results', async () => {
  const handle = makeHandle();
  const session = createScopedModelSession(handle);
  assert.equal(session.supports('generate'), true);
  assert.equal(session.require('generate'), true);

  const result = await session.generate('prompt');
  assert.equal(result.schema, 'doppler.generation-result/v1');
  assert.equal(result.outputText, 'hello');
  assert.equal(result.observation.executionClassification, 'representative');
  assert.equal(result.observation.executionChanged, false);
  assert.equal(result.resolution.logicalModelId, 'fixture-logical');
  assert.equal(result.resolution.resolvedArtifactVariantId, `sha256:${'a'.repeat(64)}`);
  assert.equal(result.resolution.resolvedExecutionId, `sha256:${'b'.repeat(64)}`);
  assert.equal(result.fingerprint.executionPlanId, 'primary');
  assert.equal(session.logicalModelId, 'fixture-model');
  assert.equal(session.manifest.modelId, 'fixture-model');
});

test('stream emits semantic events instead of return-dependent unions', async () => {
  const session = createScopedModelSession(makeHandle());
  const events = [];
  for await (const event of session.stream('prompt')) events.push(event);
  assert.deepEqual(events.map((event) => event.type), ['text-delta', 'text-delta', 'complete']);
  assert.equal(events.at(-1).outputText, 'hello');
});

test('unsupported capabilities and observation policies fail closed', async () => {
  const session = createScopedModelSession(makeHandle({
    supportsSequence: true,
  }));
  assert.throws(() => session.require('generate'), /does not support/);
  await assert.rejects(
    session.generate('prompt', { observe: 'mystery-mode' }),
    /Unsupported Doppler observation policy/
  );
});

test('close is idempotent and blocks later work', async () => {
  const handle = makeHandle();
  const session = createScopedModelSession(handle);
  await session.close();
  await session.close();
  assert.equal(handle.unloadCount, 1);
  assert.equal(session.closed, true);
  await assert.rejects(session.generate('prompt'), /session is closed/);
});

test('guided and deep inspection disclose non-representative execution', async () => {
  const session = createScopedModelSession(makeHandle());
  const guided = await session.inspect('prompt');
  assert.equal(guided.policy.id, 'demo/guided-quality');
  assert.equal(guided.policy.modifiesExecution, true);
  assert.ok(guided.quality);
  const deep = await session.inspect('prompt', {
    observationPolicy: 'deep-xray',
  });
  assert.equal(deep.policy.id, 'demo/deep-xray');
});
