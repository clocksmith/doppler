import assert from 'node:assert/strict';
import { test } from 'node:test';
import { runInstalledCapabilityLifecycle } from '../fixtures/installed-capability-lifecycle.js';

const descriptor = { capsuleUrl: 'fixture', openOptions: {}, maxDurationMs: 1000,
  request: { schema: 'doppler.capsule-operation-request/v2', operation: { name: 'generate', version: 1 }, limits: {} } };
const policy = { repeatRuns: 2, cancellation: 'after-partial' };
function fixture({ ignorePreparationAbort = false, ignoreOperationAbort = false, failCleanup = false, adapters = false } = {}) {
  const sessions = [];
  const host = { createCapsuleStreamAccumulator() {
    let completed;
    return { accept(event) { if (event.status === 'completed') completed = event; }, finish() { return completed; } };
  }, async openCapsule(_url, options) {
    options.observer?.observe?.({ type: 'target-selected' });
    if (!ignorePreparationAbort) options.signal?.throwIfAborted();
    const session = { closed: false, manifest: { architecture: {}, inference: {} },
      async *executeOperation(_request, { signal, adapterArtifactStore } = {}) {
        assert.equal(this.closed, false);
        if (!ignoreOperationAbort) signal?.throwIfAborted();
        const adapter = adapters ? _request.adapterSet?.[0] : null;
        if (adapter) await adapterArtifactStore.readArtifact(adapter.artifact);
        yield { status: 'partial' };
        if (!ignoreOperationAbort) signal?.throwIfAborted();
        yield { status: 'completed', output: { tokenIds: [1] }, receipt: { adapterReceipts: adapter ? [{
          identity: adapter.identity, sourceDigest: adapter.artifact.hash,
          runtimeIdentity: { schema: 'doppler.lora-execution-identity/v1', digest: 'sha256:' + '0'.repeat(64) },
        }] : [] } };
      },
      async close() { this.closed = true; if (failCleanup) throw new Error('fixture cleanup failed'); },
    };
    sessions.push(session);
    return session;
  } };
  return { host, sessions };
}

test('installed lifecycle acceptance covers repeated work, cancellation, and surviving sessions', async () => {
  const { host, sessions } = fixture();
  const result = await runInstalledCapabilityLifecycle(host, descriptor, {}, policy);
  assert.deepEqual(result.lifecycle.observations.map(row => row.phase), [
    'repeat-0', 'repeat-1', 'after-cancellation-and-failed-preparation', 'second-after-first-close',
  ]);
  assert.equal(result.lifecycle.checks.length, 3);
  assert(sessions.every(session => session.closed));
});

test('pre-aborted acceptance does not claim partial GPU execution', async () => {
  const { host } = fixture();
  const result = await runInstalledCapabilityLifecycle(host, descriptor, {}, { ...policy, cancellation: 'pre-aborted' });
  assert.equal(result.lifecycle.checks.find(row => row.id === 'cancellation').partials, 0);
});

test('acceptance rejects a preparation that ignores cancellation', async () => {
  const { host, sessions } = fixture({ ignorePreparationAbort: true });
  await assert.rejects(runInstalledCapabilityLifecycle(host, descriptor, {}, policy), /Preparation must reject/);
  assert(sessions.every(session => session.closed));
});

test('acceptance rejects a late completion after cancellation', async () => {
  const { host, sessions } = fixture({ ignoreOperationAbort: true });
  await assert.rejects(runInstalledCapabilityLifecycle(host, descriptor, {}, policy), /without accepting a completion/);
  assert(sessions.every(session => session.closed));
});

test('acceptance preserves the original failure when cleanup fails', async () => {
  const { host } = fixture({ ignorePreparationAbort: true, failCleanup: true });
  await assert.rejects(runInstalledCapabilityLifecycle(host, descriptor, {}, policy), error => {
    assert(error instanceof AggregateError);
    assert.match(error.cause.message, /Preparation must reject/);
    assert.match(error.errors[1].message, /fixture cleanup failed/);
    return true;
  });
});

test('adapter acceptance rejects a completion without exact loaded tensor evidence', async () => {
  const { host, sessions } = fixture();
  await assert.rejects(runInstalledCapabilityLifecycle(host, descriptor, {}, { ...policy,
    adapter: { entry: { artifact: { hash: 'sha256:fixture' } }, weightsPath: 'weights.safetensors' },
  }), /physically loaded adapter tensors/);
  assert(sessions.every(session => session.closed));
});

test('adapter acceptance checks failed preparation and base requests after unloading and cancellation', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response(new Uint8Array([1, 2, 3]));
  try {
    const { host, sessions } = fixture({ adapters: true });
    const result = await runInstalledCapabilityLifecycle(host, { ...descriptor, capsuleUrl: 'https://fixture.invalid/capsule.json' }, {},
      { ...policy, adapter: { entry: { artifact: { hash: 'sha256:fixture' } }, weightsPath: 'weights.safetensors' } });
    assert(result.lifecycle.checks.some(row => row.id === 'failed-adapter-preparation' && row.passed));
    const base = result.lifecycle.observations.filter(row => row.phase.startsWith('base-'));
    assert.equal(base.length, 3);
    assert(base.every(row => row.completed.receipt.adapterReceipts.length === 0));
    assert(sessions.every(session => session.closed));
  } finally { globalThis.fetch = original; }
});
