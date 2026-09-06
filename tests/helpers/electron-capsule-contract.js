import assert from 'node:assert/strict';
import { createDopplerRuntime } from 'doppler-gpu';
import { createElectronRendererRuntime } from 'doppler-gpu/electron';

// Signed fixture execution with a synthetic program/device, never hardware evidence.
export async function runElectronCapsuleContract({ fixture, trustedSigners, createRenderer }) {
  const { capsule, artifactStore } = fixture;
  const reference = { capsuleId: capsule.capsuleId, semanticRoot: capsule.semanticRoot, path: 'capsules/current.json' };
  let current = reference;
  let resolveFailure = null;
  const releaseState = {
    async resolveCurrent() {
      if (resolveFailure) throw resolveFailure;
      return current;
    },
  };
  const calls = [];
  let opened = 0;
  let closed = 0;
  let onOpen = null;
  let onRun = null;
  let closeFailure = null;
  const evidence = {
    schema: 'doppler_rerank_evidence/v1',
    inputHash: `sha256:${'1'.repeat(64)}`,
    outputHash: `sha256:${'2'.repeat(64)}`,
    backendIdentityHash: `sha256:${'3'.repeat(64)}`,
  };
  const ports = {
    device: {
      getProfile: () => ({ surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }),
      getDevice: () => ({
        limits: { maxBufferSize: 1024 },
        createBuffer: () => ({ destroy() {} }),
        createCommandEncoder() {},
        queue: { writeBuffer() {} },
      }),
    },
    capsuleSource: { async fetchCapsule(id) { assert.equal(id, reference.path); return capsule; } },
    artifactStore,
    trustedSigners,
    async programFactory({ options }) {
      opened += 1;
      await onOpen?.(options);
      return {
        executionGraphHash: capsule.program.executionGraphHash,
        tokenize: () => [],
        decodeTokens: () => '',
        getTokenContract: () => ({}),
        reset() {},
        async executePhase() { throw new Error('Not a generation test.'); },
        releaseStepResult() {},
        async rerank(request) { calls.push(request); await onRun?.(request); return evidence; },
        async close() { closed += 1; if (closeFailure) throw closeFailure; },
      };
    },
  };
  const renderer = createRenderer(releaseState, ports);
  const request = {
    application: structuredClone(capsule.release.application),
    query: 'local search', documents: ['first', 'second'], options: { benchmark: false },
  };
  const receipt = await renderer.rerank(request);
  assert.equal(receipt.schema, 'doppler.capsule-rerank-receipt/v1');
  assert.equal(receipt.capsule.semanticRoot, capsule.semanticRoot);
  assert.deepEqual(receipt.application, request.application);
  assert.deepEqual(calls[0], { query: request.query, documents: request.documents, options: request.options });
  assert.equal(closed, 1);

  await assert.rejects(renderer.rerank('query', ['document']), /explicit application binding/);
  const mismatched = structuredClone(request);
  mismatched.application.applicationRevision = 'not-the-signed-application';
  await assert.rejects(renderer.rerank(mismatched), /does not match the signed Capsule/);
  assert.equal(calls.length, 1, 'application mismatch must not reach inference');
  assert.equal(opened, closed);

  const abort = new AbortController();
  abort.abort();
  const beforeAbort = opened;
  await assert.rejects(renderer.rerank(request, { signal: abort.signal }), { code: 'DOPPLER_ELECTRON_CANCELLED' });
  assert.equal(opened, beforeAbort, 'cancelled requests must not load');

  const duringOpen = new AbortController();
  onOpen = (options) => {
    assert.equal(options.signal.aborted, false, 'loading receives an active cancellation scope');
    duringOpen.abort();
    assert.equal(options.signal.aborted, true, 'application cancellation reaches the loading scope');
    assert.equal(options.signal.reason, duringOpen.signal.reason);
  };
  await assert.rejects(renderer.rerank(request, { signal: duringOpen.signal }), { code: 'DOPPLER_ELECTRON_CANCELLED' });
  assert.equal(opened, closed, 'a session loaded after cancellation must close');
  onOpen = () => { current = { ...reference, semanticRoot: `sha256:${'4'.repeat(64)}` }; };
  await assert.rejects(renderer.rerank(request), { code: 'DOPPLER_ELECTRON_RELEASE_CHANGED' });
  assert.equal(opened, closed, 'a session loaded across an upgrade must close');
  onOpen = null;
  current = reference;

  resolveFailure = new Error('revocation state is expired');
  const beforeExpired = opened;
  await assert.rejects(renderer.rerank(request), /revocation state is expired/);
  assert.equal(opened, beforeExpired);
  resolveFailure = null;
  onOpen = () => { resolveFailure = new Error('revocation state is expired'); };
  await assert.rejects(renderer.rerank(request), /revocation state is expired/);
  assert.equal(opened, closed, 'expiry during loading must release the session');
  onOpen = null;
  resolveFailure = null;

  const runtime = createDopplerRuntime(ports);
  const wrongCapsuleRenderer = createElectronRendererRuntime({
    releaseState,
    async openCapsule() { return { ...await runtime.openCapsule(capsule), capsuleId: 'another-signed-capsule' }; },
  });
  await assert.rejects(wrongCapsuleRenderer.rerank(request), { code: 'DOPPLER_ELECTRON_RELEASE_CHANGED' });
  assert.equal(opened, closed, 'the path is not authority for a different Capsule');

  onRun = () => { current = { ...reference, capsuleId: 'new-release' }; };
  await assert.rejects(renderer.rerank(request), { code: 'DOPPLER_ELECTRON_RELEASE_CHANGED' });
  current = reference;
  const duringRun = new AbortController();
  onRun = (received) => {
    assert.equal(received.options.signal, duringRun.signal, 'cancellation must reach the actual Capsule program');
    duringRun.abort();
  };
  await assert.rejects(renderer.rerank(request, { signal: duringRun.signal }), { code: 'DOPPLER_ELECTRON_CANCELLED' });
  for (const cancelSource of ['open', 'request']) {
    const opening = new AbortController();
    const running = new AbortController();
    onRun = (received) => {
      const signal = received.options.signal;
      assert.notEqual(signal, opening.signal);
      assert.notEqual(signal, running.signal);
      (cancelSource === 'open' ? opening : running).abort();
      assert.equal(signal.aborted, true, 'either caller can cancel the combined execution');
    };
    await assert.rejects(renderer.rerank({ ...request, options: { signal: running.signal } },
      { signal: opening.signal }), { code: 'DOPPLER_ELECTRON_CANCELLED' });
    assert.equal(opened, closed);
  }
  const beforeRequestAbort = opened;
  await assert.rejects(renderer.rerank({ ...request, options: { signal: abort.signal } }),
    { code: 'DOPPLER_ELECTRON_CANCELLED' });
  assert.equal(opened, beforeRequestAbort);
  onRun = () => { throw Object.assign(new Error('adapter removed'), { code: 'GPU_DEVICE_LOST' }); };
  closeFailure = new Error('cleanup also failed');
  await assert.rejects(renderer.rerank(request), { code: 'DOPPLER_ELECTRON_DEVICE_LOST' });
  assert.equal(opened, closed, 'failure cleanup runs exactly once');
  onRun = () => { throw Object.assign(new Error('physical device destroyed'), { code: 'DOPPLER_GPU_DEVICE_LOST' }); };
  await assert.rejects(renderer.rerank(request), (error) => {
    assert.equal(error.code, 'DOPPLER_ELECTRON_DEVICE_LOST');
    assert.equal(error.cause.code, 'DOPPLER_GPU_DEVICE_LOST');
    return true;
  });
  assert.equal(opened, closed, 'typed physical loss preserves cleanup and original cause');
  onRun = null;
  await assert.rejects(renderer.rerank(request), /cleanup also failed/);
  closeFailure = null;

  const warm = await renderer.openCurrent();
  const warmOpenCount = opened;
  await warm.rerank(request);
  await warm.rerank(request);
  assert.equal(opened, warmOpenCount, 'explicit caller-owned sessions reuse the loaded program');
  await warm.close();
  await warm.close();
  assert.equal(opened, closed, 'caller-owned close is idempotent');
  await assert.rejects(warm.rerank(request), /session is closed/);
  assert.throws(() => createRenderer(releaseState, {}), /device port/);
}
