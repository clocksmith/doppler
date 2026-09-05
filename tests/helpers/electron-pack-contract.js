import assert from 'node:assert/strict';
import { createDopplerRuntime } from 'doppler-gpu';
import { createElectronRendererRuntime } from 'doppler-gpu/electron';

// Signed fixture execution with a synthetic program/device, never hardware evidence.
export async function runElectronPackContract({ fixture, trustedSigners, createRenderer }) {
  const { pack, artifactStore } = fixture;
  const reference = { packId: pack.packId, semanticRoot: pack.semanticRoot, path: 'packs/current.json' };
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
    packSource: { async fetchPack(id) { assert.equal(id, reference.path); return pack; } },
    artifactStore,
    trustedSigners,
    async programFactory({ options }) {
      opened += 1;
      await onOpen?.(options);
      return {
        executionGraphHash: pack.program.executionGraphHash,
        tokenize: () => [],
        decodeTokens: () => '',
        getTokenContract: () => ({}),
        reset() {},
        async executePhase() { throw new Error('Not a generation test.'); },
        releaseStepResult() {},
        async rerank(request) { calls.push(request); await onRun?.(); return evidence; },
        async close() { closed += 1; if (closeFailure) throw closeFailure; },
      };
    },
  };
  const renderer = createRenderer(releaseState, ports);
  const request = {
    application: structuredClone(pack.release.application),
    query: 'local search', documents: ['first', 'second'], options: { benchmark: false },
  };
  const receipt = await renderer.rerank(request);
  assert.equal(receipt.schema, 'doppler.pack-rerank-receipt/v1');
  assert.equal(receipt.pack.semanticRoot, pack.semanticRoot);
  assert.deepEqual(receipt.application, request.application);
  assert.deepEqual(calls[0], { query: request.query, documents: request.documents, options: request.options });
  assert.equal(closed, 1);

  await assert.rejects(renderer.rerank('query', ['document']), /explicit application binding/);
  const mismatched = structuredClone(request);
  mismatched.application.applicationRevision = 'not-the-signed-application';
  await assert.rejects(renderer.rerank(mismatched), /does not match the signed Pack/);
  assert.equal(calls.length, 1, 'application mismatch must not reach inference');
  assert.equal(opened, closed);

  const abort = new AbortController();
  abort.abort();
  const beforeAbort = opened;
  await assert.rejects(renderer.rerank(request, { signal: abort.signal }), { code: 'DOPPLER_ELECTRON_CANCELLED' });
  assert.equal(opened, beforeAbort, 'cancelled requests must not load');

  const duringOpen = new AbortController();
  onOpen = (options) => {
    assert.equal(options.signal, duringOpen.signal, 'host receives explicit session options');
    duringOpen.abort();
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
  const wrongPackRenderer = createElectronRendererRuntime({
    releaseState,
    async openPack() { return { ...await runtime.openPack(pack), packId: 'another-signed-pack' }; },
  });
  await assert.rejects(wrongPackRenderer.rerank(request), { code: 'DOPPLER_ELECTRON_RELEASE_CHANGED' });
  assert.equal(opened, closed, 'the path is not authority for a different Pack');

  onRun = () => { current = { ...reference, packId: 'new-release' }; };
  await assert.rejects(renderer.rerank(request), { code: 'DOPPLER_ELECTRON_RELEASE_CHANGED' });
  current = reference;
  const duringRun = new AbortController();
  onRun = () => duringRun.abort();
  await assert.rejects(renderer.rerank(request, { signal: duringRun.signal }), { code: 'DOPPLER_ELECTRON_CANCELLED' });
  onRun = () => { throw Object.assign(new Error('adapter removed'), { code: 'GPU_DEVICE_LOST' }); };
  closeFailure = new Error('cleanup also failed');
  await assert.rejects(renderer.rerank(request), { code: 'DOPPLER_ELECTRON_DEVICE_LOST' });
  assert.equal(opened, closed, 'failure cleanup runs exactly once');
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
