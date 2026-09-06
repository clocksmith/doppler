import assert from 'node:assert/strict';
import { getEventListeners } from 'node:events';
import { assertCapsuleLoadActive, createCapsuleLoadScope } from '../../src/client/runtime/capsule-acquisition.js';
import { createDopplerRuntime } from '../../src/client/runtime/composition-root.js';
import { openCapsule } from '../../src/capsule-runtime.js';
import { createSignedCapsuleFixture, TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PUBLIC_KEY } from '../helpers/capsule-v2-fixture.js';

const fixture = await createSignedCapsuleFixture();
const trustedSigners = { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY };
let reads = 0;
let programs = 0;
const runtime = createDopplerRuntime({
  device: { getProfile: () => ({ surface: 'test-webgpu', maxBufferSize: 0 }) },
  trustedSigners,
  artifactStore: { async readArtifact(artifact) { reads += 1; return fixture.artifactStore.readArtifact(artifact); } },
  async programFactory() { programs += 1; return {}; },
});
await assert.rejects(runtime.openCapsule(fixture.capsule), /target|compatible/i);
assert.equal(reads, 0, 'An incompatible device must not acquire any artifacts.');
assert.equal(programs, 0);

const cancelled = new AbortController();
cancelled.abort(new DOMException('Stop opening', 'AbortError'));
await assert.rejects(runtime.openCapsule(fixture.capsule, { signal: cancelled.signal }), error => error === cancelled.signal.reason);
assert.equal(reads, 0);
await assert.rejects(openCapsule(fixture.capsule, { device: {}, trustedSigners, artifactStore: fixture.artifactStore,
  programFactory: async () => { throw new Error('Cancelled public opening must not create a program.'); },
  session: { signal: cancelled.signal } }), error => error === cancelled.signal.reason);

for (const loadTimeoutMs of [0, -1, 0.5, NaN, Infinity, 2147483648, '100']) {
  await assert.rejects(runtime.openCapsule(fixture.capsule, { loadTimeoutMs }), /loadTimeoutMs/);
}
await assert.rejects(runtime.openCapsule(fixture.capsule, { signal: {} }), /AbortSignal/);
await assert.rejects(runtime.openCapsule(fixture.capsule, { onLoadProgress: true }), /onLoadProgress/);
assert.equal(reads, 0);

const device = { getProfile: () => ({ surface: 'test-webgpu', maxBufferSize: 1024 }),
  getDevice: () => ({ createBuffer() {}, createCommandEncoder() {} }) };
const untrusted = createDopplerRuntime({
  device: { getProfile() { throw new Error('Untrusted metadata must not inspect the device.'); } }, trustedSigners: {},
  artifactStore: { async readArtifact() { throw new Error('Untrusted metadata must not acquire artifacts.'); } },
  programFactory: async () => { throw new Error('Untrusted metadata must not execute.'); },
});
await assert.rejects(untrusted.openCapsule(fixture.capsule), /Untrusted/);
for (const schema of ['doppler.pack/v2', 'doppler.pack/v3']) {
  const previousFormat = { ...fixture.capsule, schema, packId: fixture.capsule.capsuleId };
  delete previousFormat.capsuleId;
  await assert.rejects(untrusted.openCapsule(previousFormat), /Unsupported Doppler Capsule schema/);
}
await assert.rejects(untrusted.openCapsule({ ...fixture.capsule, packId: fixture.capsule.capsuleId }),
  /packId is not allowed/);

for (const cleanupFails of [false, true]) {
  const controller = new AbortController();
  const failure = new DOMException('Cancelled during construction', 'AbortError');
  const cleanupFailure = new Error('Program cleanup failed');
  let closed = 0;
  const late = createDopplerRuntime({ device, trustedSigners, artifactStore: fixture.artifactStore,
    async programFactory() {
      controller.abort(failure);
      return { async close() { closed += 1; if (cleanupFails) throw cleanupFailure; } };
    } });
  await assert.rejects(late.openCapsule(fixture.capsule, { signal: controller.signal }), error => {
    if (!cleanupFails) return error === failure;
    assert.equal(error.cause, failure);
    assert.deepEqual(error.errors, [failure, cleanupFailure]);
    return true;
  });
  assert.equal(closed, 1);
  assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
}

const completed = new AbortController();
let loadingSignal;
const usable = createDopplerRuntime({ device, trustedSigners, artifactStore: fixture.artifactStore,
  async programFactory({ options }) { loadingSignal = options.signal; return { async close() {} }; } });
const session = await usable.openCapsule(fixture.capsule, { signal: completed.signal, loadTimeoutMs: 5000 });
assert.equal(getEventListeners(completed.signal, 'abort').length, 0, 'opening detaches its parent listener');
completed.abort();
assert.equal(loadingSignal.aborted, false, 'completed loading does not inherit later application cancellation');
assert.equal(session.closed, false);
await session.close();
const deadline = createCapsuleLoadScope({ loadTimeoutMs: 2 });
const began = performance.now();
while (performance.now() - began < 5) { /* Deliberately prevent timers from running. */ }
assert.throws(() => assertCapsuleLoadActive(deadline.options.signal), { name: 'TimeoutError' });
deadline.close();
console.log('capsule-loading.test: ok (signed metadata; synthetic device)');
