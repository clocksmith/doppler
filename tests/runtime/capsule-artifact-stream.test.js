import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createVerifiedCapsuleArtifactStore, createCapsuleArtifactBacking } from '../../src/client/runtime/verified-capsule-artifact-store.js';

const bytes = new Uint8Array(2 * 1024 * 1024 + 37).fill(19);
const artifact = { artifactId: 'weights', path: 'weights.bin', role: 'weight-shard', sizeBytes: bytes.length,
  hash: `sha256:${createHash('sha256').update(bytes).digest('hex')}` };
const capsule = { artifacts: [artifact] };
const noWholeRead = () => { throw new Error('Streaming must not acquire the whole artifact'); };
const source = {
  readArtifact: noWholeRead,
  async *streamArtifact(_artifact, { maxChunkBytes }) {
    const buffer = new Uint8Array(Math.min(17003, maxChunkBytes));
    for (let offset = 0; offset < bytes.length; offset += buffer.length) {
      buffer.fill(19);
      yield buffer.subarray(0, Math.min(buffer.length, bytes.length - offset));
      buffer.fill(0); // A producer may reuse its buffer only after the next pull.
    }
  },
};

// Arbitrary chunk boundaries, buffer reuse, owned slices, and shared publication.
const backing = createCapsuleArtifactBacking();
const store = createVerifiedCapsuleArtifactStore(capsule, source, {}, backing);
const hash = await store.hashArtifact(artifact);
assert.equal(hash.hash, artifact.hash);
assert.equal(store.getMetrics().streamedSourceBytes, bytes.length);
assert.equal(store.getMetrics().peakSourceChunkBytes, 17003);
assert.equal(store.getMetrics().peakSnapshotBlockBytes, 65536);
const range = await store.readArtifactRange(artifact, 65533, 23);
assert.deepEqual(range, bytes.subarray(65533, 65556)); range.fill(0);
const shared = createVerifiedCapsuleArtifactStore(capsule, { readArtifact: noWholeRead }, {}, backing);
await shared.hashArtifact(artifact);
store.close();
assert.deepEqual(await shared.readArtifactRange(artifact, 0, 3), new Uint8Array([19, 19, 19]));
shared.close();

// Timer tasks, rather than pre-aborted signals or microtasks, interrupt hashing.
for (const streamed of [false, true]) {
  const controller = new AbortController();
  const input = streamed ? source : { async readArtifact() { return bytes; } };
  const active = createVerifiedCapsuleArtifactStore(capsule, input, { signal: controller.signal, verificationYieldBytes: 65536 });
  const timer = setTimeout(() => controller.abort(new Error('timer cancellation during hashing')), 0);
  try {
    await assert.rejects(active.hashArtifact(artifact), /timer cancellation/);
    const metrics = active.getMetrics();
    assert(metrics.hashedBytes > 0 && metrics.hashedBytes < bytes.length);
    assert.equal(metrics.backingBytes, 0);
  } finally { clearTimeout(timer); active.close(); }
}

// Truncation, oversize, corruption, invalid values, and backing-buffer bounds.
for (const mode of ['truncated', 'oversized', 'corrupt', 'invalid', 'oversized-view']) {
  let returned = false;
  const invalid = { readArtifact: noWholeRead, async *streamArtifact() {
    try {
      if (mode === 'invalid') { yield 'not bytes'; return; }
      if (mode === 'oversized-view') { yield new Uint8Array(1025).subarray(0, 1); return; }
      const length = bytes.length + (mode === 'truncated' ? -1 : mode === 'oversized' ? 1 : 0);
      for (let i = 0; i < length; i += 1024) yield new Uint8Array(Math.min(1024, length - i)).fill(mode === 'corrupt' ? 0 : 19);
    } finally { returned = true; }
  } };
  const active = createVerifiedCapsuleArtifactStore(capsule, invalid, { maxAcquisitionChunkBytes: 1024 });
  await assert.rejects(active.hashArtifact(artifact), /size mismatch|byte chunks|acquisition limit/);
  await Promise.resolve();
  assert(returned); assert.equal(active.getMetrics().backingBytes, 0); active.close();
}

// Interrupted producers release on return; a later retry re-verifies from zero.
let first = true;
const retry = createVerifiedCapsuleArtifactStore(capsule, { readArtifact: noWholeRead,
  async *streamArtifact(a, o) {
    if (first) { first = false; yield bytes.slice(0, 17); throw new Error('interrupted transport'); }
    yield* source.streamArtifact(a, o);
  },
});
await assert.rejects(retry.hashArtifact(artifact), /interrupted transport/);
assert.equal(retry.getMetrics().backingBytes, 0);
await retry.hashArtifact(artifact); retry.close();

// Closing a pending acquisition rejects even if the injected producer stalls.
const waiting = Promise.withResolvers();
const entered = Promise.withResolvers();
const closed = createVerifiedCapsuleArtifactStore(capsule, { readArtifact: noWholeRead,
  async *streamArtifact() { entered.resolve(); yield await waiting.promise; },
});
const pending = closed.hashArtifact(artifact);
await entered.promise; closed.close();
await assert.rejects(pending, /closed/);
waiting.resolve(bytes.subarray(0, 7)); await Promise.resolve();
assert.equal(closed.getMetrics().backingBytes, 0);

// Overlapping sessions retain independent cancellation while publishing safely.
const owner = createCapsuleArtifactBacking();
const controller = new AbortController();
const a = createVerifiedCapsuleArtifactStore(capsule, source, { signal: controller.signal }, owner);
const b = createVerifiedCapsuleArtifactStore(capsule, source, {}, owner);
const cancelled = a.hashArtifact(artifact); const surviving = b.hashArtifact(artifact);
setTimeout(() => controller.abort(new Error('cancel only A')), 0);
await assert.rejects(cancelled, /cancel only A/); a.close();
await surviving;
assert.deepEqual(await b.readArtifactRange(artifact, 123, 9), bytes.subarray(123, 132));
b.close();
console.log('capsule-artifact-stream: passed');
