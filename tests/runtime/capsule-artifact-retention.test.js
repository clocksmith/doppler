import assert from 'node:assert/strict';
import { createVerifiedCapsuleArtifactStore } from '../../src/client/runtime/verified-capsule-artifact-store.js';
import { hashBytesSha256 } from '../../src/formats/canonical-hash.js';

const payloads = new Map(['a', 'b', 'c'].map((id, index) => [id, new Uint8Array([index + 1, 2, 3, 4])]));
const artifacts = [...payloads].map(([artifactId, bytes]) => ({ artifactId, hash: hashBytesSha256(bytes), sizeBytes: bytes.length }));
let reads = 0;
const source = { async readArtifact(artifact) { reads++; return payloads.get(artifact.artifactId); } };
for (const limit of [-1, 0.5, Infinity, '4']) {
  assert.throws(() => createVerifiedCapsuleArtifactStore({ artifacts }, source, { maxRetainedArtifactBytes: limit }), /maxRetainedArtifactBytes/);
}
const store = createVerifiedCapsuleArtifactStore({ artifacts }, source, { maxRetainedArtifactBytes: 8 });
const [a, b, c] = artifacts;
await store.hashArtifact(a); await store.hashArtifact(b);
const returned = await store.readArtifact(a); returned.fill(0);
await store.hashArtifact(c);
assert.equal(store.getMetrics().retainedBytes, 8);
assert.equal(store.getMetrics().peakRetainedBytes, 8);
payloads.get('a').fill(0);
assert.deepEqual(await store.readArtifact(a), new Uint8Array([1, 2, 3, 4]), 'retained bytes resist source and caller mutation');
await store.hashArtifact(b);
await store.hashArtifact(c);
const before = reads;
await assert.rejects(store.readArtifact(a), /hash or size mismatch/, 'evicted bytes are verified again');
assert.equal(reads, before + 1);
assert(store.getMetrics().evictions > 0);
store.close();
assert.equal(store.getMetrics().retainedBytes, 0);

const disabled = createVerifiedCapsuleArtifactStore({ artifacts }, source, { maxRetainedArtifactBytes: 0 });
await disabled.hashArtifact(b); await disabled.hashArtifact(b);
assert.equal(disabled.getMetrics().retainedBytes, 0);
assert.equal(disabled.getMetrics().hashedBytes, 8);
disabled.close();

const latch = Promise.withResolvers();
let pendingReads = 0;
const concurrent = createVerifiedCapsuleArtifactStore({ artifacts }, { readArtifact() { pendingReads++; return latch.promise; } },
  { maxRetainedArtifactBytes: 3 });
const first = concurrent.readArtifact(b); const second = concurrent.readArtifact(b);
latch.resolve(payloads.get('b'));
const values = await Promise.all([first, second]);
assert.equal(pendingReads, 1, 'concurrent reads share verification even when a file exceeds the retention budget');
values[0].fill(0); assert.deepEqual(values[1], payloads.get('b'));
assert.equal(concurrent.getMetrics().retainedBytes, 0);
concurrent.close();

const delayed = Promise.withResolvers();
const closing = createVerifiedCapsuleArtifactStore({ artifacts }, { readArtifact: () => delayed.promise }, { maxRetainedArtifactBytes: 8 });
const opening = closing.readArtifact(b);
closing.close(); delayed.resolve(payloads.get('b'));
await assert.rejects(opening, /closed/);
assert.equal(closing.getMetrics().retainedBytes, 0, 'a late read cannot repopulate a closed store');
console.log('capsule-artifact-retention.test: passed');
