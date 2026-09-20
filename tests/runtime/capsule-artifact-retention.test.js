import assert from 'node:assert/strict';
import { createVerifiedCapsuleArtifactStore, createCapsuleArtifactBacking } from '../../src/client/runtime/verified-capsule-artifact-store.js';
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
await store.readArtifact(a); await store.readArtifact(b);
const returned = await store.readArtifact(a); returned.fill(0);
await store.readArtifact(c);
assert.equal(store.getMetrics().retainedBytes, 8);
assert.equal(store.getMetrics().peakRetainedBytes, 8);
payloads.get('a').fill(0);
assert.deepEqual(await store.readArtifact(a), new Uint8Array([1, 2, 3, 4]), 'retained bytes resist source and caller mutation');
await store.readArtifact(b);
await store.readArtifact(c);
const before = reads;
assert.deepEqual(await store.readArtifact(a), new Uint8Array([1, 2, 3, 4]), 'evicted cache reads use owned immutable backing');
assert.equal(reads, before, 'cache eviction must not reacquire backing');
assert(store.getMetrics().evictions > 0);
assert.equal(store.getMetrics().backingBytes, 12);
store.close();
assert.equal(store.getMetrics().retainedBytes, 0);
assert.equal(store.getMetrics().backingBytes, 0);
const fresh = createVerifiedCapsuleArtifactStore({ artifacts }, source);
await assert.rejects(fresh.readArtifact(a), /hash or size mismatch/, 'another store cannot trust changed source bytes');
assert.equal(fresh.getMetrics().backingBytes, 0);
fresh.close();

const disabled = createVerifiedCapsuleArtifactStore({ artifacts }, source, { maxRetainedArtifactBytes: 0 });
await disabled.hashArtifact(b); await disabled.hashArtifact(b);
assert.equal(disabled.getMetrics().retainedBytes, 0);
assert.equal(disabled.getMetrics().hashedBytes, 4, 'zero cache still keeps its verified backing');
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
assert.equal(closing.getMetrics().backingBytes, 0);

// Signed weight shards never pin an additional cached ArrayBuffer, even with
// the original unlimited cache setting. Hash transfer windows stay bounded.
const weights = new Uint8Array(1024 * 1024 + 1).fill(19);
const weight = { artifactId: 'weights', role: 'weight-shard', sizeBytes: weights.length, hash: hashBytesSha256(weights) };
let weightReads = 0;
const owned = createVerifiedCapsuleArtifactStore({ artifacts: [weight] }, {
  async readArtifact() { weightReads++; return weights; },
});
await owned.hashArtifact(weight);
weights.fill(0);
assert((await owned.readArtifact(weight)).every(value => value === 19));
assert.deepEqual(await owned.readArtifactRange(weight, 3, 7), new Uint8Array(7).fill(19));
assert.equal(weightReads, 1);
assert.equal(owned.getMetrics().retainedBytes, 0);
assert.equal(owned.getMetrics().peakSnapshotBlockBytes, 65536);
assert.equal(owned.getMetrics().hashedBytes, weights.length);
owned.close();
assert.equal(owned.getMetrics().backingFiles, 0);

// Only live stores sharing an explicit host owner can reuse verified backing.
const owner = createCapsuleArtifactBacking();
assert.throws(() => createVerifiedCapsuleArtifactStore({ artifacts }, source, {}, {}), /owned/);
let sharedReads = 0;
const sharedSource = { async readArtifact() { sharedReads++; return payloads.get('b'); } };
const make = () => createVerifiedCapsuleArtifactStore({ artifacts }, sharedSource, { maxRetainedArtifactBytes: 0 }, owner);
const firstOwner = make(); const secondOwner = make();
await firstOwner.hashArtifact(b);
await secondOwner.hashArtifact(b);
assert.equal(sharedReads, 1);
assert.equal(secondOwner.getMetrics().sharedBackingBytes, 4);
firstOwner.close(); firstOwner.close();
payloads.get('b').fill(0);
assert.deepEqual(await secondOwner.readArtifact(b), new Uint8Array([2, 2, 3, 4]));
secondOwner.close();
const afterLastClose = make();
await assert.rejects(afterLastClose.readArtifact(b), /hash or size mismatch/);
assert.equal(sharedReads, 2, 'the final lease releases backing; future stores reverify');
afterLastClose.close();

// A cancelled acquisition cannot cancel a different store sharing its host.
const pendingSource = Promise.withResolvers();
const abort = new AbortController();
const cancelled = createVerifiedCapsuleArtifactStore({ artifacts }, { readArtifact: () => pendingSource.promise }, { signal: abort.signal }, owner);
const survivor = createVerifiedCapsuleArtifactStore({ artifacts }, source, {}, owner);
const cancelledRead = cancelled.readArtifact(c);
await survivor.hashArtifact(c);
abort.abort(new Error('cancel first only'));
await assert.rejects(cancelledRead, /cancel first only/);
cancelled.close(); pendingSource.resolve(payloads.get('c'));
assert.deepEqual(await survivor.readArtifact(c), payloads.get('c'));
survivor.close();

// A pending read must not acquire a second lease if another store publishes the
// same content before that read finishes. Otherwise final close leaks backing.
const racingOwner = createCapsuleArtifactBacking();
const racingRead = Promise.withResolvers();
const slow = createVerifiedCapsuleArtifactStore({ artifacts }, { readArtifact: () => racingRead.promise }, {}, racingOwner);
const fast = createVerifiedCapsuleArtifactStore({ artifacts }, source, {}, racingOwner);
const slowFirst = slow.hashArtifact(c);
await fast.hashArtifact(c);
const slowSecond = slow.hashArtifact(c);
fast.close();
racingRead.resolve(payloads.get('c'));
await Promise.all([slowFirst, slowSecond]);
slow.close();
payloads.get('c').fill(0);
const afterRace = createVerifiedCapsuleArtifactStore({ artifacts }, source, {}, racingOwner);
await assert.rejects(afterRace.hashArtifact(c), /hash or size mismatch/, 'final close must not leave a duplicate lease');
afterRace.close();
console.log('capsule-artifact-retention.test: passed');
