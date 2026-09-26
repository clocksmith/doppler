import assert from 'node:assert/strict';
import { createVerifiedCapsuleArtifactStore, createCapsuleArtifactBacking } from '../../src/client/runtime/verified-capsule-artifact-store.js';
import { hashBytesSha256 } from '../../src/formats/canonical-hash.js';

for (const artifactHashBackend of ['javascript', 'node-crypto', 'host']) {
  const bytes = new Uint8Array(65539).fill(17);
  const a = { artifactId: 'a', role: 'weight-shard', sizeBytes: bytes.length, hash: hashBytesSha256(bytes) };
  const b = { ...a, artifactId: 'b', hash: hashBytesSha256(new Uint8Array(bytes.length).fill(19)) };
  const capsule = { artifacts: [a, b] };
  let reads = 0;
  const source = { async readArtifact() { reads++; return bytes; } };
  for (const maxVerifiedBackingBytes of [-1, 0.1, Infinity, '4']) {
    assert.throws(() => createVerifiedCapsuleArtifactStore(capsule, source, { maxVerifiedBackingBytes }), /maxVerifiedBackingBytes/);
  }
  const owner = createCapsuleArtifactBacking();
  const store = createVerifiedCapsuleArtifactStore(capsule, source,
    { artifactHashBackend, maxVerifiedBackingBytes: bytes.length }, owner);
  const first = store.readArtifact(a);
  await assert.rejects(store.hashArtifact(b), /maxVerifiedBackingBytes/);
  assert.equal(reads, 1, 'Concurrent acquisitions reserve their entire backing before source I/O.');
  const release = store.releaseBacking();
  assert.deepEqual(await first, bytes, 'Release waits until outstanding readers own their result.');
  await release;
  assert.equal(store.getMetrics().backingBytes, 0);
  assert.equal(store.getMetrics().reservedBackingBytes, 0);
  assert.equal(store.getMetrics().peakReservedAndBackingBytes, bytes.length);
  await store.hashArtifact(a);
  assert.equal(reads, 2, 'A released snapshot must be acquired and authenticated again.');
  const survivor = createVerifiedCapsuleArtifactStore(capsule, source, { artifactHashBackend }, owner);
  await survivor.hashArtifact(a);
  const denied = createVerifiedCapsuleArtifactStore(capsule, source, { maxVerifiedBackingBytes: 0 }, owner);
  await assert.rejects(denied.hashArtifact(a), /maxVerifiedBackingBytes/, 'Shared leases still count against the store budget.');
  denied.close();
  await store.releaseBacking();
  bytes.fill(0);
  assert((await survivor.readArtifact(a)).every(value => value === 17), 'Releasing one lease cannot invalidate another.');
  await survivor.releaseBacking();
  await assert.rejects(store.hashArtifact(a), /hash or size mismatch/);
  assert.equal(store.getMetrics().reservedBackingBytes, 0, 'Failed authentication returns its reservation.');
  store.close(); survivor.close();
}
assert.throws(() => createVerifiedCapsuleArtifactStore({ artifacts: [] }, { readArtifact() {} },
  { artifactHashBackend: 'untrusted-verified' }), /artifactHashBackend/);
console.log('capsule-backing-budget: passed');
