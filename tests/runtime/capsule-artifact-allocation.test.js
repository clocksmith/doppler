import assert from 'node:assert/strict';
import { createVerifiedCapsuleArtifactStore } from '../../src/client/runtime/verified-capsule-artifact-store.js';
import { hashBytesSha256 } from '../../src/formats/canonical-hash.js';

const bytes = new Uint8Array(4097).fill(31);
const artifact = { artifactId: 'weights', path: 'weights.bin', sizeBytes: bytes.length, hash: hashBytesSha256(bytes) };
const capsule = { artifacts: [artifact] };
for (const boundary of ['owned-copy', 'hash-workspace']) {
  let reads = 0;
  let acquired = false;
  const source = { async readArtifact() { acquired = true; reads++; return bytes; } };
  const store = createVerifiedCapsuleArtifactStore(capsule, source);
  const from = Uint8Array.from;
  const words = globalThis.Uint32Array;
  if (boundary === 'owned-copy') Uint8Array.from = function(input, ...args) {
    if (input === bytes) throw new RangeError('Injected owned-copy allocation failure');
    return from.call(this, input, ...args);
  };
  else globalThis.Uint32Array = new Proxy(words, { construct(target, args) {
    if (acquired) throw new RangeError('Injected hash-workspace allocation failure');
    return Reflect.construct(target, args);
  } });
  try {
    await assert.rejects(store.readArtifact(artifact), new RegExp(`Injected ${boundary} allocation failure`));
    assert.equal(store.getMetrics().retainedBytes, 0);
  } finally { Uint8Array.from = from; globalThis.Uint32Array = words; }
  assert.deepEqual(await store.readArtifact(artifact), bytes, 'A failed acquisition must not poison a later retry');
  assert.equal(reads, 2);
  store.close();
  assert.equal(store.getMetrics().retainedBytes, 0);
  assert.throws(() => store.readArtifact(artifact), /closed/);
}

const store = createVerifiedCapsuleArtifactStore(capsule, { async readArtifact() { return bytes; } });
await store.hashArtifact(artifact);
const slice = Uint8Array.prototype.slice;
Uint8Array.prototype.slice = function(...args) {
  if (this.length === bytes.length) throw new RangeError('Injected returned-slice allocation failure');
  return slice.apply(this, args);
};
try { await assert.rejects(store.readArtifactRange(artifact, 1, 9), /Injected returned-slice allocation failure/); }
finally { Uint8Array.prototype.slice = slice; }
assert.deepEqual(await store.readArtifactRange(artifact, 1, 9), bytes.subarray(1, 10));
store.close();
assert.equal(store.getMetrics().retainedBytes, 0);
console.log('capsule-artifact-allocation: passed (deterministic allocation faults)');
