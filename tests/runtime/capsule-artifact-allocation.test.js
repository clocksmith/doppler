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
  const bytesConstructor = globalThis.Uint8Array;
  const words = globalThis.Uint32Array;
  if (boundary === 'owned-copy') globalThis.Uint8Array = new Proxy(bytesConstructor, { construct(target, args) {
    if (acquired && args[0] === bytes.length) throw new RangeError('Injected owned-copy allocation failure');
    return Reflect.construct(target, args);
  } });
  else globalThis.Uint32Array = new Proxy(words, { construct(target, args) {
    if (acquired) throw new RangeError('Injected hash-workspace allocation failure');
    return Reflect.construct(target, args);
  } });
  try {
    await assert.rejects(store.readArtifact(artifact), new RegExp(`Injected ${boundary} allocation failure`));
    assert.equal(store.getMetrics().retainedBytes, 0);
    assert.equal(store.getMetrics().backingBytes, 0);
  } finally { globalThis.Uint8Array = bytesConstructor; globalThis.Uint32Array = words; }
  assert.deepEqual(await store.readArtifact(artifact), bytes, 'A failed acquisition must not poison a later retry');
  assert.equal(reads, 2);
  store.close();
  assert.equal(store.getMetrics().retainedBytes, 0);
  assert.throws(() => store.readArtifact(artifact), /closed/);
}

const store = createVerifiedCapsuleArtifactStore(capsule, { async readArtifact() { return bytes; } });
await store.hashArtifact(artifact);
const array = globalThis.Uint8Array;
globalThis.Uint8Array = new Proxy(array, { construct(target, args) {
  if (args[0] === 9) throw new RangeError('Injected returned-slice allocation failure');
  return Reflect.construct(target, args);
} });
try { await assert.rejects(store.readArtifactRange(artifact, 1, 9), /Injected returned-slice allocation failure/); }
finally { globalThis.Uint8Array = array; }
assert.deepEqual(await store.readArtifactRange(artifact, 1, 9), bytes.subarray(1, 10));
store.close();
assert.equal(store.getMetrics().retainedBytes, 0);

// Close/abort while acquisition is pending cannot publish verification.
for (const abort of [false, true]) {
  const controller = new AbortController();
  const latch = Promise.withResolvers();
  const entered = Promise.withResolvers();
  const closing = createVerifiedCapsuleArtifactStore(capsule, { readArtifact() { entered.resolve(); return latch.promise; } }, { signal: controller.signal });
  try {
    const read = closing.hashArtifact(artifact);
    await entered.promise;
    if (abort) controller.abort(new Error('Injected read cancellation'));
    else closing.close();
    latch.resolve(bytes);
    await assert.rejects(read, abort ? /cancellation/ : /closed/);
    assert.equal(closing.getMetrics().backingBytes, 0);
  } finally { closing.close(); }
}

// Independent allocation guard: verification cannot allocate an input-sized
// owned array, even when retention is unlimited. The caller owns this input.
const large = new Uint8Array(1024 * 1024 + 13).fill(7);
const largeArtifact = { artifactId: 'large', role: 'weight-shard', sizeBytes: large.length, hash: hashBytesSha256(large) };
const bounded = createVerifiedCapsuleArtifactStore({ artifacts: [largeArtifact] }, { async readArtifact() { return large; } });
let peak = 0;
globalThis.Uint8Array = new Proxy(array, { construct(target, args) {
  if (typeof args[0] === 'number') {
    peak = Math.max(peak, args[0]);
    if (args[0] > 65536) throw new RangeError('Forbidden whole-input allocation');
  }
  return Reflect.construct(target, args);
} });
try {
  await bounded.hashArtifact(largeArtifact);
  assert.deepEqual(await bounded.readArtifactRange(largeArtifact, 65533, 19), new array(19).fill(7));
  assert.equal(peak, 65536);
} finally { globalThis.Uint8Array = array; bounded.close(); }
console.log('capsule-artifact-allocation: passed (deterministic allocation faults)');
