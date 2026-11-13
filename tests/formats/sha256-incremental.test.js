import assert from 'node:assert/strict';
import { createHash, webcrypto } from 'node:crypto';
import { createSha256Hasher, sha256BytesHex, sha256Hex } from '../../src/formats/sha256.js';
import { computeSHA256, createStreamingHasher } from '../../src/storage/shards/integrity.js';

const reference = bytes => createHash('sha256').update(bytes).digest('hex');
const hex = bytes => Buffer.from(bytes).toString('hex');
for (const length of [0, 1, 2, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 1023, 65537]) {
  const backing = Uint8Array.from({ length: length + 17 }, (_, i) => (i * 131 + 29) & 255);
  const bytes = backing.subarray(7, 7 + length);
  const expected = reference(bytes);
  assert.equal(sha256BytesHex(bytes), expected);
  assert.equal(await computeSHA256(bytes), expected);
  for (const chunkSize of [1, 7, 55, 64, 65, 97, 1024]) {
    const hasher = createSha256Hasher();
    const stream = await createStreamingHasher('sha256');
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      const chunk = bytes.slice(offset, offset + chunkSize);
      hasher.update(chunk);
      stream.update(chunk);
      chunk.fill(0); // update must not retain borrowed storage, including a partial block.
      hasher.update(new Uint8Array());
    }
    assert.equal(hasher.digestHex(), expected, `length=${length}, chunk=${chunkSize}`);
    const digest = hasher.digest();
    assert.equal(hex(digest), expected);
    digest.fill(0);
    assert.equal(hasher.digestHex(), expected, 'Digest ownership is independent');
    assert.equal(hex(await stream.finalize()), expected);
    assert.equal(hex(await stream.finalize()), expected);
    hasher.update(new Uint8Array([42]));
    stream.update(new Uint8Array([42]).buffer);
    const extended = createHash('sha256').update(bytes).update(new Uint8Array([42])).digest('hex');
    assert.equal(hasher.digestHex(), extended);
    assert.equal(hex(await stream.finalize()), extended);
  }
}

// Deterministic arbitrary divisions and offset views.
const input = Uint8Array.from({ length: 1024 * 1024 + 59 }, (_, i) => (i * 17 + 3) & 255);
const randomChunks = createSha256Hasher();
let seed = 19;
for (let offset = 0; offset < input.length;) {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  const length = Math.min(1 + seed % 4099, input.length - offset);
  randomChunks.update(input.subarray(offset, offset + length));
  offset += length;
}
assert.equal(randomChunks.digestHex(), reference(input));
for (const value of [null, undefined, '', 'abc', '你好🙂', '\ud800', 42]) {
  assert.equal(sha256Hex(value), reference(String(value ?? '')));
}
assert.throws(() => sha256BytesHex(new ArrayBuffer(1)), /Uint8Array/);
const invalid = createSha256Hasher();
assert.throws(() => invalid.update('invalid'), /Uint8Array/);
assert.equal(invalid.digestHex(), reference(new Uint8Array()));

// Host acceleration preserves the streaming contract, including strict input
// validation and borrowed digest ownership. Browser hosts retain portable SHA.
for (const portable of [false, true]) {
  const hostProcess = globalThis.process;
  let stream;
  try {
    if (portable) globalThis.process = undefined;
    stream = await createStreamingHasher('sha256');
  } finally { globalThis.process = hostProcess; }
  assert.throws(() => stream.update('invalid'), /Uint8Array/);
  assert.equal(hex(await stream.finalize()), reference(new Uint8Array()));
  const bytes = input.subarray(7, 131);
  stream.update(bytes);
  const digest = await stream.finalize();
  assert.equal(hex(digest), reference(bytes));
  digest.fill(0);
  assert.equal(hex(await stream.finalize()), reference(bytes));
  stream.update(new Uint8Array([42]).buffer);
  assert.equal(hex(await stream.finalize()), createHash('sha256').update(bytes).update(new Uint8Array([42])).digest('hex'));
}

// Instrument backing allocations, not caller-owned inputs or zero-copy views.
const originals = { Uint8Array, Uint32Array };
const allocations = [];
for (const name of Object.keys(originals)) {
  globalThis[name] = new Proxy(originals[name], { construct(target, args) {
    const value = Reflect.construct(target, args);
    if (!(args[0] instanceof ArrayBuffer)) {
      allocations.push(value.byteLength);
      assert(value.byteLength <= 256, `Unexpected hashing workspace allocation: ${value.byteLength}`);
    }
    return value;
  } });
}
let workspace;
try {
  allocations.length = 0;
  assert.equal(sha256BytesHex(input), reference(input));
  const oneShot = allocations.reduce((a, b) => a + b, 0);
  allocations.length = 0;
  const streaming = await createStreamingHasher('sha256');
  for (let offset = 0; offset < input.length; offset += 1009) streaming.update(input.subarray(offset, offset + 1009));
  assert.equal(hex(await streaming.finalize()), reference(input));
  const streamed = allocations.reduce((a, b) => a + b, 0);
  assert(oneShot <= 512 && streamed <= 544);
  workspace = { inputBytes: input.length, oneShotTypedArrayBytes: oneShot, streamingTypedArrayBytes: streamed };
} finally { Object.assign(globalThis, originals); }

// Finalization allocation failure must leave the running hash retryable.
const retry = createSha256Hasher();
retry.update(input);
globalThis.Uint8Array = new Proxy(Uint8Array, { construct() { throw new RangeError('Injected padding allocation failure'); } });
try { assert.throws(() => retry.digestHex(), /Injected padding allocation failure/); }
finally { globalThis.Uint8Array = originals.Uint8Array; }
assert.equal(retry.digestHex(), reference(input));

// Cross the 32-bit bit-length boundary without owning a half-gigabyte input.
const chunk = new Uint8Array(1024 * 1024).fill(0xa5);
const long = createSha256Hasher();
const native = createHash('sha256');
for (let i = 0; i < 513; i++) { long.update(chunk); native.update(chunk); }
assert.equal(long.digestHex(), native.digest('hex'));

// Complete-byte storage hashing uses the native primitive when available,
// honors borrowed view extents, and never retries a native failure in JS.
const cryptoDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
let nativeCalls = 0;
const offsetBytes = Uint8Array.from({ length: 97 }, (_, index) => index).subarray(7, 81);
try {
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: { subtle: {
    async digest(algorithm, bytes) {
      nativeCalls++;
      assert.equal(algorithm, 'SHA-256');
      assert.equal(bytes.byteLength, offsetBytes.byteLength);
      return webcrypto.subtle.digest(algorithm, bytes);
    },
  } } });
  assert.equal(await computeSHA256(offsetBytes), reference(offsetBytes));
  assert.equal(await computeSHA256(offsetBytes.slice().buffer), reference(offsetBytes));
  assert.equal(nativeCalls, 2);
  const shared = new Uint8Array(new SharedArrayBuffer(offsetBytes.byteLength));
  shared.set(offsetBytes);
  assert.equal(await computeSHA256(shared), reference(offsetBytes));
  assert.equal(nativeCalls, 2, 'Shared storage is not passed to WebCrypto');
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: undefined });
  assert.equal(await computeSHA256(offsetBytes), reference(offsetBytes));
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: { subtle: {
    digest() { throw new Error('Injected native digest failure'); },
  } } });
  await assert.rejects(computeSHA256(offsetBytes), /Injected native digest failure/);
  await assert.rejects(computeSHA256('invalid'), /Uint8Array/);
} finally {
  if (cryptoDescriptor) Object.defineProperty(globalThis, 'crypto', cryptoDescriptor);
  else delete globalThis.crypto;
}
console.log(JSON.stringify({ test: 'sha256-incremental', passed: true, workspace, longInputBytes: chunk.length * 513 }));
