import assert from 'node:assert/strict';
import { createFetchCapsuleArtifactStore } from '../../src/client/runtime/fetch-capsule-artifact-store.js';

const originalFetch = globalThis.fetch;
const artifact = { artifactId: 'weights', path: 'weights.bin', sizeBytes: 2 * 1024 * 1024 + 17 };
const source = createFetchCapsuleArtifactStore('https://example.invalid/capsule.json');
try {
  // A large network enqueue must not dictate the consumer's allocation size.
  globalThis.fetch = async () => new Response(new ReadableStream({ type: 'bytes', start(controller) {
    controller.enqueue(new Uint8Array(artifact.sizeBytes).fill(7)); controller.close();
  } }));
  let total = 0;
  for await (const chunk of source.streamArtifact(artifact, { maxChunkBytes: 65536 })) {
    assert(chunk.buffer.byteLength <= 65536);
    assert(chunk.every(value => value === 7)); total += chunk.length;
  }
  assert.equal(total, artifact.sizeBytes);

  // Non-byte-stream hosts remain usable only if their buffers respect the bound.
  globalThis.fetch = async () => new Response(new ReadableStream({ start(controller) {
    controller.enqueue(new Uint8Array(5).fill(3)); controller.close();
  } }));
  let values = [];
  for await (const chunk of source.streamArtifact({ ...artifact, sizeBytes: 5 }, { maxChunkBytes: 8 })) values.push(...chunk);
  assert.deepEqual(values, [3, 3, 3, 3, 3]);

  let cancelled = false;
  globalThis.fetch = async () => new Response(new ReadableStream({ start(controller) {
    controller.enqueue(new Uint8Array(1024).subarray(0, 5));
  }, cancel() { cancelled = true; } }));
  await assert.rejects(async () => {
    for await (const chunk of source.streamArtifact({ ...artifact, sizeBytes: 5 }, { maxChunkBytes: 8 })) void chunk;
  }, /acquisition limit/);
  assert(cancelled, 'a tiny view cannot hide an oversized backing allocation');

  cancelled = false;
  globalThis.fetch = async () => new Response(new ReadableStream({ type: 'bytes', start(controller) {
    controller.enqueue(new Uint8Array(1024));
  }, cancel() { cancelled = true; } }));
  const stream = source.streamArtifact({ ...artifact, sizeBytes: 1024 }, { maxChunkBytes: 64 });
  await stream.next(); await stream.return();
  assert(cancelled, 'return disposes an incomplete response');
} finally { globalThis.fetch = originalFetch; }
console.log('capsule-http-stream: passed');
