import assert from 'node:assert/strict';
import { createMemoryStore } from '../../src/storage/backends/memory-store.js';
import { createModelReadSession } from '../../src/storage/model-read-session.js';

const backend = createMemoryStore({ maxBytes: 1024 });
await backend.openModel('a');
await backend.writeFile('same.bin', new Uint8Array([1, 2, 3]));
const a = await createModelReadSession(backend, 'a', 2);
await backend.openModel('b');
await backend.writeFile('same.bin', new Uint8Array([4, 5, 6]));
const b = await createModelReadSession(backend, 'b', 2);
assert.deepEqual([...new Uint8Array(await a.readFile('same.bin'))], [1, 2, 3]);
const stream = b.streamRange('same.bin')[Symbol.asyncIterator]();
assert.deepEqual([...(await stream.next()).value], [4, 5]);
await a.close();
assert.deepEqual([...(await stream.next()).value], [6]);
assert.equal((await stream.next()).done, true);
await assert.rejects(a.readFile('same.bin'), /closed/);
await assert.rejects(b.readRange('same.bin', -1, 1), /non-negative integer/);
await assert.rejects(b.readRange('same.bin', 0, 1.5), /non-negative integer/);
await b.close();
assert.deepEqual([...new Uint8Array(await backend.readFile('same.bin'))], [4, 5, 6]);
assert.equal(backend.getCurrentModelId(), 'b');

// Download writers, resume readers, manifests and cancellation have the same
// ownership rule as immutable read handles, including identical shard names.
const stores = await import('../../src/storage/shard-manager.js');
const { createRDRRManifestFixture } = await import('../helpers/rdrr-manifest-fixture.js');
const { downloadDistributedShard } = await import('../../src/storage/distribution-transport.js');
const payload = new Uint8Array([7, 8, 9]);
const hash = await stores.computeHash(payload, 'sha256');
const manifest = createRDRRManifestFixture();
manifest.hashAlgorithm = 'sha256';
manifest.shards = [{ index: 0, filename: 'shared.bin', size: payload.byteLength, hash, offset: 0 }];
const [storeA, storeB] = await Promise.all([
  stores.openModelStoreSession('download-a', manifest), stores.openModelStoreSession('download-b', manifest),
]);
const originalFetch = globalThis.fetch;
try {
  let fetches = 0;
  globalThis.fetch = async () => { fetches++; return new Response(payload); };
  const request = { algorithm: 'sha256', expectedHash: hash, expectedSize: payload.byteLength,
    expectedManifestVersionSet: 'test-v1', writeToStore: true, enableSourceCache: false,
    distributionConfig: { sourceOrder: ['http'], requiredContentEncoding: null } };
  await Promise.all([storeA, storeB].map(store => downloadDistributedShard(
    'https://storage.test/model', 0, manifest.shards[0], { ...request, store }
  )));
  assert.equal(fetches, 2, 'delivery deduplication cannot merge different storage owners');
  await stores.openModelStore('unrelated-default');
  for (const store of [storeA, storeB]) {
    assert.deepEqual(new Uint8Array(await store.loadShard(0, { verify: true })), payload);
    await store.deleteShard(0);
  }
  const gate = Promise.withResolvers();
  const started = Promise.withResolvers();
  const cancelA = new AbortController(), cancelB = new AbortController();
  let startedCount = 0;
  globalThis.fetch = async (_url, { signal }) => {
    if (++startedCount === 2) started.resolve();
    if (signal === cancelA.signal) return new Promise((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(new DOMException('cancel A', 'AbortError')), { once: true });
    });
    await gate.promise;
    return new Response(payload);
  };
  const pendingA = downloadDistributedShard('https://storage.test/model', 0, manifest.shards[0],
    { ...request, store: storeA, signal: cancelA.signal });
  const rejectedA = assert.rejects(pendingA, error => error.name === 'AbortError');
  const pendingB = downloadDistributedShard('https://storage.test/model', 0, manifest.shards[0],
    { ...request, store: storeB, signal: cancelB.signal });
  await started.promise;
  cancelA.abort();
  await rejectedA;
  await storeA.close();
  gate.resolve();
  await pendingB;
  assert.deepEqual(new Uint8Array(await storeB.loadShard(0, { verify: true })), payload);
  await assert.rejects(storeA.loadShard(0), /closed/);
  await assert.rejects(downloadDistributedShard('https://storage.test/model', 0, manifest.shards[0],
    { ...request, distributionConfig: { sourceOrder: ['p2p'], p2p: { enabled: true } } }), /host-supplied transport/);
  const result = await downloadDistributedShard('https://storage.test/model', 0, manifest.shards[0],
    { ...request, transport: async (_url, _index, _info, options) => {
      assert.equal(options.transport, undefined);
      return { injected: true };
    } });
  assert.equal(result.injected, true);
} finally {
  globalThis.fetch = originalFetch;
  await storeA.close();
  await storeB.close();
  await stores.cleanup();
}
