import assert from 'node:assert/strict';
import { createDocumentSearchController } from '../../examples/document-search/controller.js';

// Synthetic storage implementation for headless controller test
function createMockStore() {
  const files = new Map();
  return {
    async readFile(path) {
      if (!files.has(path)) throw new DOMException(`File not found: ${path}`, 'NotFoundError');
      return files.get(path).slice();
    },
    async readText(path) {
      if (!files.has(path)) throw new DOMException(`File not found: ${path}`, 'NotFoundError');
      return new TextDecoder().decode(files.get(path));
    },
    async writeFile(path, data) {
      const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : data;
      files.set(path, new Uint8Array(bytes));
    },
    async deleteFile(path) {
      return files.delete(path);
    },
    _files: files,
  };
}

const embeddingIdentity = { schema: 'doppler.capsule/v3', semanticRoot: 'sha256:emb-root', envelopeDigest: 'sha256:emb-env' };
const rerankerIdentity = { schema: 'doppler.capsule/v3', semanticRoot: 'sha256:rerank-root', envelopeDigest: 'sha256:rerank-env' };

const baseConfig = {
  models: [
    {
      role: 'embedding',
      capsuleUrl: 'http://localhost/emb/capsule.json',
      storageId: 'emb-store',
      identity: embeddingIdentity,
      options: { trustedSigners: {}, acceptedTargetPlanDigests: {}, releaseTrustedSigners: {} },
    },
    {
      role: 'reranker',
      capsuleUrl: 'http://localhost/rerank/capsule.json',
      storageId: 'rerank-store',
      identity: rerankerIdentity,
      options: { trustedSigners: {}, acceptedTargetPlanDigests: {}, releaseTrustedSigners: {} },
    },
  ],
  search: {
    dimension: 2,
    candidateCount: 2,
    queryPrefix: 'query: ',
    documentPrefix: '',
  },
};

// 1. Coalesced Init Promise
{
  const stores = { 'emb-store': createMockStore(), 'rerank-store': createMockStore(), documents: createMockStore() };
  let openCount = 0;
  const controller = createDocumentSearchController({
    config: baseConfig,
    storeFor: async id => stores[id],
    openCapsule: async () => {
      openCount++;
      await new Promise(r => setTimeout(r, 10));
      return { close: async () => {} };
    },
    fetchArtifact: async () => new Uint8Array([1, 2, 3]),
    fetch: async () => ({
      ok: true,
      json: async () => ({ artifacts: [] }),
    }),
    authorizeRecord: () => true,
  });

  const p1 = controller.initialize({ install: true });
  const p2 = controller.initialize({ install: true });
  assert.equal(p1, p2, 'Concurrent initialize() calls return the exact same promise');
  const result = await p1;
  assert.deepEqual(result.models, ['embedding', 'reranker']);
  assert.equal(openCount, 2, 'Opened each model exactly once despite concurrent init calls');
  await controller.close();
}

// 2. Partial Failure Cleanup: If Model 2 fails, Model 1's session is closed deterministically
{
  const stores = { 'emb-store': createMockStore(), 'rerank-store': createMockStore(), documents: createMockStore() };
  let model1Closed = false;
  let model1Opened = false;
  const controller = createDocumentSearchController({
    config: baseConfig,
    storeFor: async id => stores[id],
    openCapsule: async capsule => {
      if (!model1Opened) {
        model1Opened = true;
        return {
          close: async () => {
            model1Closed = true;
          },
        };
      }
      throw new Error('Model 2 initialization failed');
    },
    fetchArtifact: async () => new Uint8Array([1, 2, 3]),
    fetch: async () => ({
      ok: true,
      json: async () => ({ artifacts: [] }),
    }),
    authorizeRecord: () => true,
  });

  await assert.rejects(
    controller.initialize({ install: true }),
    /Model 2 initialization failed/,
    'Initialisation rejects when second model fails'
  );
  assert.equal(model1Closed, true, 'Model 1 session was closed when Model 2 failed');
  assert.equal(controller.getState().hasSessions, false);
}

// 3. Disposal during pending init
{
  const stores = { 'emb-store': createMockStore(), 'rerank-store': createMockStore(), documents: createMockStore() };
  let sessionClosed = false;
  const controller = createDocumentSearchController({
    config: baseConfig,
    storeFor: async id => stores[id],
    openCapsule: async () => {
      await new Promise(r => setTimeout(r, 50));
      return { close: async () => { sessionClosed = true; } };
    },
    fetchArtifact: async () => new Uint8Array([1, 2, 3]),
    fetch: async () => {
      await new Promise(r => setTimeout(r, 20));
      return { ok: true, json: async () => ({ artifacts: [] }) };
    },
    authorizeRecord: () => true,
  });

  const initPromise = controller.initialize({ install: true });
  // Immediately dispose
  await controller.dispose();
  await assert.rejects(initPromise, /AbortError|Controller disposed/i);
  assert.equal(controller.getState().isDisposed, true);
}

// 4. Separate Loading vs Search Cancellation & Retry
{
  const stores = { 'emb-store': createMockStore(), 'rerank-store': createMockStore(), documents: createMockStore() };
  let sessionsClosed = 0;
  const controller = createDocumentSearchController({
    config: baseConfig,
    storeFor: async id => stores[id],
    openCapsule: async () => ({
      async embed(request) {
        return { embedding: [1, 0] };
      },
      async rerank(request) {
        // Slow rerank to allow cancellation
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(resolve, 100);
          request.signal?.addEventListener('abort', () => {
            clearTimeout(timeout);
            reject(request.signal.reason);
          }, { once: true });
        });
        return { evidence: { scores: [{ index: 0, score: 1 }] } };
      },
      async close() { sessionsClosed++; },
    }),
    fetchArtifact: async () => new Uint8Array([1, 2, 3]),
    fetch: async () => ({ ok: true, json: async () => ({ artifacts: [] }) }),
    authorizeRecord: () => true,
  });

  await controller.initialize({ install: true });
  await controller.indexDocuments([
    { id: 'doc1', title: 'Doc 1', text: 'Hello world', mediaType: 'text/plain' },
  ]);

  // Launch search and cancel it
  const searchPromise = controller.search('hello');
  controller.cancelSearch(new DOMException('Query aborted by user', 'AbortError'));
  await assert.rejects(searchPromise, /Query aborted by user|AbortError/);

  assert.equal(sessionsClosed, 0, 'Sessions remain intact after search cancellation');
  assert.equal(controller.getState().hasSessions, true);

  // Immediate retry succeeds without re-initializing
  const retryController = createDocumentSearchController({
    config: baseConfig,
    storeFor: async id => stores[id],
    openCapsule: async () => ({
      async embed(request) { return { embedding: [1, 0] }; },
      async rerank(request) { return { evidence: { scores: [{ index: 0, score: 1 }] } }; },
      async close() { sessionsClosed++; },
    }),
    fetchArtifact: async () => new Uint8Array([1, 2, 3]),
    fetch: async () => ({ ok: true, json: async () => ({ artifacts: [] }) }),
    authorizeRecord: () => true,
  });
  await retryController.initialize({ install: true });
  await retryController.indexDocuments([
    { id: 'doc1', title: 'Doc 1', text: 'Hello world', mediaType: 'text/plain' },
  ]);
  const retryResult = await retryController.search('hello');
  assert.equal(retryResult.results.length, 1);
  assert.equal(retryResult.results[0].document.id, 'doc1');
  await retryController.close();
}

// 5. Monotonic Query ID & Late-Result Suppression
{
  const stores = { 'emb-store': createMockStore(), 'rerank-store': createMockStore(), documents: createMockStore() };
  let searchDelayMs = 0;
  const controller = createDocumentSearchController({
    config: baseConfig,
    storeFor: async id => stores[id],
    openCapsule: async () => ({
      async embed(request) { return { embedding: [1, 0] }; },
      async rerank(request) {
        if (searchDelayMs > 0) {
          await new Promise(r => setTimeout(r, searchDelayMs));
        }
        return { evidence: { scores: [{ index: 0, score: 1 }] } };
      },
      async close() {},
    }),
    fetchArtifact: async () => new Uint8Array([1, 2, 3]),
    fetch: async () => ({ ok: true, json: async () => ({ artifacts: [] }) }),
    authorizeRecord: () => true,
  });

  await controller.initialize({ install: true });
  await controller.indexDocuments([
    { id: 'doc1', title: 'Doc 1', text: 'Doc one', mediaType: 'text/plain' },
  ]);

  // Query 1 is delayed; Query 2 is immediate
  searchDelayMs = 30;
  const q1Promise = controller.search('first query');
  searchDelayMs = 0;
  const q2Promise = controller.search('second query');

  const q2Result = await q2Promise;
  assert.equal(q2Result.superseded, undefined);
  assert.equal(q2Result.results.length, 1);

  const q1Result = await q1Promise;
  assert.equal(q1Result.superseded, true, 'Late query 1 result was marked superseded');
  assert.equal(q1Result.results.length, 0);

  await controller.close();
}

// 6. Incremental Indexing and Rebuild
{
  const stores = { 'emb-store': createMockStore(), 'rerank-store': createMockStore(), documents: createMockStore() };
  let embedCalls = 0;
  const controller = createDocumentSearchController({
    config: baseConfig,
    storeFor: async id => stores[id],
    openCapsule: async () => ({
      async embed(request) {
        embedCalls++;
        return { embedding: [0.5, 0.5] };
      },
      async rerank(request) { return { evidence: { scores: [{ index: 0, score: 1 }] } }; },
      async close() {},
    }),
    fetchArtifact: async () => new Uint8Array([1, 2, 3]),
    fetch: async () => ({ ok: true, json: async () => ({ artifacts: [] }) }),
    authorizeRecord: () => true,
  });

  await controller.initialize({ install: true });

  const initialDocs = [
    { id: 'a', title: 'A', text: 'Text A', mediaType: 'text/plain' },
    { id: 'b', title: 'B', text: 'Text B', mediaType: 'text/plain' },
  ];
  await controller.indexDocuments(initialDocs);
  assert.equal(embedCalls, 2);

  // Add document C without changing A or B
  const updatedDocs = [
    { id: 'a', title: 'A', text: 'Text A', mediaType: 'text/plain' },
    { id: 'b', title: 'B', text: 'Text B', mediaType: 'text/plain' },
    { id: 'c', title: 'C', text: 'Text C', mediaType: 'text/plain' },
  ];
  await controller.indexDocuments(updatedDocs);
  assert.equal(embedCalls, 3, 'Only newly added document triggered embed() call');
  assert.equal(controller.getIndex().documents.length, 3);

  // Rebuild forces re-embedding all documents
  await controller.rebuild();
  assert.equal(embedCalls, 6, 'Rebuild re-computed embeddings for all documents');

  await controller.close();
}

// 7. Retained Index Incompatibility
{
  const stores = { 'emb-store': createMockStore(), 'rerank-store': createMockStore(), documents: createMockStore() };
  // Pre-seed documents store with an incompatible index
  const incompatibleIndex = {
    schema: 'doppler.document-search-index/v1',
    embeddingIdentity: { semanticRoot: 'sha256:different-root' },
    dimension: 2,
    documentPrefix: '',
    documents: [{ id: 'x', title: 'X', text: 'X', mediaType: 'text/plain', vector: [0, 1] }],
  };
  const indexBytes = new TextEncoder().encode(JSON.stringify(incompatibleIndex));
  const digest = 'sha256:' + Buffer.from(await crypto.subtle.digest('SHA-256', indexBytes)).toString('hex');
  await stores.documents.writeFile('index.json', JSON.stringify({ index: incompatibleIndex, digest }));

  const controller = createDocumentSearchController({
    config: baseConfig,
    storeFor: async id => stores[id],
    openCapsule: async () => ({
      async embed() { return { embedding: [1, 0] }; },
      async rerank() { return { evidence: { scores: [] } }; },
      async close() {},
    }),
    fetchArtifact: async () => new Uint8Array([1, 2, 3]),
    fetch: async () => ({ ok: true, json: async () => ({ artifacts: [] }) }),
    authorizeRecord: () => true,
  });

  const result = await controller.initialize({ install: true });
  assert.equal(result.indexInvalidated, true, 'Index was marked invalidated due to release difference');
  assert.equal(controller.getState().indexInvalidated, true);
  assert.equal(controller.getState().hasIndex, false);

  await controller.close();
}

console.log('document-search-controller.test: passed (lifecycle coordination and headless state verification)');
