import assert from 'node:assert/strict';
import { createDocumentSearch } from '../../examples/document-search/search.js';

// Synthetic vectors test index invalidation and candidate-to-document mapping.
const application = { applicationId: 'test' };
let embeddingCalls = 0;
const create = identity => createDocumentSearch({ dimension: 2, candidateCount: 2, queryPrefix: 'query: ', documentPrefix: '',
  embeddingApplication: application, rerankerApplication: application, embeddingIdentity: identity,
  embedding: { async embed(request) { embeddingCalls++; assert.equal(request.application, application);
    return { embedding: request.text === 'second' ? [0, 1] : [1, 0] }; } },
  reranker: { async rerank(request) { assert.deepEqual(request.documents, ['first', 'second']);
    return { evidence: { scores: [{ index: 1, score: 9 }, { index: 0, score: 1 }] } }; } } });
const search = create({ semanticRoot: 'one' });
const index = await search.indexDocuments([{ id: 'a', title: 'First', text: 'first', mediaType: 'text/plain' },
  { id: 'b', title: 'Second', text: 'second', mediaType: 'text/markdown' }]);
assert.equal((await search.search(index, 'which')).results[0].document.id, 'b');
const calls = embeddingCalls;
await assert.rejects(create({ semanticRoot: 'two' }).search(index, 'which'), /incompatible/);
assert.equal(embeddingCalls, calls, 'incompatible index fails before model work');
const cancellation = new AbortController();
cancellation.abort(new Error('Index cancelled'));
await assert.rejects(search.indexDocuments(index.documents, { signal: cancellation.signal }), /Index cancelled/);
assert.equal(embeddingCalls, calls, 'cancelled indexing fails before model work');
const invalid = structuredClone(index); invalid.documents[0].vector[0] = Infinity;
await assert.rejects(search.search(invalid, 'which'), /Invalid retained/);
await assert.rejects(search.indexDocuments([{ id: 'c', title: 'PDF', text: 'data', mediaType: 'application/pdf' }]), /Only text/);
console.log('document-search.test: passed (synthetic vectors, no retrieval-quality claim)');
