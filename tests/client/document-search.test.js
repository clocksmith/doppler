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

// Incremental indexing verification
const callsBeforeReuse = embeddingCalls;
const reusedIndex = await search.indexDocuments(
  [{ id: 'a', title: 'First', text: 'first', mediaType: 'text/plain' },
   { id: 'b', title: 'Second', text: 'second', mediaType: 'text/markdown' }],
  index
);
assert.equal(embeddingCalls, callsBeforeReuse, 'unchanged documents reuse prior vectors without embedding calls');
assert.deepEqual(reusedIndex.documents[0].vector, index.documents[0].vector);
assert.deepEqual(reusedIndex.documents[1].vector, index.documents[1].vector);

// Modified document re-embeds only the modified item
const modifiedIndex = await search.indexDocuments(
  [{ id: 'a', title: 'First', text: 'first', mediaType: 'text/plain' },
   { id: 'b', title: 'Second Modified', text: 'second changed', mediaType: 'text/markdown' }],
  reusedIndex
);
assert.equal(embeddingCalls, callsBeforeReuse + 1, 'only modified document triggered an embedding call');
assert.deepEqual(modifiedIndex.documents[0].vector, index.documents[0].vector);

// Removed document is omitted from the resulting index
const reducedIndex = await search.indexDocuments(
  [{ id: 'b', title: 'Second Modified', text: 'second changed', mediaType: 'text/markdown' }],
  modifiedIndex
);
assert.equal(reducedIndex.documents.length, 1);
assert.equal(reducedIndex.documents[0].id, 'b');

// Incompatible index throws DOCUMENT_SEARCH_INDEX_INCOMPATIBLE
const incompatibleIndex = structuredClone(index);
incompatibleIndex.embeddingIdentity = { semanticRoot: 'different' };
await assert.rejects(
  search.indexDocuments([{ id: 'a', title: 'First', text: 'first', mediaType: 'text/plain' }], incompatibleIndex),
  (err) => {
    assert.equal(err.code, 'DOCUMENT_SEARCH_INDEX_INCOMPATIBLE');
    assert.match(err.message, /incompatible/);
    return true;
  },
  'incompatible priorIndex throws DOCUMENT_SEARCH_INDEX_INCOMPATIBLE'
);

console.log('document-search.test: passed (synthetic vectors, no retrieval-quality claim)');

