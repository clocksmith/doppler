import assert from 'node:assert/strict';
import { prepareEvaluation, evaluateSearchCorpus } from '../../tools/evaluate-document-search-corpus.js';

const corpus = {
  preprocessing: { version: 'unicode-window-v1', maxCodePoints: 4, overlapCodePoints: 1 },
  search: { candidateCount: 2, queryPrefix: '', documentPrefix: '' },
  acceptance: { minimumAnswerableTop1Rate: 1, requiredIdentifierTop1Rate: 1,
    requiredLongDocumentTop1Rate: 1, maximumNoAnswerFalsePositives: 0 },
  documents: [{ id: 'one', title: 'First', mediaType: 'text/plain', text: 'abc😀defgh' }],
  queries: [{ id: 'id', text: 'identifier', relevantIds: ['one'], category: 'identifier' },
    { id: 'tail', text: 'tail', relevantIds: ['one'], category: 'long-document' },
    { id: 'none', text: 'absent', relevantIds: [], category: 'no-answer' }],
};
const prepared = prepareEvaluation(corpus);
assert.equal(prepared.documents.map((p, i) => i ? Array.from(p.text).slice(1).join('') : p.text).join(''), corpus.documents[0].text);
assert.deepEqual(prepareEvaluation(corpus), prepared);
assert.equal(prepared.documents.at(-1).endCodePoint, Array.from(corpus.documents[0].text).length);
const renamed = structuredClone(corpus); renamed.documents[0].title = 'Renamed';
assert.deepEqual(prepareEvaluation(renamed).documents.map(p => p.id), prepared.documents.map(p => p.id));
assert.throws(() => prepareEvaluation({ ...corpus, preprocessing: { ...corpus.preprocessing, overlapCodePoints: 4 } }));
const models = { dimension: 2, embeddingIdentity: 'fixture', minimumRerankScore: 0,
  embedding: { async embed() { return { embedding: [1, 0] }; } },
  reranker: { async rerank({ query, documents }) { return { evidence: {
    scores: documents.map((_, index) => ({ index, score: query === 'absent' ? -2 : 2 })) } }; } },
};
const result = await evaluateSearchCorpus(corpus, prepared, models);
assert.equal(result.passed, true);
const falsePositive = await evaluateSearchCorpus(corpus, prepared, { ...models, minimumRerankScore: -3 });
assert.equal(falsePositive.passed, false);
assert.equal(falsePositive.metrics.noAnswerFalsePositives, 1);
