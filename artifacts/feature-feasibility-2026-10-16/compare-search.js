// Fresh lexical baseline against retained physical model results, not a GPU run.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { prepareEvaluation } from '../../tools/evaluate-document-search-corpus.js';

const root = new URL('../document-search-compact-2026-09-26/', import.meta.url);
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const corpusBytes = await fs.readFile(new URL('corpus.json', root));
const corpus = JSON.parse(corpusBytes);
const frozen = JSON.parse(await fs.readFile(new URL('corpus-freeze.json', root)));
assert.equal(digest(corpusBytes), frozen.sha256, 'Original corpus must remain frozen');
const prepared = prepareEvaluation(corpus);
const modelBytes = await fs.readFile(new URL('incumbent-final-screen.json', root));
const retained = JSON.parse(modelBytes);
const k1 = 1.2, b = 0.75;
const stopwords = new Set(('a an and are as at be been but by can do does for from had has have how '
  + 'i if in into is it its me my of on or our should than that the their them then there these '
  + 'they this to was we were what when where which who why will with would you your').split(' '));
const tokenize = text => (text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [])
  .filter(term => !stopwords.has(term));
const documents = prepared.documents.map(document => ({ ...document,
  terms: tokenize(document.title + ' ' + document.text) }));
const averageLength = documents.reduce((sum, document) => sum + document.terms.length, 0)
  / documents.length;
const frequencies = new Map();
for (const document of documents) {
  for (const term of new Set(document.terms)) frequencies.set(term, (frequencies.get(term) ?? 0) + 1);
}
const rows = prepared.queries.map(query => {
  const terms = new Set(tokenize(query.text));
  const ranked = documents.map(document => {
    let score = 0;
    for (const term of terms) {
      const count = document.terms.filter(value => value === term).length;
      const frequency = frequencies.get(term) ?? 0;
      if (count) score += Math.log(1 + (documents.length - frequency + 0.5) / (frequency + 0.5))
        * count * (k1 + 1) / (count + k1 * (1 - b + b * document.terms.length / averageLength));
    }
    return { documentId: document.originalDocumentId, score };
  }).sort((left, right) => right.score - left.score);
  const model = retained.rows.find(row => row.id === query.id);
  assert(model, 'Each query must have a retained model observation');
  return { id: query.id, query: query.text, relevantIds: query.relevantIds,
    lexicalTopId: ranked[0].documentId,
    lexicalCorrect: query.relevantIds.length ? query.relevantIds.includes(ranked[0].documentId) : null,
    retainedModelTopId: model.topId, retainedModelCorrect: model.rankingCorrect,
    retainedModelElapsedMs: model.elapsedMs };
});
const answerable = rows.filter(row => row.relevantIds.length);
const timings = answerable.map(row => row.retainedModelElapsedMs).sort((left, right) => left - right);
const middle = Math.floor(timings.length / 2);
const medianMs = timings.length % 2 ? timings[middle] : (timings[middle - 1] + timings[middle]) / 2;
const report = {
  schema: 'doppler.feature-search-comparison/v1',
  corpusSha256: digest(corpusBytes), retainedModelReportSha256: digest(modelBytes),
  scope: 'Fresh untuned BM25 ranking versus retained physical Doppler results; no new model execution or controlled timing comparison.',
  lexicalPolicy: { k1, b, stopwords: [...stopwords], input: 'Title plus existing Unicode-window passages',
    tokenizer: 'Lowercase Unicode letters/numbers; fixed stopwords; no stemming',
    ties: 'Stable original passage order', tuning: 'None; this is one baseline, not the strongest possible lexical system' },
  documents: corpus.documents.length, passages: documents.length,
  answerableQueries: answerable.length,
  lexicalTop1Correct: answerable.filter(row => row.lexicalCorrect).length,
  retainedModelTop1Correct: answerable.filter(row => row.retainedModelCorrect).length,
  retainedModelMedianMs: medianMs,
  retainedModelIndexMs: retained.indexMs, retainedModelMetrics: retained.metrics,
  limitations: ['Internal, small corpus; no independent user-preference trial',
    'Original upgrade query annotation is known ambiguous and has not been changed',
    'No remote API quality, latency, or monetary-cost comparison',
    'Ranking does not establish answerability; all three absent-answer cases failed the retained threshold requirement'],
  rows,
};
if (process.argv[2]) await fs.writeFile(process.argv[2], JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
else console.log(JSON.stringify(report, null, 2));
