#!/usr/bin/env node
// Candidate screening composes the application's search; it does not implement retrieval.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { createDocumentSearch } from '../examples/document-search/search.js';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');

export function prepareEvaluation(corpus) {
  const policy = corpus.preprocessing;
  assert.equal(policy.version, 'unicode-window-v1');
  assert(Number.isSafeInteger(policy.maxCodePoints) && policy.maxCodePoints > 0);
  assert(Number.isSafeInteger(policy.overlapCodePoints) && policy.overlapCodePoints >= 0
    && policy.overlapCodePoints < policy.maxCodePoints);
  const documents = [];
  const ids = new Set();
  for (const document of corpus.documents) {
    assert(document.id && !ids.has(document.id)); ids.add(document.id);
    const points = Array.from(document.text);
    assert(points.length > 0);
    for (let start = 0; start < points.length;) {
      const end = Math.min(points.length, start + policy.maxCodePoints);
      const text = points.slice(start, end).join('');
      const id = hash(JSON.stringify([policy, document.id, start, end, text]));
      documents.push({ ...document, id, originalDocumentId: document.id,
        startCodePoint: start, endCodePoint: end, text });
      if (end === points.length) break;
      start = end - policy.overlapCodePoints;
    }
  }
  return { schema: 'doppler.search-screen-input/v1', preprocessing: policy, documents, queries: corpus.queries };
}

export async function evaluateSearchCorpus(corpus, prepared, models, createSearch = createDocumentSearch) {
  const search = createSearch({ ...corpus.search, ...models });
  const started = performance.now();
  const index = await search.indexDocuments(prepared.documents);
  const indexMs = performance.now() - started;
  const rows = [];
  for (const query of prepared.queries) {
    const began = performance.now();
    const result = await search.search(index, query.text);
    const top = result.results[0];
    const accepted = top != null && top.rerankScore >= models.minimumRerankScore;
    const relevant = new Set(query.relevantIds);
    rows.push({ id: query.id, category: query.category, answerable: relevant.size > 0,
      topId: top?.document.originalDocumentId ?? null, accepted,
      rankingCorrect: relevant.size ? relevant.has(top?.document.originalDocumentId) : null,
      correct: relevant.size ? accepted && relevant.has(top?.document.originalDocumentId) : !accepted,
      candidateRecall: relevant.size ? result.candidates.some(row => relevant.has(row.document.originalDocumentId)) : null,
      elapsedMs: performance.now() - began,
      results: result.results.map(row => ({ passageId: row.document.id,
        documentId: row.document.originalDocumentId, similarity: row.similarity, score: row.rerankScore })) });
  }
  const rate = selected => selected.length ? selected.filter(row => row.correct).length / selected.length : 0;
  const answerable = rows.filter(row => row.answerable);
  const metrics = { rankingTop1Rate: answerable.filter(row => row.rankingCorrect).length / answerable.length,
    answerableTop1Rate: rate(answerable),
    identifierTop1Rate: rate(rows.filter(row => row.category === 'identifier')),
    longDocumentTop1Rate: rate(rows.filter(row => row.category === 'long-document')),
    noAnswerFalsePositives: rows.filter(row => !row.answerable && row.accepted).length };
  const acceptance = corpus.acceptance;
  return { passed: metrics.answerableTop1Rate >= acceptance.minimumAnswerableTop1Rate
      && metrics.identifierTop1Rate >= acceptance.requiredIdentifierTop1Rate
      && metrics.longDocumentTop1Rate >= acceptance.requiredLongDocumentTop1Rate
      && metrics.noAnswerFalsePositives <= acceptance.maximumNoAnswerFalsePositives,
    metrics, rows, indexMs };
}

async function main(configPath) {
  const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
  const bytes = await fs.readFile(config.corpusPath);
  const corpus = JSON.parse(bytes);
  const frozen = JSON.parse(await fs.readFile(config.freezePath, 'utf8'));
  assert.equal(hash(bytes), frozen.sha256, 'Frozen corpus identity changed.');
  const prepared = prepareEvaluation(corpus);
  if (config.mode === 'prepare') {
    await fs.writeFile(config.outputPath, JSON.stringify(prepared, null, 2) + '\n', { flag: 'wx' });
    return;
  }
  assert.equal(config.mode, 'source-replay');
  const capturedBytes = await fs.readFile(config.capturePath);
  const capture = JSON.parse(capturedBytes);
  assert.equal(capture.preparedSha256, hash(await fs.readFile(config.preparedPath)));
  assert.deepEqual(JSON.parse(await fs.readFile(config.preparedPath, 'utf8')), prepared);
  const embeddings = new Map(capture.embeddings.map(row => [row.text, row.vector]));
  const scores = new Map(capture.reranking.map(row => [row.query, new Map(row.scores.map(score => [score.text, score.score]))]));
  const result = await evaluateSearchCorpus(corpus, prepared, {
    dimension: capture.dimension, embeddingIdentity: { source: capture.sources[0], preprocessing: corpus.preprocessing },
    embedding: { async embed({ text }) { assert(embeddings.has(text)); return { embedding: embeddings.get(text) }; } },
    reranker: { async rerank({ query, documents }) { return { evidence: { scores: documents.map((text, index) => {
      const score = scores.get(query)?.get(text); assert(Number.isFinite(score)); return { index, score };
    }) } }; } }, minimumRerankScore: capture.policy.minimumRerankScore,
  });
  const report = { schema: 'doppler.search-source-screen/v1', ...result,
    corpusSha256: hash(bytes), captureSha256: hash(capturedBytes), sources: capture.sources,
    policy: capture.policy, physicalDopplerExecution: false, signedCapsuleExecution: false,
    timingScope: 'Replay orchestration only. Source inference timings are in the capture; these are not Run timings.',
    probeSha256: hash(await fs.readFile(new URL(import.meta.url))),
    searchSha256: hash(await fs.readFile(new URL('../examples/document-search/search.js', import.meta.url))) };
  await fs.writeFile(config.outputPath, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ passed: report.passed, metrics: report.metrics, outputPath: config.outputPath }));
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) await main(process.argv[2]);
