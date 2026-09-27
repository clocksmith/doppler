#!/usr/bin/env node
// Offline component isolation and abstention experiments, outside the shipped app.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { prepareEvaluation, evaluateSearchCorpus } from './evaluate-document-search-corpus.js';

export const digest = bytes => createHash('sha256').update(bytes).digest('hex');

export function captureModelIdentity(capture) {
  return digest(JSON.stringify({ models: capture.models ?? capture.sources,
    dimension: capture.dimension, scoreContract: capture.policy?.scoreContract ?? 'signed-application-contract' }));
}

export function assessRelevance(score, policy, binding) {
  if (policy == null || policy.status !== 'qualified') return { outcome: 'unassessed' };
  assert.equal(policy.binding, binding, 'Abstention policy does not match this search configuration.');
  assert.equal(policy.rule, 'highest-score-at-least');
  assert(Number.isFinite(policy.threshold), 'Finite relevance threshold required.');
  assert(score == null || Number.isFinite(score), 'Non-finite relevance score.');
  return score != null && score >= policy.threshold
    ? { outcome: 'match' }
    : { outcome: 'abstain', message: 'No sufficiently relevant result found' };
}

export function decisionMetrics(rows, threshold) {
  assert(Number.isFinite(threshold));
  const answerable = rows.filter(row => row.answerable);
  const unanswered = rows.filter(row => !row.answerable);
  assert(answerable.length && unanswered.length, 'Both answerable and no-answer cases are required.');
  const accepted = row => row.results.length > 0 && row.results[0].score >= threshold;
  const useful = answerable.filter(row => accepted(row) && row.rankingCorrect).length;
  const falseAccepts = unanswered.filter(accepted).length;
  return { usefulAccepted: useful, answerableCount: answerable.length,
    usefulAcceptance: useful / answerable.length, falseAccepts, noAnswerCount: unanswered.length,
    falseAcceptance: falseAccepts / unanswered.length,
    rankingTop1: answerable.filter(row => row.rankingCorrect).length / answerable.length };
}

export function calibrateThreshold(rows, contract) {
  assert(contract.maximumFalseAcceptance === 0, 'This experiment requires zero calibration false acceptances.');
  const scores = rows.flatMap(row => row.results.length ? [row.results[0].score] : []);
  assert(scores.length && scores.every(Number.isFinite));
  const max = Math.max(...scores);
  const above = max + Math.max(Number.MIN_VALUE, Math.abs(max) * Number.EPSILON);
  assert(Number.isFinite(above) && above > max);
  const thresholds = [...new Set([...scores, above])].sort((a, b) => a - b);
  const options = thresholds.map(threshold => ({ threshold, ...decisionMetrics(rows, threshold) }))
    .filter(row => row.falseAcceptance <= contract.maximumFalseAcceptance)
    .sort((a, b) => b.usefulAcceptance - a.usefulAcceptance || a.threshold - b.threshold);
  const selected = options[0];
  return { ...selected, calibrationPassed: selected.usefulAcceptance >= contract.minimumUsefulAcceptance };
}

export function replayModels(embeddingCapture, rerankerCapture, binding) {
  const vectors = new Map(embeddingCapture.embeddings.map(row => [row.text, row.vector]));
  const scores = new Map(rerankerCapture.reranking.map(row => [row.query,
    new Map(row.scores.map(score => [score.text, score.score]))]));
  return { dimension: embeddingCapture.dimension, embeddingIdentity: binding,
    minimumRerankScore: -Number.MAX_VALUE,
    embedding: { async embed({ text }) { assert(vectors.has(text), `Missing captured embedding: ${text}`);
      return { embedding: vectors.get(text) }; } },
    reranker: { async rerank({ query, documents }) { return { evidence: {
      scores: documents.map((text, index) => {
        const score = scores.get(query)?.get(text); assert(Number.isFinite(score), 'Missing finite captured score.');
        return { index, score };
      }) } }; } },
  };
}

export async function componentMatrix(corpus, captures, identities) {
  const prepared = prepareEvaluation(corpus);
  const pairs = [];
  for (const embedding of ['incumbent', 'minilm']) {
    for (const reranker of ['incumbent', 'minilm']) {
      const binding = digest(JSON.stringify({ embedding: identities[embedding], reranker: identities[reranker],
        search: corpus.search, preprocessing: corpus.preprocessing }));
      const result = await evaluateSearchCorpus(corpus, prepared,
        replayModels(captures[embedding], captures[reranker], binding));
      const answerable = result.rows.filter(row => row.answerable);
      const categories = Object.fromEntries([...new Set(answerable.map(row => row.category))].map(category => {
        const rows = answerable.filter(row => row.category === category);
        return [category, { total: rows.length, retrieved: rows.filter(row => row.candidateRecall).length,
          rankedCorrect: rows.filter(row => row.rankingCorrect).length }];
      }));
      pairs.push({ embedding, reranker, binding, categories, rows: result.rows,
        rankingCorrect: answerable.filter(row => row.rankingCorrect).length,
        candidateRecall: answerable.filter(row => row.candidateRecall).length,
        answerableCount: answerable.length, abstention: 'unassessed', releaseQualified: false });
    }
  }
  // Identical exhaustive passage list, including positives. Never a retrieval-success metric.
  const fixedCandidateReranking = Object.fromEntries(['incumbent', 'minilm'].map(name => {
    const scores = new Map(captures[name].reranking.map(row => [row.query, new Map(row.scores.map(s => [s.text, s.score]))]));
    return [name, prepared.queries.filter(q => q.relevantIds.length).map(query => {
      const ranked = prepared.documents.map(doc => ({ id: doc.originalDocumentId,
        score: scores.get(query.text)?.get(doc.text) })).sort((a, b) => b.score - a.score);
      assert(ranked.every(row => Number.isFinite(row.score)));
      return { query: query.id, topId: ranked[0].id, correct: query.relevantIds.includes(ranked[0].id) };
    })];
  }));
  return { pairs, fixedCandidateReranking, fixedCandidateScope: 'All frozen passages supplied; not end-to-end retrieval.' };
}

async function main(filename) {
  const config = JSON.parse(await fs.readFile(filename, 'utf8'));
  const corpusBytes = await fs.readFile(config.corpusPath);
  assert.equal(digest(corpusBytes), JSON.parse(await fs.readFile(config.freezePath, 'utf8')).sha256);
  const captures = {}; const identities = {}; const captureHashes = {};
  for (const name of ['incumbent', 'minilm']) {
    const bytes = await fs.readFile(config.captures[name]);
    captures[name] = JSON.parse(bytes); captureHashes[name] = digest(bytes);
    identities[name] = captureModelIdentity(captures[name]);
  }
  const result = await componentMatrix(JSON.parse(corpusBytes), captures, identities);
  const report = { schema: 'doppler.search-component-isolation/v1', config,
    corpusSha256: digest(corpusBytes), modelIdentities: identities, captureHashes,
    probeSha256: digest(await fs.readFile(new URL(import.meta.url))), ...result };
  await fs.writeFile(config.outputPath, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify(report.pairs.map(({ embedding, reranker, rankingCorrect, candidateRecall, answerableCount }) =>
    ({ embedding, reranker, rankingCorrect, candidateRecall, answerableCount }))));
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) await main(process.argv[2]);
