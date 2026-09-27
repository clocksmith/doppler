#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { componentMatrix, decisionMetrics, captureModelIdentity, digest } from './evaluate-search-components.js';

const read = async name => JSON.parse(await fs.readFile(name, 'utf8'));
const config = await read(process.argv[2]);
const policyBytes = await fs.readFile(config.policyPath);
const frozen = JSON.parse(policyBytes);
const corpusBytes = await fs.readFile(config.corpusPath);
const corpus = JSON.parse(corpusBytes);
const split = await read(config.freezePath);
assert.equal(digest(corpusBytes), frozen.confirmationSha256);
assert.equal(digest(corpusBytes), split.sha256);
const developmentBytes = await fs.readFile(config.developmentMatrix);
assert.equal(digest(developmentBytes), frozen.developmentMatrixSha256);
const development = JSON.parse(developmentBytes);
const captures = {}; const identities = {}; const captureHashes = {};
for (const name of ['incumbent', 'minilm']) {
  const bytes = await fs.readFile(config.captures[name]); captures[name] = JSON.parse(bytes);
  identities[name] = captureModelIdentity(captures[name]); captureHashes[name] = digest(bytes);
  assert.equal(identities[name], development.modelIdentities[name], 'Model or scoring identity changed.');
}
const results = [];
for (const [name, ids] of Object.entries(split.corpusSubsets)) {
  const subset = { ...corpus, documents: corpus.documents.filter(document => ids.includes(document.id)) };
  assert.equal(subset.documents.length, ids.length);
  const matrix = await componentMatrix(subset, captures, identities);
  for (const pair of matrix.pairs) {
    const policy = frozen.policies.find(p => p.binding === pair.binding);
    assert(policy, 'No frozen policy for this model/search/preprocessing binding.');
    const metrics = decisionMetrics(pair.rows, policy.threshold);
    results.push({ corpus: name, documentCount: ids.length, embedding: pair.embedding, reranker: pair.reranker,
      binding: pair.binding, threshold: policy.threshold, metrics, categories: pair.categories,
      calibrationPassed: policy.calibrationPassed,
      confirmationPassed: metrics.usefulAcceptance >= frozen.contract.minimumUsefulAcceptance
        && metrics.falseAcceptance <= frozen.contract.maximumFalseAcceptance,
      rows: pair.rows.map(row => ({ ...row,
        experimentalDecision: row.results[0]?.score >= policy.threshold ? 'match' : 'abstain' })) });
  }
}
const qualifications = frozen.policies.map(policy => ({ ...policy,
  status: policy.calibrationPassed && results.filter(row => row.binding === policy.binding).every(row => row.confirmationPassed)
    ? 'qualified' : 'rejected',
  scope: 'Author-created calibration and confirmation corpora only; not universal relevance or shipped starter acceptance.' }));
const report = { schema: 'doppler.search-abstention-confirmation/v1', config, qualifications, results,
  policySha256: digest(policyBytes), corpusSha256: digest(corpusBytes), captureHashes,
  probeSha256: digest(await fs.readFile(new URL(import.meta.url))),
  evaluatorSha256: digest(await fs.readFile(new URL('./evaluate-search-components.js', import.meta.url))),
  shippedStarterChanged: false };
await fs.writeFile(config.outputPath, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify(results.map(({ rows, categories, ...rest }) => rest), null, 2));
