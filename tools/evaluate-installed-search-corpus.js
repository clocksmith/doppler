#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { prepareEvaluation, evaluateSearchCorpus } from './evaluate-document-search-corpus.js';

const read = async filename => JSON.parse(await fs.readFile(filename, 'utf8'));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const config = await read(process.argv[2]);
const corpusBytes = await fs.readFile(config.corpusPath);
const corpus = JSON.parse(corpusBytes);
assert.equal(hash(corpusBytes), (await read(config.freezePath)).sha256);
const build = await read(path.join(config.applicationDir, 'build-receipt.json'));
for (const [prefix, assets] of [['', build.assets], ['node_modules/doppler-gpu', build.runtimeAssets]]) {
  assert(assets.length > 0);
  for (const asset of assets) {
    const bytes = await fs.readFile(path.join(config.applicationDir, prefix, asset.path));
    assert.equal(bytes.length, asset.sizeBytes); assert.equal(hash(bytes), asset.sha256);
  }
}
assert.equal(hash(await fs.readFile(path.join(config.applicationDir, 'vendor', build.installedPackage.filename))), build.installedPackage.sha256);
const report = { schema: 'doppler.installed-search-screen/v1', passed: false, config,
  corpusSha256: hash(corpusBytes), installedPackage: build.installedPackage,
  buildReceiptSha256: hash(await fs.readFile(path.join(config.applicationDir, 'build-receipt.json'))),
  probeSha256: hash(await fs.readFile(new URL(import.meta.url))),
  evaluatorSha256: hash(await fs.readFile(new URL('./evaluate-document-search-corpus.js', import.meta.url))),
  searchSha256: hash(await fs.readFile(path.join(config.applicationDir, 'search.js'))),
  physicalDopplerExecution: false, signedCapsuleExecution: false, externalAdoption: false };
const previousFetch = globalThis.fetch;
let app;
let candidate;
try {
  globalThis.fetch = async () => { throw new Error('Network disabled for search screening.'); };
  const { createNodeDocumentSearch } = await import(pathToFileURL(path.join(config.applicationDir, 'node.js')));
  const { createDocumentSearch } = await import(pathToFileURL(path.join(config.applicationDir, 'search.js')));
  app = await createNodeDocumentSearch({ storageDir: config.storageDir });
  const adapter = await navigator.gpu.requestAdapter();
  const info = adapter.info;
  report.hardware = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, info[key]]));
  assert.equal(info.vendor.toLowerCase(), config.requiredVendor); assert.equal(info.isFallbackAdapter, false);
  const began = performance.now();
  await app.controller.openRetained();
  report.openMs = performance.now() - began;
  const models = app.config.models;
  const embedding = models.find(model => model.role === 'embedding');
  const reranker = models.find(model => model.role === 'reranker');
  const sessions = app.controller.getSessions();
  assert.deepEqual(Object.keys(sessions).sort(), ['embedding', 'reranker']);
  report.models = models.map(({ role, identity }) => ({ role, identity }));
  let embeddingIdentity = embedding.identity;
  if (config.embeddingCandidateDir) {
    const manifestBytes = await fs.readFile(path.join(config.embeddingCandidateDir, 'manifest.json'));
    const manifest = JSON.parse(manifestBytes);
    assert.equal(manifest.hashAlgorithm, 'sha256', 'New candidates require standard SHA-256 shard identities.');
    for (const shard of manifest.shards) {
      const bytes = await fs.readFile(path.join(config.embeddingCandidateDir, shard.filename));
      assert.equal(bytes.length, shard.size); assert.equal(hash(bytes), shard.hash);
    }
    await sessions.embedding.close();
    const { load } = await import(pathToFileURL(path.join(config.applicationDir, 'node_modules/doppler-gpu/src/client/doppler-api.js')));
    const started = performance.now();
    candidate = await load({ url: pathToFileURL(config.embeddingCandidateDir + '/').href }, { runtimeConfig: config.candidateRuntimeConfig });
    report.candidateLoadMs = performance.now() - started;
    embeddingIdentity = { manifestSha256: hash(manifestBytes), artifactIdentity: manifest.artifactIdentity };
    report.embeddingCandidate = embeddingIdentity;
    sessions.embedding = { async embed({ text, options }) { return candidate.embedWithEvidence(text, options); } };
  }
  Object.assign(report, await evaluateSearchCorpus(corpus, prepareEvaluation(corpus), {
    ...sessions, dimension: app.config.search.dimension, embeddingIdentity,
    embeddingApplication: embedding.application, rerankerApplication: reranker.application,
    minimumRerankScore: config.minimumRerankScore,
  }, createDocumentSearch));
  report.physicalDopplerExecution = true; report.signedCapsuleExecution = !candidate;
} catch (error) { report.error = { message: error.message, stack: error.stack }; }
finally {
  try { try { await candidate?.unload(); } finally { await app?.close(); } report.closed = true; }
  catch (error) { report.closed = false; report.passed = false; report.cleanupError = error.message; }
  globalThis.fetch = previousFetch;
  report.processHighWaterRssBytes = process.resourceUsage().maxRSS * 1024;
  await fs.writeFile(config.outputPath, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
}
console.log(JSON.stringify({ passed: report.passed, metrics: report.metrics, error: report.error, outputPath: config.outputPath }));
if (report.error || !report.closed) process.exitCode = 1;
