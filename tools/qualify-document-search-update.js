#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';
import { computeCanonicalSha256, hashBytesSha256 } from '../src/formats/canonical-hash.js';

const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const prior = JSON.parse(await fs.readFile(config.priorQualification, 'utf8'));
assert(prior.passed && prior.physicalExecution);
const fixtureBytes = await fs.readFile(prior.config.fixturePath);
assert.equal(hashBytesSha256(fixtureBytes), prior.fixtureDigest);
const fixture = JSON.parse(fixtureBytes);
const build = JSON.parse(await fs.readFile(path.join(config.nextApplicationDir, 'build-receipt.json'), 'utf8'));
assert.equal(build.installedPackage.sha256, prior.build.installedPackage.sha256);
const oldEmbedding = prior.build.models.find(model => model.role === 'embedding');
const nextEmbedding = build.models.find(model => model.role === 'embedding');
assert.notEqual(nextEmbedding.identity.semanticRoot, oldEmbedding.identity.semanticRoot);
assert.deepEqual(build.models.find(model => model.role === 'reranker'), prior.build.models.find(model => model.role === 'reranker'));
await fs.mkdir(config.outputDir);
const profile = path.join(config.outputDir, 'profile');
await fs.cp(path.join(prior.config.outputDir, 'profile'), profile, { recursive: true });
const report = { schema: 'doppler.document-search-update-qualification/v1', passed: false, config, build,
  priorQualificationDigest: hashBytesSha256(await fs.readFile(config.priorQualification)), fixtureDigest: prior.fixtureDigest,
  requests: [], webSockets: [], phases: {}, scope: 'Explicit application update to a different qualified embedding Capsule; original profile and release streams retained.',
  physicalExecution: false, externalAdoption: false };
let context;
let server;
let phase;
const timer = setTimeout(() => {
  report.timeout = true;
  context?.close().catch(error => { report.timeoutCleanupError = error.message; });
}, prior.config.timeoutMs);
function stage(name) { phase = name; console.log(JSON.stringify({ stage: name })); }
async function launch(offline) {
  context = await chromium.launchPersistentContext(profile, { headless: true, args: prior.config.launchArgs,
    timeout: prior.config.timeoutMs, env: { ...process.env, TMPDIR: prior.config.temporaryDirectory } });
  context.on('request', request => report.requests.push({ phase, url: request.url(), method: request.method(), body: request.postData() }));
  await context.setOffline(offline);
  const page = context.pages()[0] ?? await context.newPage();
  page.setDefaultTimeout(prior.config.timeoutMs);
  page.on('websocket', socket => report.webSockets.push({ phase, url: socket.url() }));
  await page.goto(prior.origin + '/index.html');
  await page.waitForFunction(() => globalThis.documentSearch?.ready);
  return page;
}
async function checkpoints(page, modelConfig) {
  return page.evaluate(async config => {
    const root = await navigator.storage.getDirectory();
    const storage = await root.getDirectoryHandle(config.storage.opfsRootDir);
    const records = {};
    for (const model of config.models) {
      const directory = await storage.getDirectoryHandle(model.storageId);
      records[model.storageId] = await (await (await directory.getFileHandle('release-checkpoint.json')).getFile()).text();
    }
    return records;
  }, modelConfig);
}
async function searchAll(page) {
  const rows = [];
  for (const query of fixture.queries) {
    const result = await page.evaluate(query => globalThis.documentSearch.search(query), query.text);
    assert.equal(result.results[0].document.id, query.expectedTopId);
    rows.push({ queryId: query.id, result });
  }
  return rows;
}
try {
  stage('retained-prior-application');
  let page = await launch(true);
  const priorConfig = await page.evaluate(async () => (await fetch('./models.json')).json());
  assert.equal(priorConfig.models.find(model => model.role === 'embedding').identity.semanticRoot, oldEmbedding.identity.semanticRoot);
  report.priorCheckpoints = await checkpoints(page, priorConfig);
  server = await createStaticFileServer({ rootDir: config.nextApplicationDir, host: '127.0.0.1', port: Number(new URL(prior.origin).port) });
  assert.equal(server.baseUrl, prior.origin);
  stage('explicit-application-update');
  await context.setOffline(false);
  await page.evaluate(async () => {
    const registration = globalThis.documentSearch.registration;
    const controlled = new Promise(resolve => navigator.serviceWorker.addEventListener('controllerchange', resolve, { once: true }));
    await registration.update();
    await controlled;
  });
  await page.reload();
  await page.waitForFunction(() => globalThis.documentSearch?.ready);
  const nextConfig = await page.evaluate(async () => (await fetch('./models.json')).json());
  assert.equal(nextConfig.models.find(model => model.role === 'embedding').identity.semanticRoot, nextEmbedding.identity.semanticRoot);
  report.hardware = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    return Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, adapter.info[key]]));
  });
  assert.deepEqual(report.hardware, prior.hardware);
  stage('explicit-model-installation');
  await page.check('#retention');
  report.installed = await page.evaluate(() => globalThis.documentSearch.install());
  assert.equal(report.installed.indexInvalidated, true);
  report.staleIndexRejected = await page.evaluate(async query => {
    try { await globalThis.documentSearch.search(query); return false; }
    catch (error) { return /document index/.test(error.message); }
  }, fixture.queries[0].text);
  assert(report.staleIndexRejected);
  assert.deepEqual(await checkpoints(page, priorConfig), report.priorCheckpoints);
  report.priorCheckpointsPreserved = true;
  stage('online-rebuild');
  report.rebuilt = await page.evaluate(() => globalThis.documentSearch.rebuildIndex());
  assert.equal(report.rebuilt.documents, fixture.documents.length);
  stage('online-search');
  report.phases.online = await searchAll(page);
  report.physicalExecution = true;
  await page.evaluate(() => globalThis.documentSearch.close());
  await context.close(); context = null;
  await server.close(); server = null;
  report.serverStoppedBeforeRestart = true;
  stage('offline-restart');
  page = await launch(true);
  report.reopened = await page.evaluate(() => globalThis.documentSearch.open());
  assert.equal(report.reopened.indexInvalidated, false);
  stage('offline-search');
  report.phases.offline = await searchAll(page);
  report.parity = report.phases.offline.map((row, index) => {
    const priorResult = report.phases.online[index].result.results;
    assert.deepEqual(row.result.results.map(value => value.document.id), priorResult.map(value => value.document.id));
    const maxScoreDifference = Math.max(...row.result.results.map((value, position) => Math.abs(value.rerankScore - priorResult[position].rerankScore)));
    assert(maxScoreDifference <= fixture.acceptance.offlineScoreMaxAbs);
    return { queryId: row.queryId, maxScoreDifference, rankingDigest: computeCanonicalSha256(row.result.results.map(value => value.document.id)) };
  });
  report.privacy = { onlineRequests: report.requests.filter(request => ['online-rebuild', 'online-search'].includes(request.phase)),
    scope: 'Browser context requests including service workers; online reembedding and search must make zero requests.' };
  assert.equal(report.privacy.onlineRequests.length, 0);
  assert.equal(report.webSockets.length, 0);
  assert.deepEqual(await checkpoints(page, priorConfig), report.priorCheckpoints);
  await page.evaluate(() => globalThis.documentSearch.close());
  report.passed = true;
} catch (error) { report.error = { phase, message: error.message, stack: error.stack }; }
finally {
  clearTimeout(timer);
  for (const resource of [context, server]) {
    try { await resource?.close(); } catch (error) { report.cleanupError = error.message; report.passed = false; }
  }
  await fs.writeFile(path.join(config.outputDir, 'qualification.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
}
console.log(JSON.stringify({ passed: report.passed, error: report.error?.message, outputDir: config.outputDir }));
if (!report.passed) process.exitCode = 1;
