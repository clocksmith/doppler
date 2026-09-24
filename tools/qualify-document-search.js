#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';
import { hashBytesSha256 } from '../src/formats/canonical-hash.js';
import { checkInterruptedInstallation, checkDamagedInstallation, checkIncompatibleIndex, checkDeviceLoss } from './document-search-recovery-checks.js';
import { observeDocumentSearchGpu, readDocumentSearchGpuObservation } from './document-search-gpu-observation.js';
import { verifyInstalledSearchBuild } from './document-search-installed-build.js';
import { checkSearchLifecycle } from './document-search-lifecycle-checks.js';

const [configPath] = process.argv.slice(2);
const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
const fixtureBytes = await fs.readFile(config.fixturePath);
const fixture = JSON.parse(fixtureBytes);
const build = JSON.parse(await fs.readFile(path.join(config.applicationDir, 'build-receipt.json'), 'utf8'));
await fs.mkdir(config.outputDir);
const report = { schema: 'doppler.offline-document-search-qualification/v1', passed: false, stage: 'launch',
  generatedAt: new Date().toISOString(), config, fixtureDigest: hashBytesSha256(fixtureBytes), build,
  physicalExecution: false, externalAdoption: false, requests: [], webSockets: [], logs: [], phases: {} };
report.profileFilesystem = { type: (await fs.statfs(config.outputDir)).type,
  note: 'Filesystem backing is part of this receipt; installation timings are not transferable across storage devices.' };
report.probes = {};
for (const name of ['qualify-document-search.js', 'document-search-recovery-checks.js', 'document-search-gpu-observation.js',
  'document-search-installed-build.js', 'document-search-lifecycle-checks.js', 'check-document-search-starter.js']) {
  report.probes[name] = hashBytesSha256(await fs.readFile(fileURLToPath(new URL(name, import.meta.url))));
}
function stage(name) { report.stage = name; console.log(JSON.stringify({ stage: name })); }
const tokens = text => text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
function incumbent(query) {
  const corpus = fixture.documents.map(document => tokens(document.text));
  const averageLength = corpus.reduce((sum, values) => sum + values.length, 0) / corpus.length;
  return corpus.map((values, index) => {
    let score = 0;
    for (const term of new Set(tokens(query))) {
      const frequency = values.filter(value => value === term).length;
      const containing = corpus.filter(document => document.includes(term)).length;
      const idf = Math.log(1 + (corpus.length - containing + 0.5) / (containing + 0.5));
      const { k1, b } = fixture.incumbent;
      score += idf * frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * values.length / averageLength));
    }
    return { id: fixture.documents[index].id, score, index };
  }).sort((a, b) => b.score - a.score || a.index - b.index);
}
let context;
let server;
let timer;
async function launch(offline, url, profileName = 'profile') {
  context = await chromium.launchPersistentContext(path.join(config.outputDir, profileName), {
    headless: true, args: config.launchArgs, timeout: config.timeoutMs, channel: config.channel,
    env: { ...process.env, TMPDIR: config.temporaryDirectory } });
  report.browserVersion = context.browser().version();
  context.on('request', request => report.requests.push({ stage: report.stage, url: request.url(), method: request.method(), body: request.postData() }));
  await context.setOffline(offline);
  const page = context.pages()[0] ?? await context.newPage();
  page.setDefaultTimeout(config.timeoutMs);
  await page.addInitScript(observeDocumentSearchGpu);
  page.on('websocket', socket => report.webSockets.push({ stage: report.stage, url: socket.url() }));
  page.on('console', message => report.logs.push({ stage: report.stage, type: message.type(), text: message.text() }));
  page.on('pageerror', error => report.logs.push({ stage: report.stage, type: 'pageerror', text: error.message }));
  await page.goto(url);
  await page.waitForFunction(() => globalThis.documentSearch?.ready);
  const hardware = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No physical WebGPU adapter available');
    return { ...Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, adapter.info[key]])),
      features: [...adapter.features].sort(), maxBufferSize: adapter.limits.maxBufferSize };
  });
  assert.equal(hardware.vendor, config.requiredVendor); assert.equal(hardware.isFallbackAdapter, false);
  report.hardware = hardware;
  return page;
}
async function searchAll(page) {
  const results = [];
  for (const query of fixture.queries) {
    const start = performance.now();
    const result = await page.evaluate(text => globalThis.documentSearch.search(text), query.text);
    const bm25 = incumbent(query.text);
    results.push({ query, elapsedMs: performance.now() - start, result, incumbent: bm25,
      modelPassed: result.results[0]?.document.id === query.expectedTopId,
      incumbentPassed: bm25[0]?.id === query.expectedTopId });
  }
  return results;
}
try {
  if (build.schema === 'doppler.installed-document-search-build/v1') {
    stage('installed-identity');
    report.installedIdentity = await verifyInstalledSearchBuild(config.applicationDir, build);
    stage('acquisition-preflight');
    assert.deepEqual(build.unavailableArtifacts, [], 'Publish all exact model files before physical qualification');
    const { createServer } = await import(pathToFileURL(path.join(config.applicationDir, 'server.js')).href);
    const instance = createServer();
    await new Promise(resolve => instance.listen(0, '127.0.0.1', resolve));
    server = { baseUrl: 'http://127.0.0.1:' + instance.address().port,
      close: () => new Promise(resolve => instance.close(resolve)) };
  } else server = await createStaticFileServer({ rootDir: config.applicationDir, host: '127.0.0.1', port: 0 });
  const url = server.baseUrl + '/index.html';
  report.origin = server.baseUrl;
  timer = setTimeout(() => context?.close().catch(error => report.logs.push({ timeoutCleanup: error.message })), config.timeoutMs);
  stage('online-install');
  let page = await launch(false, url, config.recovery === true ? 'interruption-profile' : 'profile');
  await page.check('#retention');
  if (config.recovery === true) {
    stage('interruption-and-quota');
    report.interruptedInstallation = await checkInterruptedInstallation(page, context, server.baseUrl);
    await context.close(); context = null;
    stage('online-install');
    page = await launch(false, url);
    await page.check('#retention');
  }
  report.installationCacheState = 'Fresh browser profile; interruption and quota probes use a separate profile.';
  const start = performance.now();
  report.install = await page.evaluate(() => globalThis.documentSearch.install());
  report.installMs = performance.now() - start;
  report.installGpu = await page.evaluate(readDocumentSearchGpuObservation);
  report.residentModels = await page.evaluate(() => globalThis.documentSearch.controller.getState().sessionRoles.sort());
  assert.deepEqual(report.residentModels, ['embedding', 'reranker']);
  stage('index');
  const indexStart = performance.now();
  report.index = await page.evaluate(documents => globalThis.documentSearch.indexDocuments(documents), fixture.documents);
  report.indexMs = performance.now() - indexStart;
  stage('online-search');
  report.phases.online = await searchAll(page);
  report.timings = await page.evaluate(() => globalThis.documentSearch.timings);
  report.physicalExecution = true;
  if (config.recovery === true) {
    stage('search-lifecycle');
    report.lifecycle = await checkSearchLifecycle(page, fixture,
      JSON.parse(await fs.readFile(path.join(config.applicationDir, 'models.json'))).storage);
    stage('damaged-cache-repair');
    report.damagedInstallation = await checkDamagedInstallation(page, fixture);
  }
  report.onlineObservations = await page.evaluate(() => globalThis.documentSearch.observations);
  const artifactReads = report.onlineObservations.flatMap(event => event.acquisition ?? []);
  report.artifactCosts = Object.fromEntries(['acquisitionMs', 'verificationMs', 'storageReadMs', 'storageWriteMs'].map(key =>
    [key, artifactReads.reduce((sum, read) => sum + (read[key] ?? 0), 0)]));
  const closeStart = performance.now();
  await page.evaluate(() => globalThis.documentSearch.close());
  report.closeMs = performance.now() - closeStart;
  assert.equal(await page.evaluate(() => globalThis.documentSearch.controller.getState().hasSessions), false);
  if (report.installedIdentity) assert.deepEqual(await verifyInstalledSearchBuild(config.applicationDir, build), report.installedIdentity);
  await context.close(); context = null;
  await server.close(); server = null;
  report.serverStoppedBeforeRestart = true;
  stage('offline-restart');
  page = await launch(true, url);
  const reopenStart = performance.now();
  report.reopen = await page.evaluate(() => globalThis.documentSearch.open());
  report.reopenMs = performance.now() - reopenStart;
  report.reopenGpu = await page.evaluate(readDocumentSearchGpuObservation);
  stage('offline-search');
  report.phases.offline = await searchAll(page);
  report.offlineObservations = await page.evaluate(() => globalThis.documentSearch.observations);
  report.offlineParity = report.phases.offline.map((row, index) => {
    const prior = report.phases.online[index];
    return { queryId: row.query.id, sameRanking: JSON.stringify(row.result.results.map(result => result.document.id))
      === JSON.stringify(prior.result.results.map(result => result.document.id)),
    maxScoreDifference: Math.max(...row.result.results.map((result, position) => Math.abs(result.rerankScore - prior.result.results[position].rerankScore))) };
  });
  const texts = [...fixture.documents.map(document => document.text), ...fixture.queries.map(query => query.text)];
  report.privacy = { nonGetRequests: report.requests.filter(request => request.method !== 'GET'),
    contentTransmissions: report.requests.filter(request => texts.some(text => (request.url + (request.body ?? '')).includes(text))),
    onlineSearchRequests: report.requests.filter(request => ['index', 'online-search', 'search-lifecycle'].includes(request.stage)),
    scope: 'Browser context requests (including service workers) and page WebSockets on this corpus. Online indexing and search require zero requests; no external adoption claim.' };
  assert.equal(report.privacy.nonGetRequests.length, 0); assert.equal(report.privacy.contentTransmissions.length, 0);
  assert.equal(report.privacy.onlineSearchRequests.length, 0); assert.equal(report.webSockets.length, 0);
  assert.equal(report.phases.online.filter(row => row.modelPassed).length, fixture.acceptance.requiredTop1);
  assert.equal(report.phases.offline.filter(row => row.modelPassed).length, fixture.acceptance.requiredTop1);
  assert(report.offlineParity.every(row => row.sameRanking && row.maxScoreDifference <= fixture.acceptance.offlineScoreMaxAbs));
  assert(report.offlineObservations.filter(event => event.acquisition).every(event => event.acquisition.every(read => read.source === 'storage')));
  if (config.recovery === true) {
    stage('offline-index-invalidation');
    report.indexInvalidation = await checkIncompatibleIndex(page, fixture);
    stage('offline-device-loss');
    report.deviceLoss = await checkDeviceLoss(page, fixture);
  }
  report.physicalExecution = true;
  await page.evaluate(() => globalThis.documentSearch.close());
  report.passed = true; report.stage = 'complete';
} catch (error) { report.error = { message: error.message, stack: error.stack, evidence: error.evidence }; }
finally {
  clearTimeout(timer);
  const errors = [];
  for (const resource of [context, server]) { try { await resource?.close(); } catch (error) { errors.push(error.message); } }
  report.cleanup = { passed: !errors.length, errors }; report.passed &&= report.cleanup.passed;
  await fs.writeFile(path.join(config.outputDir, 'qualification.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
}
console.log(JSON.stringify({ passed: report.passed, stage: report.stage, error: report.error?.message, outputDir: config.outputDir }));
if (!report.passed) process.exitCode = 1;
