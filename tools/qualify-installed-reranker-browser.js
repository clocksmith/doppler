#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';
import { evaluateRerankReference } from '../src/config/rerank-reference.js';
import { createRetainedReleaseCheckpoint } from './retained-release-checkpoint.js';
import { rendererPids, rendererRss } from './browser-renderer-memory.js';
import { assertRerankerRunCoverage, buildRerankerReferenceSchedule } from './reranker-reference-schedule.js';

const hash = bytes => `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const config = await read(process.argv[2]);
assert.equal(config.schema, 'doppler.installed-browser-reranker-qualification/v1');
for (const key of ['repeatRuns', 'timeoutMs', 'sampleIntervalMs']) assert(Number.isSafeInteger(config[key]) && config[key] > 0);
const runSchedule = buildRerankerReferenceSchedule(config.sampling ?? null);
const bundle = await read(path.join(config.packageBundlePath, 'receipt.json')); assert(bundle.passed);
assert.equal(hash(await fs.readFile(path.join(config.packageBundlePath, bundle.package.filename))), `sha256:${bundle.package.sha256}`);
const installedRoot = path.join(config.packageBundlePath, 'consumer/node_modules/doppler-gpu');
const capsule = await read(path.join(config.capsuleRoot, 'distribution/capsule-v3.json'));
const options = await read(path.join(config.capsuleRoot, 'current-open-options.json'));
const references = [];
for (const input of config.references) {
  const bytes = await fs.readFile(input.path); assert.equal(hash(bytes), input.digest);
  references.push(JSON.parse(bytes.toString('utf8')));
}
assert(references.length > 0);
await fs.mkdir(config.outputDir);
const checkpoint = await createRetainedReleaseCheckpoint(
  path.join(config.capsuleRoot, 'release-checkpoints.json'), config.outputDir, capsule.semanticRoot);
options.releasePolicy = { ...options.releasePolicy, now: new Date().toISOString() };
const report = { schema: 'doppler.installed-browser-reranker-qualification-result/v1', passed: false,
  config, installedPackage: bundle.package, startedAtUtc: new Date().toISOString(), phases: [], logs: [], requests: [], samples: [],
  qualifierDigest: hash(await fs.readFile(new URL(import.meta.url))), capsuleDigest: hash(await fs.readFile(path.join(config.capsuleRoot, 'distribution/capsule-v3.json'))),
  checkpointWriterDigest: hash(await fs.readFile(new URL('./retained-release-checkpoint.js', import.meta.url))),
  scheduleDigest: hash(await fs.readFile(new URL('./reranker-reference-schedule.js', import.meta.url))),
  memorySamplerDigest: hash(await fs.readFile(new URL('./browser-renderer-memory.js', import.meta.url))),
  sampling: config.sampling ?? null, runSchedule,
  claimAllowed: false, scope: 'Installed signed Capsule on browser WebGPU against unchanged source references; no paired performance claim.' };
let server, context, timer, pending;
try {
  server = await createStaticFileServer({ rootDir: installedRoot, host: '127.0.0.1', port: 0,
    staticMounts: [{ urlPrefix: '/retained-capsule', rootDir: path.join(config.capsuleRoot, 'distribution') }] });
  const profile = path.join(config.outputDir, 'profile');
  const startupStarted = performance.now();
  context = await chromium.launchPersistentContext(profile, { headless: true, args: config.launchArgs,
    timeout: config.timeoutMs, env: { ...process.env, TMPDIR: config.temporaryDirectory } });
  const page = context.pages()[0]; page.setDefaultTimeout(config.timeoutMs);
  report.startup = { scope: 'Browser launch through model readiness and first completed query; local artifact verification and server setup excluded.' };
  await page.exposeFunction('qualificationModelReady', () => {
    report.startup.modelReadyMs ??= performance.now() - startupStarted;
  });
  await page.exposeFunction('qualificationFirstResult', () => {
    report.startup.firstResultMs ??= performance.now() - startupStarted;
  });
  report.browser = await (await context.newCDPSession(page)).send('Browser.getVersion');
  page.on('console', message => report.logs.push({ type: message.type(), text: message.text() }));
  page.on('pageerror', error => report.logs.push({ type: 'pageerror', text: error.message }));
  await page.route('**/*', route => {
    const url = new URL(route.request().url()); report.requests.push(url.href);
    if (url.origin !== server.baseUrl || route.request().method() !== 'GET') return route.abort();
    if (url.pathname === '/qualification') return route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>Installed browser reranker</title>' });
    return route.continue();
  });
  await page.goto(server.baseUrl + '/qualification');
  report.hardware = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    return Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, adapter.info[key]]));
  });
  assert.equal(report.hardware.vendor, config.requiredVendor); assert.equal(report.hardware.isFallbackAdapter, false);
  await page.exposeFunction('persistReleaseCheckpoint', checkpoint.persist);
  const pids = await rendererPids(profile);
  timer = setInterval(() => {
    if (pending) return;
    pending = rendererRss(pids).then(bytes => report.samples.push({ at: performance.now(), rendererRssBytes: bytes }))
      .catch(error => { report.memoryError = error.message; }).finally(() => { pending = null; });
  }, config.sampleIntervalMs);
  for (let repeat = 0; repeat < config.repeatRuns; repeat++) {
    const observation = await page.evaluate(async ({ options, references, runSchedule }) => {
      const { openCapsule } = await import('/src/client/doppler-api.browser.js');
      const { destroyDevice } = await import('/src/gpu/device.js');
      const started = performance.now();
      const session = await openCapsule(location.origin + '/retained-capsule/capsule-v3.json', {
        ...options, persistReleaseCheckpoint: globalThis.persistReleaseCheckpoint });
      try {
        const result = { modelLoadMs: performance.now() - started, identity: session.capsuleIdentity,
          targetPlanDigest: session.selectedTargetPlanDigest, scoringConfig: session.manifest.inference.rerank, runs: [] };
        await globalThis.qualificationModelReady();
        const application = options.releaseEvents.at(-1).release.application;
        for (const sample of runSchedule) {
          for (const [referenceIndex, reference] of references.entries()) {
            const start = performance.now();
            const receipt = await session.rerank({ application, ...reference.input });
            result.runs.push({ ...sample, referenceIndex, durationMs: performance.now() - start, scores: receipt.evidence.scores, receipt });
            if (result.runs.length === 1) await globalThis.qualificationFirstResult();
          }
        }
        return result;
      } finally { try { await session.close(); } finally { destroyDevice(); } }
    }, { options, references, runSchedule });
    assertRerankerRunCoverage(observation.runs, runSchedule, references.length);
    const comparisons = observation.runs.map(run => {
      const reference = references[run.referenceIndex];
      return { phase: run.phase, iteration: run.iteration, referenceIndex: run.referenceIndex,
        ...evaluateRerankReference(reference, { input: { query: run.receipt.evidence.query, documents: run.receipt.evidence.documents },
          scoringConfig: observation.scoringConfig, outputs: run.scores }) };
    });
    report.phases.push({ repeat, observation, comparisons });
    await fs.writeFile(path.join(config.outputDir, 'progress.json'), JSON.stringify(report.phases, null, 2));
    assert(comparisons.every(comparison => comparison.passed), 'Frozen source reference failed.');
  }
  clearInterval(timer); await pending;
  report.peakRendererRssBytes = Math.max(...report.samples.map(sample => sample.rendererRssBytes));
  assert(Number.isFinite(report.peakRendererRssBytes) && !report.memoryError);
  report.passed = true;
} catch (error) { report.error = { message: error.message, stack: error.stack }; }
finally {
  clearInterval(timer); await pending; report.cleanup = [];
  for (const close of [() => context?.close(), () => server?.close(), () => checkpoint.verifySourceUnchanged()]) {
    try { await close(); } catch (error) { report.cleanup.push(error.message); }
  }
  report.passed &&= report.cleanup.length === 0;
  report.completedAtUtc = new Date().toISOString();
  await fs.writeFile(path.join(config.outputDir, 'qualification.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
}
console.log(JSON.stringify({ passed: report.passed, outputDir: config.outputDir, error: report.error }));
if (!report.passed) process.exitCode = 1;
