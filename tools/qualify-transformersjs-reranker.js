#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';
import { evaluateRerankReference } from '../src/config/rerank-reference.js';
import { rendererPids, rendererRss } from './browser-renderer-memory.js';
import { assertRerankerRunCoverage, buildRerankerReferenceSchedule } from './reranker-reference-schedule.js';

const hash = bytes => `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const config = await read(process.argv[2]);
assert.equal(config.schema, 'doppler.transformersjs-reranker-qualification/v1');
assert(Number.isSafeInteger(config.repeatRuns) && config.repeatRuns > 0);
assert(Number.isSafeInteger(config.timeoutMs) && config.timeoutMs > 0);
assert(Number.isSafeInteger(config.sampleIntervalMs) && config.sampleIntervalMs > 0);
const runSchedule = buildRerankerReferenceSchedule(config.sampling ?? null);
const acquisition = await read(path.join(config.modelRoot, 'acquisition.json'));
assert.equal(acquisition.repository, config.modelId);
assert.equal(acquisition.revision, config.revision);
for (const file of acquisition.files) {
  const target = path.resolve(config.modelRoot, config.modelId, file.path);
  assert(target.startsWith(path.resolve(config.modelRoot, config.modelId) + path.sep));
  const digest = createHash('sha256');
  for await (const bytes of createReadStream(target)) digest.update(bytes);
  assert.equal(digest.digest('hex'), file.sha256);
}
const references = [];
for (const input of config.references) {
  const bytes = await fs.readFile(input.path); assert.equal(hash(bytes), input.digest);
  references.push(JSON.parse(bytes.toString('utf8')));
}
assert(references.length > 0);
for (const reference of references) assert.deepEqual(reference.scoringConfig, references[0].scoringConfig);
await fs.mkdir(config.outputDir);
const report = { schema: 'doppler.transformersjs-reranker-qualification-result/v1', passed: false,
  config, acquisition, startedAtUtc: new Date().toISOString(), qualifierDigest: hash(await fs.readFile(new URL(import.meta.url))),
  runnerDigest: hash(await fs.readFile('benchmarks/runners/transformersjs-runner.html')),
  scheduleDigest: hash(await fs.readFile(new URL('./reranker-reference-schedule.js', import.meta.url))),
  memorySamplerDigest: hash(await fs.readFile(new URL('./browser-renderer-memory.js', import.meta.url))),
  sampling: config.sampling ?? null, runSchedule,
  packageLockDigest: hash(await fs.readFile('package-lock.json')), phases: [], logs: [], requests: [], samples: [],
  claimAllowed: false, scope: 'Pinned ONNX product path against unchanged source tokens and numerical tolerances.' };
let context, server, timer, pending;
let rejectStartup;
const startupFailure = new Promise((resolve, reject) => { rejectStartup = reject; });
// Attach immediately: browser errors can arrive before navigation finishes.
startupFailure.catch(() => {});
try {
  server = await createStaticFileServer({ rootDir: path.resolve('.'), host: '127.0.0.1', port: 0,
    staticMounts: [{ urlPrefix: '/retained-models', rootDir: config.modelRoot }] });
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
  page.on('console', message => {
    report.logs.push({ type: message.type(), text: message.text() });
    console.log(JSON.stringify(report.logs.at(-1)));
  });
  page.on('pageerror', error => {
    report.logs.push({ type: 'pageerror', text: error.message });
    rejectStartup(error);
  });
  await page.route('**/*', route => {
    const url = new URL(route.request().url()); report.requests.push(url.href);
    return url.origin === server.baseUrl && route.request().method() === 'GET' ? route.continue() : route.abort();
  });
  await page.goto(server.baseUrl + '/benchmarks/runners/transformersjs-runner.html?v=4&localModelPath=/retained-models/');
  await Promise.race([
    page.waitForFunction(() => typeof window.__runRerankReference === 'function'),
    startupFailure,
  ]);
  report.hardware = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    return Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, adapter.info[key]]));
  });
  assert.equal(report.hardware.vendor, config.requiredVendor); assert.equal(report.hardware.isFallbackAdapter, false);
  const pids = await rendererPids(profile);
  timer = setInterval(() => {
    if (pending) return;
    pending = rendererRss(pids).then(bytes => report.samples.push({ at: performance.now(), rendererRssBytes: bytes }))
      .catch(error => { report.memoryError = error.message; }).finally(() => { pending = null; });
  }, config.sampleIntervalMs);
  for (let repeat = 0; repeat < config.repeatRuns; repeat++) {
    const observation = await page.evaluate(input => window.__runRerankReference(input), {
      modelId: config.modelId, dtype: config.dtype, format: 'onnx',
      scoringConfig: references[0].scoringConfig, inputs: references.map(reference => reference.input), runSchedule });
    assertRerankerRunCoverage(observation.runs, runSchedule, references.length);
    const comparisons = observation.runs.map(run => {
      const reference = references[run.referenceIndex];
      return { phase: run.phase, iteration: run.iteration, referenceIndex: run.referenceIndex,
        ...evaluateRerankReference(reference, { input: { query: run.query, documents: run.documents },
          scoringConfig: observation.scoringConfig, outputs: run.scores }) };
    });
    report.phases.push({ repeat, observation, comparisons });
    await fs.writeFile(path.join(config.outputDir, 'progress.json'), JSON.stringify(report.phases, null, 2));
    assert(comparisons.every(comparison => comparison.passed), 'Frozen source reference failed.');
    assert.equal(observation.executionProviderMode, 'webgpu-only');
    assert.equal(observation.fallbackUsed, false); assert.equal(observation.executionProviderFallbackUsed, false);
  }
  clearInterval(timer); await pending;
  report.peakRendererRssBytes = Math.max(...report.samples.map(sample => sample.rendererRssBytes));
  assert(Number.isFinite(report.peakRendererRssBytes) && !report.memoryError);
  report.passed = true;
} catch (error) { report.error = { message: error.message, stack: error.stack }; }
finally {
  clearInterval(timer); await pending; report.cleanup = [];
  for (const close of [() => context?.close(), () => server?.close()]) {
    try { await close(); } catch (error) { report.cleanup.push(error.message); }
  }
  report.passed &&= report.cleanup.length === 0;
  report.completedAtUtc = new Date().toISOString();
  await fs.writeFile(path.join(config.outputDir, 'qualification.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
}
console.log(JSON.stringify({ passed: report.passed, outputDir: config.outputDir, error: report.error }));
if (!report.passed) process.exitCode = 1;
