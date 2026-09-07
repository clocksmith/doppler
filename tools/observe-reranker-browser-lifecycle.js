#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';
import { evaluateRerankReference } from '../src/config/rerank-reference.js';
import { createRetainedReleaseCheckpoint } from './retained-release-checkpoint.js';

const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
async function digest(file) {
  const hash = createHash('sha256');
  for await (const bytes of createReadStream(file)) hash.update(bytes);
  return `sha256:${hash.digest('hex')}`;
}
const config = await read(process.argv[2]);
assert.equal(config.schema, 'doppler.reranker-browser-lifecycle/v1');
assert(['doppler', 'transformersjs'].includes(config.engine));
for (const key of ['operationTimeoutMs', 'maxRecoveryMs']) assert(Number.isSafeInteger(config[key]) && config[key] > 0);
const prerequisite = await read(config.qualificationPath);
assert.equal(prerequisite.passed, true, 'Source qualification must pass before lifecycle observation.');
assert.equal(await digest(config.qualificationPath), config.qualificationDigest);
const base = prerequisite.config;
const references = [];
for (const file of base.references) {
  assert.equal(await digest(file.path), file.digest);
  references.push(await read(file.path));
}
const native = config.engine === 'doppler';
let installedRoot, options, capsule;
if (native) {
  const bundle = await read(path.join(base.packageBundlePath, 'receipt.json'));
  assert.equal(await digest(path.join(base.packageBundlePath, bundle.package.filename)), `sha256:${bundle.package.sha256}`);
  assert.deepEqual(bundle.package, prerequisite.installedPackage);
  installedRoot = path.join(base.packageBundlePath, 'consumer/node_modules/doppler-gpu');
  options = await read(path.join(base.capsuleRoot, 'current-open-options.json'));
  capsule = await read(path.join(base.capsuleRoot, 'distribution/capsule-v3.json'));
} else {
  assert.deepEqual(await read(path.join(base.modelRoot, 'acquisition.json')), prerequisite.acquisition);
  for (const file of prerequisite.acquisition.files) {
    const target = path.resolve(base.modelRoot, base.modelId, file.path);
    assert(target.startsWith(path.resolve(base.modelRoot, base.modelId) + path.sep));
    assert.equal(await digest(target), `sha256:${file.sha256}`);
  }
}
await fs.mkdir(config.outputDir);
const report = { schema: 'doppler.reranker-browser-lifecycle-result/v1', passed: false,
  scope: 'Observation completion and frozen source validity; cancellation and device-loss capability flags are separate results.',
  config, prerequisiteDigest: config.qualificationDigest, startedAtUtc: new Date().toISOString(),
  implementation: {}, phases: [], logs: [], requests: [], cleanup: [], claimAllowed: false };
for (const file of ['tools/observe-reranker-browser-lifecycle.js', 'tools/reranker-lifecycle-observation.js',
  'benchmarks/runners/transformersjs-runner.html', 'package-lock.json']) report.implementation[file] = await digest(file);
let server, context, checkpoint;
async function retain(phase) {
  report.phases.push(phase);
  await fs.writeFile(path.join(config.outputDir, 'progress.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ stage: phase.stage }));
}
function comparisons(observation) {
  return observation.runs.map((run, index) => evaluateRerankReference(references[index], {
    input: { query: run.query, documents: run.documents }, scoringConfig: observation.scoringConfig, outputs: run.scores,
  }));
}
async function openPage(name) {
  context = await chromium.launchPersistentContext(path.join(config.outputDir, name), {
    headless: true, args: base.launchArgs, timeout: base.timeoutMs,
    env: { ...process.env, TMPDIR: base.temporaryDirectory } });
  const page = context.pages()[0]; page.setDefaultTimeout(base.timeoutMs);
  report.browser ??= await (await context.newCDPSession(page)).send('Browser.getVersion');
  page.on('console', message => report.logs.push({ phase: name, type: message.type(), text: message.text() }));
  page.on('pageerror', error => report.logs.push({ phase: name, type: 'pageerror', text: error.message }));
  await page.route('**/*', route => {
    const url = new URL(route.request().url()); report.requests.push(url.href);
    if (url.origin !== server.baseUrl || route.request().method() !== 'GET') return route.abort();
    if (url.pathname === '/qualification') return route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>Reranker lifecycle observation</title>' });
    return route.continue();
  });
  if (native) await page.exposeFunction('persistReleaseCheckpoint', checkpoint.persist);
  await page.goto(server.baseUrl + (native ? '/qualification' : '/benchmarks/runners/transformersjs-runner.html?v=4&localModelPath=/retained-models/'));
  if (!native) await page.waitForFunction(() => typeof window.__openRerankLifecycle === 'function');
  await page.evaluate(async ({ native, base, options }) => {
    const { captureRequestedDevices } = await import('/qualification-tools/reranker-lifecycle-observation.js');
    const capture = captureRequestedDevices();
    globalThis.lifecycleDevices = capture.devices;
    try {
      if (native) {
        const { openCapsule } = await import('/src/client/doppler-api.browser.js');
        const session = await openCapsule(location.origin + '/retained-capsule/capsule-v3.json', {
          ...options, persistReleaseCheckpoint: globalThis.persistReleaseCheckpoint });
        const application = options.releaseEvents.at(-1).release.application;
        globalThis.lifecycleRuntime = { scoringConfig: session.manifest.inference.rerank,
          cancellationMode: 'runtime-abort-signal', identity: session.capsuleIdentity,
          rerank: async (input, signal) => {
            const result = await session.rerank({ application, ...input, options: { signal } });
            return { ...input, scores: result.evidence.scores };
          }, close: () => session.close() };
      } else {
        globalThis.lifecycleRuntime = await window.__openRerankLifecycle({ modelId: base.modelId,
          dtype: base.dtype, format: 'onnx', scoringConfig: base.scoringConfig });
      }
    } finally { capture.restore(); }
  }, { native, base: { ...base, scoringConfig: references[0].scoringConfig }, options });
  const hardware = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    return Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, adapter.info[key]]));
  });
  assert.equal(hardware.vendor, base.requiredVendor); assert.equal(hardware.isFallbackAdapter, false);
  if (report.hardware) assert.deepEqual(hardware, report.hardware); else report.hardware = hardware;
  return page;
}
async function verify(page, stage) {
  const observation = await page.evaluate(async inputs => {
    const runtime = globalThis.lifecycleRuntime;
    const runs = [];
    for (const input of inputs) runs.push(await runtime.rerank(input));
    return { runs, scoringConfig: runtime.scoringConfig };
  }, references.map(reference => reference.input));
  const evaluated = comparisons(observation);
  await retain({ stage, observation, comparisons: evaluated });
  assert(evaluated.every(result => result.passed), `${stage}: frozen source reference failed.`);
}
try {
  const mounts = [{ urlPrefix: '/qualification-tools', rootDir: import.meta.dirname }];
  if (native) {
    checkpoint = await createRetainedReleaseCheckpoint(path.join(base.capsuleRoot, 'release-checkpoints.json'), config.outputDir, capsule.semanticRoot);
    options.releasePolicy = { ...options.releasePolicy, now: new Date().toISOString() };
    mounts.push({ urlPrefix: '/retained-capsule', rootDir: path.join(base.capsuleRoot, 'distribution') });
  } else mounts.push({ urlPrefix: '/retained-models', rootDir: base.modelRoot });
  server = await createStaticFileServer({ rootDir: native ? installedRoot : path.resolve('.'),
    host: '127.0.0.1', port: 0, staticMounts: mounts });
  let page = await openPage('initial-profile');
  await verify(page, 'initial');
  const cancellation = await page.evaluate(async ({ input, timeoutMs }) => {
    const { observeSubmittedCancellation } = await import('/qualification-tools/reranker-lifecycle-observation.js');
    const result = await observeSubmittedCancellation(signal => globalThis.lifecycleRuntime.rerank(input, signal), globalThis.lifecycleDevices, timeoutMs);
    globalThis.lifecycleSubmittedDevice = result.device;
    return { ...result.observation, identifiedSubmittedDevice: !!result.device,
      apiMode: globalThis.lifecycleRuntime.cancellationMode };
  }, { input: references[0].input, timeoutMs: config.operationTimeoutMs });
  await retain({ stage: 'cancellation', ...cancellation });
  assert(cancellation.triggeredAfterSubmission && cancellation.identifiedSubmittedDevice);
  assert.notEqual(cancellation.outcome.status, 'timeout');
  if (native) assert(cancellation.cancellationHonored, 'Doppler must honor submitted cancellation.');
  await verify(page, 'after-cancellation');
  const loss = await page.evaluate(async ({ input, timeoutMs }) => {
    const { observeSettled } = await import('/qualification-tools/reranker-lifecycle-observation.js');
    const device = globalThis.lifecycleSubmittedDevice;
    device.destroy();
    const lost = await observeSettled(async () => {
      const info = await device.lost; return { reason: info.reason, message: info.message };
    }, timeoutMs);
    const oldSession = await observeSettled(() => globalThis.lifecycleRuntime.rerank(input), timeoutMs);
    const close = await observeSettled(() => globalThis.lifecycleRuntime.close(), timeoutMs);
    return { lost, oldSession, close, oldSessionRejected: oldSession.status === 'rejected' };
  }, { input: references[0].input, timeoutMs: config.operationTimeoutMs });
  await retain({ stage: 'device-loss', ...loss });
  assert.equal(loss.lost.status, 'fulfilled');
  if (native) assert(loss.oldSessionRejected, 'Doppler must reject a lost device.');
  const recoveryStarted = performance.now();
  await context.close(); context = null;
  page = await openPage('recovery-profile');
  await verify(page, 'recovered');
  const recoveryMs = performance.now() - recoveryStarted;
  await retain({ stage: 'recovery', strategy: 'close browser context and reopen in a fresh profile',
    scope: 'Context closure, browser launch, model opening and every frozen reference', recoveryMs, maxRecoveryMs: config.maxRecoveryMs });
  assert(recoveryMs <= config.maxRecoveryMs);
  await page.evaluate(() => globalThis.lifecycleRuntime.close());
  report.passed = true;
} catch (error) { report.error = { message: error.message, stack: error.stack }; }
finally {
  for (const close of [() => context?.close(), () => server?.close(), () => checkpoint?.verifySourceUnchanged()]) {
    try { await close(); } catch (error) { report.cleanup.push(error.message); }
  }
  report.passed &&= report.cleanup.length === 0;
  report.completedAtUtc = new Date().toISOString();
  await fs.writeFile(path.join(config.outputDir, 'observation.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
}
console.log(JSON.stringify({ passed: report.passed, outputDir: config.outputDir, error: report.error }));
if (!report.passed) process.exitCode = 1;
