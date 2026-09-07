#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';
import { runGenerationQualificationScenario } from './generation-qualification-scenario.js';
import { rendererPids, rendererRss } from './browser-renderer-memory.js';

const hash = bytes => `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const config = await read(process.argv[2]);
assert.equal(config.schema, 'doppler.installed-generation-qualification/v1');
assert(['node', 'bun', 'browser'].includes(config.surface));
if (config.surface !== 'browser') assert.equal(config.surface, process.versions.bun ? 'bun' : 'node');
assert(Number.isSafeInteger(config.timeoutMs) && config.timeoutMs > 0);
assert(Number.isSafeInteger(config.repeatRuns) && config.repeatRuns > 0);
assert(Number.isSafeInteger(config.sampleIntervalMs) && config.sampleIntervalMs > 0);
for (const name of ['packageBundlePath', 'modelDir', 'outputDir']) assert(path.isAbsolute(config[name]));
const referenceBytes = await fs.readFile(config.reference.path);
assert.equal(hash(referenceBytes), config.reference.digest);
assert.equal(hash(await fs.readFile(path.join(config.modelDir, 'manifest.json'))), config.manifestDigest);
const reference = JSON.parse(referenceBytes.toString('utf8'));
assert.equal(reference.schema, 'doppler.document-generation-source-reference/v1');
assert.equal(config.generation.maxTokens, reference.policy.maxNewTokens);
assert.equal(config.generation.temperature, 0); assert.equal(config.generation.topK, 1);
const installed = await read(path.join(config.packageBundlePath, 'receipt.json'));
assert(installed.passed);
assert.equal(hash(await fs.readFile(path.join(config.packageBundlePath, installed.package.filename))), `sha256:${installed.package.sha256}`);
const installedRoot = path.join(config.packageBundlePath, 'consumer/node_modules/doppler-gpu');
const moduleUrl = file => pathToFileURL(path.join(installedRoot, file)).href;
const scenarioPath = new URL('./generation-qualification-scenario.js', import.meta.url);
await fs.mkdir(config.outputDir);
const report = { schema: 'doppler.installed-generation-qualification-result/v1', passed: false, config,
  installedPackage: installed.package, referenceDigest: config.reference.digest, manifestDigest: config.manifestDigest,
  source: reference.source, surface: config.surface, startedAtUtc: new Date().toISOString(),
  qualifierDigest: hash(await fs.readFile(new URL(import.meta.url))), scenarioDigest: hash(await fs.readFile(scenarioPath)),
  stage: 'provider', phases: [], requests: [], logs: [], memorySamples: [], physicalExecution: false,
  signedCapsule: false, externalAdoption: false, comparisonScope: 'Exact frozen source tokens; no vendor performance comparison.' };
let context, server, release, restoreFetch, timer, pendingSample;
const observe = async phase => {
  report.phases.push(phase); report.stage = phase.stage;
  await fs.writeFile(path.join(config.outputDir, 'progress.json'), JSON.stringify({ stage: report.stage, phases: report.phases }, null, 2));
  console.log(JSON.stringify(phase));
};
const scenario = { generation: config.generation, runtimeConfig: config.runtimeConfig,
  cancellation: config.cancellation, repeatRuns: config.repeatRuns };
try {
  if (config.surface === 'browser') {
    server = await createStaticFileServer({ rootDir: installedRoot, host: '127.0.0.1', port: 0,
      staticMounts: [{ urlPrefix: '/model', rootDir: config.modelDir },
        { urlPrefix: '/qualification-tools', rootDir: import.meta.dirname }] });
    const profile = path.join(config.outputDir, 'profile');
    context = await chromium.launchPersistentContext(profile, { headless: true, args: config.launchArgs,
      timeout: config.timeoutMs, env: { ...process.env, TMPDIR: config.temporaryDirectory } });
    report.browserVersion = context.browser()?.version() ?? null;
    const page = context.pages()[0]; page.setDefaultTimeout(config.timeoutMs);
    const cdp = await context.newCDPSession(page);
    report.browserIdentity = await cdp.send('Browser.getVersion');
    page.on('console', message => report.logs.push({ type: message.type(), text: message.text() }));
    page.on('pageerror', error => report.logs.push({ type: 'pageerror', text: error.message }));
    await page.route('**/*', route => {
      const url = new URL(route.request().url()); report.requests.push(url.href);
      if (url.origin !== server.baseUrl || route.request().method() !== 'GET') return route.abort();
      if (url.pathname === '/qualification') return route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>Installed generation qualification</title>' });
      return route.continue();
    });
    await page.goto(server.baseUrl + '/qualification');
    report.hardware = await page.evaluate(async () => {
      const adapter = await navigator.gpu.requestAdapter();
      return Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, adapter.info[key]]));
    });
    assert.equal(report.hardware.vendor, config.requiredVendor); assert.equal(report.hardware.isFallbackAdapter, false);
    await page.exposeFunction('retainGenerationPhase', observe);
    const pids = await rendererPids(profile);
    timer = setInterval(() => {
      if (pendingSample) return;
      pendingSample = rendererRss(pids).then(bytes => report.memorySamples.push({ at: performance.now(), rendererRssBytes: bytes }))
        .catch(error => { report.memoryError = error.message; }).finally(() => { pendingSample = null; });
    }, config.sampleIntervalMs);
    report.result = await page.evaluate(async ({ scenario, reference }) => {
      const { runGenerationQualificationScenario } = await import('/qualification-tools/generation-qualification-scenario.js');
      return runGenerationQualificationScenario({ ...scenario, modelUrl: location.origin + '/model/',
        apiModule: '/src/client/doppler-api.browser.js', deviceModule: '/src/gpu/device.js',
        identityModule: '/src/config/initial-execution-identity.js' }, reference, globalThis.retainGenerationPhase);
    }, { scenario, reference });
    clearInterval(timer); await pendingSample;
    report.peakRssBytes = Math.max(...report.memorySamples.map(sample => sample.rendererRssBytes));
    assert(Number.isFinite(report.peakRssBytes) && !report.memoryError);
    report.memoryDefinition = 'Sampled Linux renderer RSS, including shared mappings, across load, generation, cancellation and recovery.';
  } else {
    const { bootstrapNodeWebGPUProvider, releaseNodeWebGPU } = await import(moduleUrl('src/tooling/node-webgpu.js'));
    release = releaseNodeWebGPU;
    const provider = await bootstrapNodeWebGPUProvider('webgpu', { createArgs: config.providerCreateArgs });
    const info = provider.session.adapter.info;
    report.provider = provider.receipt;
    report.hardware = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, info[key]]));
    assert.equal(report.hardware.vendor, config.requiredVendor); assert.equal(report.hardware.isFallbackAdapter, false);
    report.runtime = { name: config.surface, version: process.versions.bun ?? process.version };
    const originalFetch = globalThis.fetch; restoreFetch = () => { globalThis.fetch = originalFetch; };
    globalThis.fetch = async input => { report.requests.push(String(input)); throw new Error('Network disabled for retained generation qualification.'); };
    const { installNodeFileFetchShim } = await import(moduleUrl('src/tooling/node-file-fetch.js'));
    installNodeFileFetchShim();
    report.fileAccess = 'Installed Node file-fetch adapter; non-file acquisition is rejected.';
    report.result = await runGenerationQualificationScenario({ ...scenario, modelUrl: pathToFileURL(config.modelDir + '/').href,
      apiModule: moduleUrl('src/client/doppler-api.js'), deviceModule: moduleUrl('src/gpu/device.js'),
      identityModule: moduleUrl('src/config/initial-execution-identity.js') }, reference, observe);
    report.peakRssBytes = process.resourceUsage().maxRSS * 1024;
    report.memoryDefinition = 'Linux fresh-process peak RSS, including shared mappings, across load, generation, cancellation and recovery.';
  }
  report.passed = report.result.passed; report.physicalExecution = report.result.outputs.length > 0;
} catch (error) { report.error = { message: error.message, stack: error.stack }; }
finally {
  clearInterval(timer); await pendingSample;
  report.cleanup = [];
  for (const close of [() => context?.close(), () => server?.close(), () => release?.()]) {
    try { await close(); } catch (error) { report.cleanup.push(error.message); }
  }
  restoreFetch?.(); report.passed &&= report.cleanup.length === 0;
  report.completedAtUtc = new Date().toISOString();
  await fs.writeFile(path.join(config.outputDir, 'qualification.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
}
console.log(JSON.stringify({ passed: report.passed, surface: report.surface, outputDir: config.outputDir, error: report.error ?? report.result?.error }));
if (!report.passed) process.exitCode = 1;
