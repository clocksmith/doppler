#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';
import { runGenerationQualificationScenario } from './generation-qualification-scenario.js';
import { runW8A8QualificationScenario } from './w8a8-qualification-scenario.js';
import { runW8A8BenchmarkScenario } from './w8a8-benchmark-scenario.js';
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
const w8a8 = reference.schema === 'doppler.w8a8-model-reference/v1';
const w8Benchmark = w8a8 && config.phase === 'fixed-work-benchmark';
const w8Boundary = w8a8 && config.phase === 'boundary-diagnostic';
if (w8Boundary) assert.equal(config.surface, 'browser');
if (w8Benchmark) {
  assert.equal(config.comparisonModels.length, 2);
  for (const model of config.comparisonModels) {
    assert(path.isAbsolute(model.modelDir));
    assert.equal(hash(await fs.readFile(path.join(model.modelDir, 'manifest.json'))), model.manifestDigest);
  }
}
if (!w8a8) assert.equal(reference.schema, 'doppler.document-generation-source-reference/v1');
else {
  assert([undefined, 'quality', 'fixed-work-benchmark', 'boundary-diagnostic'].includes(config.phase));
  assert.deepEqual(config.generation, reference.suite.generation);
}
assert.equal(config.generation.maxTokens, w8a8 ? reference.suite.generation.maxTokens : reference.policy.maxNewTokens);
assert.equal(config.generation.temperature, 0); assert.equal(config.generation.topK, 1);
const installed = await read(path.join(config.packageBundlePath, 'receipt.json'));
assert(installed.passed);
assert.equal(hash(await fs.readFile(path.join(config.packageBundlePath, installed.package.filename))), `sha256:${installed.package.sha256}`);
const installedRoot = path.join(config.packageBundlePath, 'consumer/node_modules/doppler-gpu');
const moduleUrl = file => pathToFileURL(path.join(installedRoot, file)).href;
const scenarioFilename = w8Boundary ? 'w8a8-boundary-observer.js' : w8Benchmark ? 'w8a8-benchmark-scenario.js' : w8a8 ? 'w8a8-qualification-scenario.js' : 'generation-qualification-scenario.js';
const scenarioPath = new URL('./' + scenarioFilename, import.meta.url);
const installedMetadata = await read(path.join(installedRoot, 'package.json'));
const publicApiTarget = installedMetadata.exports['./compat'].import;
const publicDeviceTarget = installedMetadata.exports['./tooling/device'].import;
const publicPartitionsTarget = installedMetadata.exports['./partitions'].import;
await fs.mkdir(config.outputDir);
const report = { schema: 'doppler.installed-generation-qualification-result/v1', passed: false, config,
  installedPackage: installed.package, referenceDigest: config.reference.digest, manifestDigest: config.manifestDigest,
  source: reference.source, surface: config.surface, startedAtUtc: new Date().toISOString(),
  qualifierDigest: hash(await fs.readFile(new URL(import.meta.url))), scenarioDigest: hash(await fs.readFile(scenarioPath)),
  stage: 'provider', phases: [], requests: [], logs: [], memorySamples: [], physicalExecution: false,
  signedCapsule: false, externalAdoption: false, comparisonScope: w8a8 ? 'Frozen quantization approximation, task quality and lifecycle; execution success is not quality acceptance.' : 'Exact frozen source tokens; no vendor performance comparison.' };
let context, server, release, restoreFetch, timer, pendingSample;
const observe = async phase => {
  report.phases.push(phase); report.stage = phase.stage;
  await fs.writeFile(path.join(config.outputDir, 'progress.json'), JSON.stringify({ stage: report.stage, phases: report.phases }, null, 2));
  console.log(JSON.stringify(phase));
};
const scenario = { generation: config.generation, runtimeConfig: config.runtimeConfig, manifestDigest: config.manifestDigest, boundaryStage: config.boundaryStage, boundaryLayer: config.boundaryLayer,
  cancellation: config.cancellation, repeatRuns: config.repeatRuns };
try {
  if (config.surface === 'browser') {
    server = await createStaticFileServer({ rootDir: installedRoot, host: '127.0.0.1', port: 0,
      staticMounts: [{ urlPrefix: '/model', rootDir: config.modelDir },
        { urlPrefix: '/qualification-tools', rootDir: import.meta.dirname },
        ...(w8Benchmark ? config.comparisonModels.map((model, index) => ({ urlPrefix: '/comparison-model/' + index, rootDir: model.modelDir })) : [])] });
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
    if (w8Benchmark) report.memorySamples.push({ at: performance.now(), rendererRssBytes: await rendererRss(pids) });
    else timer = setInterval(() => {
      if (pendingSample) return;
      pendingSample = rendererRss(pids).then(bytes => report.memorySamples.push({ at: performance.now(), rendererRssBytes: bytes }))
        .catch(error => { report.memoryError = error.message; }).finally(() => { pendingSample = null; });
    }, config.sampleIntervalMs);
    report.result = await page.evaluate(async ({ scenario, reference, scenarioFilename, w8a8, w8Benchmark, w8Boundary, comparisonModels, publicApiTarget, publicDeviceTarget, publicPartitionsTarget }) => {
      const owner = await import('/qualification-tools/' + scenarioFilename);
      const run = w8Boundary ? owner.runW8A8BoundaryScenario : w8Benchmark ? owner.runW8A8BenchmarkScenario : w8a8 ? owner.runW8A8QualificationScenario : owner.runGenerationQualificationScenario;
      return run({ ...scenario, comparisonModels: comparisonModels?.map((model, index) => ({ ...model, modelUrl: location.origin + '/comparison-model/' + index + '/' })), modelUrl: location.origin + '/model/',
        apiModule: '/' + publicApiTarget.replace(/^\.\//, ''), deviceModule: w8a8 ? '/' + publicDeviceTarget.replace(/^\.\//, '') : '/src/gpu/device.js',
        partitionsModule: '/' + publicPartitionsTarget.replace(/^\.\//, ''), poolObservationModule: '/src/memory/buffer-pool.js',
        identityModule: '/src/config/initial-execution-identity.js' }, reference, globalThis.retainGenerationPhase);
    }, { scenario, reference, scenarioFilename, w8a8, w8Benchmark, w8Boundary, comparisonModels: config.comparisonModels, publicApiTarget, publicDeviceTarget, publicPartitionsTarget });
    clearInterval(timer); await pendingSample;
    if (w8Benchmark) {
      report.memorySamples.push({ at: performance.now(), rendererRssBytes: await rendererRss(pids) });
      report.memoryDefinition = 'Two renderer RSS observations outside primary timing; no RSS peak claim or periodic sampler during benchmark.';
    } else {
      report.peakRssBytes = Math.max(...report.memorySamples.map(sample => sample.rendererRssBytes));
      assert(Number.isFinite(report.peakRssBytes) && !report.memoryError);
      report.memoryDefinition = 'Sampled Linux renderer RSS, including shared mappings, across load, generation, cancellation and recovery.';
    }
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
    const runScenario = w8Boundary ? (await import('./w8a8-boundary-observer.js')).runW8A8BoundaryScenario : w8Benchmark ? runW8A8BenchmarkScenario : w8a8 ? runW8A8QualificationScenario : runGenerationQualificationScenario;
    report.result = await runScenario({ ...scenario, comparisonModels: config.comparisonModels?.map(model => ({ ...model, modelUrl: pathToFileURL(model.modelDir + '/').href })), modelUrl: pathToFileURL(config.modelDir + '/').href,
      apiModule: moduleUrl(publicApiTarget), deviceModule: moduleUrl(w8a8 ? publicDeviceTarget : 'src/gpu/device.js'),
      partitionsModule: moduleUrl(publicPartitionsTarget), poolObservationModule: moduleUrl('src/memory/buffer-pool.js'),
      identityModule: moduleUrl('src/config/initial-execution-identity.js') }, reference, observe);
    report.peakRssBytes = process.resourceUsage().maxRSS * 1024;
    report.memoryDefinition = 'Linux fresh-process peak RSS, including shared mappings, across load, generation, cancellation and recovery.';
  }
  report.passed = report.result.passed; report.physicalExecution = report.result.outputs.length > 0 || (report.result.pairs?.length ?? 0) > 0;
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
