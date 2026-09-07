#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';
import { computeCanonicalSha256, hashBytesSha256 } from '../src/formats/canonical-hash.js';
import { assertRerankSourceIdentity, evaluateRerankReference } from '../src/config/rerank-reference.js';
import { hashStableJson } from '../src/tooling/program-bundle/materialize.js';

// Qualification owns observation and the frozen oracle; installed code owns inference.
export async function qualifyRerankerBrowser(config) {
  for (const key of ['packageBundlePath', 'modelDir', 'referencePath', 'outputDir']) {
    if (!path.isAbsolute(config[key] ?? '')) throw new Error(`Absolute ${key} required.`);
  }
  if (!['model', 'capsule'].includes(config.mode) || !config.runtimeConfig
    || !Number.isSafeInteger(config.timeoutMs) || config.timeoutMs <= 0
    || !Number.isSafeInteger(config.repeatRuns) || config.repeatRuns < 1
    || !Array.isArray(config.launchArgs) || !config.requiredVendor) {
    throw new Error('Explicit execution and physical device policy required.');
  }
  if (config.mode === 'capsule' && (!config.capsulePath || !config.application
    || !config.openOptions?.trustedSigners || !config.openOptions?.acceptedTargetPlanDigests?.length)) {
    throw new Error('Capsule execution requires explicit artifact, trust and application approval.');
  }
  const installed = JSON.parse(await fs.readFile(path.join(config.packageBundlePath, 'receipt.json'), 'utf8'));
  const archive = await fs.readFile(path.join(config.packageBundlePath, installed.package.filename));
  if (!installed.passed || hashBytesSha256(archive) !== `sha256:${installed.package.sha256}`) {
    throw new Error('Installed-package evidence or archive integrity failed.');
  }
  const reference = JSON.parse(await fs.readFile(config.referencePath, 'utf8'));
  const manifestBytes = await fs.readFile(path.join(config.modelDir, 'manifest.json'));
  const manifest = JSON.parse(manifestBytes);
  assertRerankSourceIdentity(manifest.artifactIdentity, reference);
  await fs.mkdir(config.outputDir);
  await fs.writeFile(path.join(config.outputDir, 'index.html'), '<!doctype html><meta http-equiv="Content-Security-Policy" content="default-src \'self\'; script-src \'self\' \'unsafe-eval\'; connect-src \'self\'"><title>Reranker qualification</title>', { flag: 'wx' });
  const report = { schema: config.mode === 'capsule' ? 'doppler.rerankCapsuleQualification.v1' : 'doppler.rerankModelQualification.v1',
    passed: false, generatedAt: new Date().toISOString(), config, installedPackage: installed.package,
    model: { modelId: manifest.modelId, manifestHash: hashBytesSha256(manifestBytes), artifactIdentity: manifest.artifactIdentity },
    reference, referenceDigest: computeCanonicalSha256(reference),
    runtime: { surface: 'browser-webgpu', host: 'chromium', temporaryDirectory: tmpdir(), executionGraphHash: hashStableJson(manifest.inference.execution) },
    boundary: { externalAdoption: false, physicalCapsuleExecution: false, coldOperatingSystemCache: false },
    requests: [], logs: [], stage: 'launch' };
  let server;
  let browser;
  let timer;
  try {
    server = await createStaticFileServer({ rootDir: path.join(config.packageBundlePath, 'consumer/node_modules/doppler-gpu'),
      host: '127.0.0.1', port: 0, staticMounts: [{ urlPrefix: '/model', rootDir: config.modelDir },
        { urlPrefix: '/qualification', rootDir: config.outputDir },
        ...(config.mode === 'capsule' ? [{ urlPrefix: '/capsule', rootDir: path.dirname(config.capsulePath) }] : [])] });
    browser = await chromium.launch({ headless: true, args: config.launchArgs, timeout: config.timeoutMs });
    report.runtime.browserVersion = browser.version();
    timer = setTimeout(() => browser.close().catch(error => report.logs.push({ cleanupError: error.message })), config.timeoutMs);
    const page = await browser.newPage();
    page.on('console', message => report.logs.push({ type: message.type(), text: message.text() }));
    page.on('pageerror', error => report.logs.push({ type: 'pageerror', text: error.message }));
    page.on('requestfailed', request => report.logs.push({ type: 'requestfailed', url: request.url(), failure: request.failure() }));
    page.on('request', request => report.requests.push(request.url()));
    await page.addInitScript(({ capsuleMode }) => {
      const fetchLocal = globalThis.fetch;
      globalThis.fetch = (input, options = {}) => {
        const request = new Request(input, options);
        const url = new URL(request.url);
        if (url.origin !== location.origin || request.method !== 'GET'
          || (capsuleMode && url.pathname.startsWith('/model/'))) {
          throw new Error('Qualification permits local GET acquisition only.');
        }
        return fetchLocal(request);
      };
    }, { capsuleMode: config.mode === 'capsule' });
    await page.goto(`${server.baseUrl}/qualification/index.html`);
    report.runtime.adapterInfo = await page.evaluate(async () => {
      const adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter.');
      return Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, adapter.info[key]]));
    });
    const adapter = report.runtime.adapterInfo;
    if (adapter.isFallbackAdapter !== false || adapter.vendor.toLowerCase() !== config.requiredVendor
      || /swiftshader|llvmpipe|software/i.test(JSON.stringify(adapter))) throw new Error('Required physical GPU unavailable.');
    report.stage = 'execution';
    report.raw = await page.evaluate(async ({ config, input }) => {
      const api = await import('/src/client/doppler-api.browser.js');
      const { observeInitialExecutionIdentity } = await import('/src/config/initial-execution-identity.js');
      const { getBufferPool } = await import('/src/memory/buffer-pool.js');
      let session;
      const observations = [];
      const progress = [];
      const executions = [];
      const started = performance.now();
      try {
        session = config.mode === 'capsule'
          ? await api.openCapsule(`${location.origin}/capsule/${encodeURIComponent(config.capsuleFilename)}`, {
            ...config.openOptions, observer: { observe: event => observations.push({ elapsedMs: performance.now() - started, ...event }) },
            onLoadProgress: event => progress.push({ elapsedMs: performance.now() - started, ...event }) })
          : await api.load({ url: `${location.origin}/model/` }, { runtimeConfig: config.runtimeConfig });
        const loadMs = performance.now() - started;
        const initialExecutionIdentity = config.mode === 'capsule' ? session.observedInitialExecutionIdentity
          : observeInitialExecutionIdentity(session.advanced.getResolvedRuntimeSession());
        for (let repeat = 0; repeat <= config.repeatRuns; repeat++) {
          const began = performance.now();
          const receipt = config.mode === 'capsule' ? await session.rerank({ application: config.application, ...input }) : null;
          const evidence = receipt?.evidence ?? await session.rerankWithEvidence(input.query, input.documents);
          executions.push({ repeat, elapsedMs: performance.now() - began, receipt, evidence });
        }
        return { loadMs, initialExecutionIdentity, executions, scoringConfig: session.manifest.inference.rerank,
          observations, progress, capsuleIdentity: session.capsuleIdentity ?? null };
      } finally {
        if (config.mode === 'capsule') await session?.close(); else await session?.unload();
        globalThis.__qualificationCleanup = getBufferPool().getStats();
      }
    }, { config: { ...config, capsuleFilename: config.capsulePath ? path.basename(config.capsulePath) : null }, input: reference.input });
    report.initialExecutionIdentity = report.raw.initialExecutionIdentity;
    report.observation = { input: reference.input, scoringConfig: report.raw.scoringConfig, outputs: report.raw.executions[0].evidence.scores };
    report.comparisons = report.raw.executions.map(run => evaluateRerankReference(reference, {
      input: reference.input, scoringConfig: report.raw.scoringConfig, outputs: run.evidence.scores }));
    report.bufferPoolAfterClose = await page.evaluate(() => globalThis.__qualificationCleanup);
    report.passed = report.comparisons.every(result => result.passed);
    report.boundary.physicalCapsuleExecution = config.mode === 'capsule' && report.passed;
    report.stage = 'complete';
  } catch (error) { report.error = { message: error.message, stack: error.stack }; }
  finally {
    clearTimeout(timer);
    const errors = [];
    for (const resource of [browser, server]) {
      try { await resource?.close(); } catch (error) { errors.push(error.message); }
    }
    report.cleanup = { passed: errors.length === 0, errors };
    report.passed &&= report.cleanup.passed;
    await fs.writeFile(path.join(config.outputDir, 'qualification.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
  }
  return report;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const report = await qualifyRerankerBrowser(JSON.parse(await fs.readFile(process.argv[2], 'utf8')));
  console.log(JSON.stringify({ passed: report.passed, stage: report.stage, error: report.error?.message, outputDir: report.config.outputDir }));
  if (!report.passed) process.exitCode = 1;
}
