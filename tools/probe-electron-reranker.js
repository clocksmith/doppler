#!/usr/bin/env node

import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';

const require = createRequire(import.meta.url);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

async function hashFile(filename) {
  const hash = createHash('sha256');
  let sizeBytes = 0;
  for await (const chunk of createReadStream(filename)) { hash.update(chunk); sizeBytes += chunk.length; }
  return { sha256: hash.digest('hex'), sizeBytes };
}

export function assertPhysicalAdapter(adapter, vendor) {
  const identity = [adapter.vendor, adapter.architecture, adapter.device, adapter.description].join(' ').toLowerCase();
  if (adapter.isFallbackAdapter || /swiftshader|llvmpipe|software rasterizer/.test(identity)) {
    throw new Error(`Physical Electron probe rejects software adapters: ${identity}`);
  }
  if (adapter.vendor?.toLowerCase() !== vendor) throw new Error(`Expected GPU vendor ${vendor}; observed ${identity}`);
}

export function assertRanking(report, policy) {
  // Verify and bench expose the same output contract; their metric layouts differ.
  const rows = report?.output?.ranking;
  if (!Array.isArray(rows) || rows.length !== policy.documents.length) throw new Error('Incomplete rerank output.');
  const indices = new Set(rows.map((row) => row.index));
  if (indices.size !== policy.documents.length || rows.some((row) => (
    !Number.isInteger(row.index) || row.index < 0 || row.index >= policy.documents.length
    || row.document !== policy.documents[row.index]
  ))) throw new Error('Rerank output does not bind the frozen documents.');
  if (report.metrics.topDocumentIndex !== policy.acceptance.topDocumentIndex) throw new Error('Rerank top-document oracle failed.');
  if (policy.acceptance.requireFiniteScores && rows.some((row) => !Number.isFinite(row.score))) {
    throw new Error('Non-finite rerank output.');
  }
  if (!report.results?.length || report.results.some((result) => result.passed !== true)) {
    throw new Error('Rerank harness acceptance failed.');
  }
}

async function main(argv) {
  if (argv.length !== 4) throw new Error('Usage: node tools/probe-electron-reranker.js <policy.json> <retained-package-bundle> <model-directory> <new-output-directory>');
  const [policyPath, bundleDir, modelDir, outputDir] = argv.map((value) => path.resolve(value));
  const policy = JSON.parse(await fs.readFile(policyPath, 'utf8'));
  const electronVersion = require('electron/package.json').version;
  if (electronVersion !== policy.electronVersion) throw new Error(`Pinned Electron ${policy.electronVersion} required, found ${electronVersion}.`);
  const packageReceipt = JSON.parse(await fs.readFile(path.join(bundleDir, 'receipt.json'), 'utf8'));
  if (!packageReceipt.passed) throw new Error('An installed-package smoke pass is required.');
  const tarball = path.join(bundleDir, packageReceipt.package.filename);
  const packageIdentity = await hashFile(tarball);
  if (packageIdentity.sha256 !== packageReceipt.package.sha256) throw new Error('Retained package bytes changed.');
  const packageRoot = path.join(bundleDir, 'consumer/node_modules/doppler-gpu');
  const manifest = JSON.parse(await fs.readFile(path.join(modelDir, 'manifest.json'), 'utf8'));
  if (manifest.modelId !== policy.modelId || manifest.artifactIdentity?.sourceCheckpointId !== policy.sourceCheckpoint) {
    throw new Error('Model identity does not match the frozen probe policy.');
  }
  await fs.mkdir(path.dirname(outputDir), { recursive: true });
  await fs.mkdir(outputDir);
  const evidence = {
    policy, packageIdentity, electronVersion, physicalExecution: false, passed: false,
    capturedAtUtc: new Date().toISOString(), artifactInventory: [], runs: [], console: [], failure: null,
  };
  let server;
  let application;
  try {
    await fs.copyFile(policyPath, path.join(outputDir, 'policy.json'));
    const artifactNames = ['manifest.json', manifest.tokenizer.file, ...manifest.shards.map((shard) => shard.filename)];
    for (const name of artifactNames) {
      const filename = path.resolve(modelDir, name);
      if (!filename.startsWith(`${modelDir}${path.sep}`)) throw new Error('Model artifact escaped its directory.');
      evidence.artifactInventory.push({ name, ...await hashFile(filename) });
    }
    evidence.sourceProvenance = manifest.metadata?.source ?? manifest.source ?? null;
    evidence.artifactIdentity = manifest.artifactIdentity;
    server = await createStaticFileServer({ rootDir: packageRoot, host: '127.0.0.1', port: 0,
      staticMounts: [{ urlPrefix: '/probe-model', rootDir: modelDir }] });
    application = await electron.launch({ executablePath: require('electron'), timeout: policy.timeoutMs,
      args: [...policy.launchArgs, `--doppler-probe-user-data=${path.join(outputDir, 'user-data')}`,
        path.join(ROOT, 'tools/fixtures/electron-webgpu-main.js')] });
    const page = await application.firstWindow();
    page.on('console', (message) => evidence.console.push({ type: message.type(), text: message.text() }));
    page.on('pageerror', (error) => evidence.console.push({ type: 'pageerror', text: String(error) }));
    await page.goto(`${server.baseUrl}/src/tooling/command-runner.html`);
    await page.waitForFunction(() => globalThis.__dopplerRunnerReady === true, null, { timeout: policy.timeoutMs });
    evidence.gpuInfo = await application.evaluate(async ({ app }) => app.getGPUInfo('complete'));
    evidence.processArguments = await application.evaluate(() => process.argv);
    evidence.adapter = await page.evaluate(async () => {
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('Electron renderer exposes no WebGPU adapter.');
      const info = adapter.info;
      return { vendor: info.vendor, architecture: info.architecture, device: info.device,
        description: info.description, isFallbackAdapter: adapter.isFallbackAdapter ?? info.isFallbackAdapter };
    });
    assertPhysicalAdapter(evidence.adapter, policy.requiredVendor);
    const request = { command: 'verify', workload: 'rerank', modelId: policy.modelId,
      modelUrl: `${server.baseUrl}/probe-model/`, loadMode: policy.loadMode,
      runtimeConfig: { ...policy.runtimeConfig,
        inference: { ...policy.runtimeConfig.inference, rerank: { query: policy.query, documents: policy.documents } } } };
    const first = await page.evaluate((input) => globalThis.__dopplerRunBrowserCommand(input), request);
    evidence.runs.push({ phase: 'initial-load-and-verify', request, response: first });
    assertRanking(first.result?.report, policy);
    evidence.physicalExecution = true;
    const benchRequest = { ...request, command: 'bench',
      runtimeConfig: { ...request.runtimeConfig, shared: { ...request.runtimeConfig.shared, benchmark: { run: {
        warmupRuns: policy.warmupRuns, timedRuns: policy.timedRuns, loadMode: policy.loadMode,
      } } } } };
    const bench = await page.evaluate((input) => globalThis.__dopplerRunBrowserCommand(input), benchRequest);
    evidence.runs.push({ phase: 'warm-execution', request: benchRequest, response: bench });
    assertRanking(bench.result?.report, policy);
    evidence.processMetrics = await application.evaluate(({ app }) => app.getAppMetrics());
    evidence.passed = true;
  } catch (error) {
    evidence.failure = { message: error.message, stack: error.stack };
    throw error;
  } finally {
    // Retain the execution outcome even if host teardown fails or hangs.
    const observationPath = path.join(outputDir, 'observation.json');
    await fs.writeFile(observationPath, JSON.stringify(evidence, null, 2));
    const cleanupErrors = [];
    for (const resource of [application, server]) {
      try { await resource?.close(); } catch (error) { cleanupErrors.push(String(error)); }
    }
    evidence.cleanup = { passed: cleanupErrors.length === 0, errors: cleanupErrors };
    evidence.passed = evidence.passed && evidence.cleanup.passed;
    await fs.writeFile(observationPath, JSON.stringify(evidence, null, 2));
    console.log(`Electron component evidence retained: ${outputDir}`);
    if (cleanupErrors.length && !evidence.failure) throw new Error(`Probe cleanup failed: ${cleanupErrors.join('; ')}`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch((error) => { console.error(error.stack); process.exitCode = 1; });
}
