#!/usr/bin/env node
/** Physical Chromium source comparison. Installed bytes and observations remain explicit. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';
import { computeCanonicalSha256, hashBytesSha256 } from '../src/formats/canonical-hash.js';
import { assertEmbeddingReference, assertEmbeddingSourceIdentity, evaluateEmbeddingReference } from '../src/config/embedding-reference.js';
import { resolveCapsuleEmbeddingContract } from '../src/config/embedding-contract.js';
import { hashStableJson } from '../src/tooling/program-bundle/materialize.js';
import { parseManifest } from '../src/formats/rdrr/parsing.js';

export async function qualifyEmbeddingBrowser(config) {
  for (const key of ['packageBundlePath', 'modelDir', 'referencePath', 'outputDir']) {
    if (typeof config?.[key] !== 'string' || !path.isAbsolute(config[key])) throw new Error(`Absolute ${key} required.`);
  }
  if (!Number.isSafeInteger(config.timeoutMs) || config.timeoutMs <= 0
    || !Number.isSafeInteger(config.repeatRuns) || config.repeatRuns < 1
    || !Array.isArray(config.launchArgs) || typeof config.requiredVendor !== 'string'
    || !config.runtimeConfig || !['model', 'capsule'].includes(config.mode)) throw new Error('Explicit browser execution policy required.');
  if (config.mode === 'capsule' && (!config.capsulePath || !config.application || !config.openOptions?.trustedSigners)) {
    throw new Error('Capsule qualification requires a Capsule path, application and explicit trust.');
  }
  const bundle = config.packageBundlePath;
  const installed = JSON.parse(await fs.readFile(path.join(bundle, 'receipt.json'), 'utf8'));
  if (!installed.passed || hashBytesSha256(await fs.readFile(path.join(bundle, installed.package.filename))) !== `sha256:${installed.package.sha256}`) {
    throw new Error('Installed package must match retained smoke evidence.');
  }
  const packageRoot = path.join(bundle, 'consumer/node_modules/doppler-gpu');
  const reference = assertEmbeddingReference(JSON.parse(await fs.readFile(config.referencePath, 'utf8')));
  const manifestBytes = await fs.readFile(path.join(config.modelDir, 'manifest.json'));
  const manifest = JSON.parse(manifestBytes);
  assertEmbeddingSourceIdentity(manifest.artifactIdentity, reference);
  if (computeCanonicalSha256(resolveCapsuleEmbeddingContract(manifest)) !== computeCanonicalSha256(reference.embeddingContract)) {
    throw new Error('Frozen source and manifest embedding semantics differ.');
  }
  await fs.mkdir(config.outputDir, { recursive: false });
  const report = { schema: config.mode === 'capsule' ? 'doppler.embeddingCapsuleQualification.v1' : 'doppler.embeddingModelQualification.v1',
    passed: false, generatedAt: new Date().toISOString(), config, installedPackage: installed.package,
    model: { modelId: manifest.modelId, manifestHash: hashBytesSha256(manifestBytes), artifactIdentity: manifest.artifactIdentity },
    reference, referenceDigest: computeCanonicalSha256(reference),
    runtime: { surface: 'browser-webgpu', host: 'chromium', executionGraphHash: hashStableJson(manifest.inference.execution) },
    boundary: { operatorCount: 1, sourceComparison: true, signedCapsuleExecution: false,
      externalAdoption: false, coldOperatingSystemCache: false, independentMachineEvidence: false },
    requests: [], logs: [], stage: 'launch' };
  let server;
  let browser;
  let timer;
  try {
    server = await createStaticFileServer({ rootDir: packageRoot, host: '127.0.0.1', port: 0,
      staticMounts: [{ urlPrefix: '/model', rootDir: config.modelDir },
        ...(config.mode === 'capsule' ? [{ urlPrefix: '/capsule', rootDir: path.dirname(config.capsulePath) }] : [])] });
    browser = await chromium.launch({ headless: true, args: config.launchArgs, timeout: config.timeoutMs });
    report.runtime.browserVersion = browser.version();
    timer = setTimeout(() => { browser.close().catch(() => {}); }, config.timeoutMs);
    const page = await browser.newPage();
    page.on('console', message => report.logs.push({ type: message.type(), text: message.text() }));
    page.on('pageerror', error => report.logs.push({ type: 'pageerror', text: error.message }));
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      report.requests.push(url.href);
      if (url.origin !== server.baseUrl || (config.mode === 'capsule' && url.pathname.startsWith('/model/'))) return route.abort();
      if (url.pathname === '/qualification') return route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>Embedding qualification</title>' });
      return route.continue();
    });
    await page.goto(`${server.baseUrl}/qualification`);
    report.runtime.adapterInfo = await page.evaluate(async () => {
      const adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter.');
      return { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device,
        description: adapter.info.description, isFallbackAdapter: adapter.isFallbackAdapter ?? adapter.info.isFallbackAdapter };
    });
    const adapter = report.runtime.adapterInfo;
    if (adapter.isFallbackAdapter !== false || adapter.vendor.toLowerCase() !== config.requiredVendor
      || /swiftshader|llvmpipe|software rasterizer/i.test(JSON.stringify(adapter))) throw new Error('Required physical GPU unavailable.');
    report.stage = 'embedding';
    report.raw = await page.evaluate(async ({ config, texts }) => {
      const api = await import('/src/client/doppler-api.browser.js');
      const { observeInitialExecutionIdentity } = await import('/src/config/initial-execution-identity.js');
      const { getBufferPool } = await import('/src/memory/buffer-pool.js');
      const started = performance.now();
      let session;
      const outputs = [];
      const timings = [];
      const receipts = [];
      const memory = () => performance.memory ? { usedJSHeapSize: performance.memory.usedJSHeapSize,
        totalJSHeapSize: performance.memory.totalJSHeapSize, jsHeapSizeLimit: performance.memory.jsHeapSizeLimit } : null;
      const before = memory();
      try {
        session = config.mode === 'capsule'
          ? await api.openCapsule(`${location.origin}/capsule/${encodeURIComponent(config.capsuleFilename)}`, config.openOptions)
          : await api.load({ url: `${location.origin}/model/` }, { runtimeConfig: config.runtimeConfig });
        const loaded = performance.now();
        const afterLoad = memory();
        const buffersAfterLoad = getBufferPool().getStats();
        const identity = config.mode === 'capsule' ? session.observedInitialExecutionIdentity
          : observeInitialExecutionIdentity(session.advanced.getResolvedRuntimeSession());
        for (let repeat = 0; repeat <= config.repeatRuns; repeat++) {
          for (const [index, text] of texts.entries()) {
            const began = performance.now();
            let evidence;
            if (config.mode === 'capsule') {
              let completed;
              const request = { schema: 'doppler.capsule-operation-request/v1', operation: { name: 'embed', version: 1 },
                input: { texts: [text], application: config.application }, options: {}, assignment: null,
                limits: { maxInputBytes: 1048576, maxOutputBytes: 1048576, deadlineAt: Date.now() + config.timeoutMs } };
              for await (const event of session.executeOperation(request)) if (event.status === 'completed') completed = event;
              if (!completed) throw new Error('Embedding operation ended without completion.');
              evidence = completed.output.embeddings[0];
              receipts.push(completed.receipt);
            } else evidence = await session.embedWithEvidence(text);
            timings.push({ repeat, index, elapsedMs: performance.now() - began });
            if (repeat === 0) outputs.push({ text, tokenIds: evidence.tokens, embedding: Array.from(evidence.embedding) });
            else if (JSON.stringify(Array.from(evidence.embedding)) !== JSON.stringify(outputs[index].embedding)) throw new Error('Repeated embedding changed.');
          }
        }
        return { outputs, timings, receipts, manifest: session.manifest, initialExecutionIdentity: identity,
          loadMs: loaded - started, elapsedMs: performance.now() - started,
          memory: { before, afterLoad, afterExecution: memory(), gpuBytes: null,
            buffersAfterLoad, buffersAfterExecution: getBufferPool().getStats(),
            scope: 'browser-reported-JS-heap-and-Doppler-buffer-pool; excludes-untracked-GPU-and-process-overhead' },
          capsuleIdentity: session.capsuleIdentity ?? null, selectedTargetPlanDigest: session.selectedTargetPlanDigest ?? null };
      } finally { if (config.mode === 'capsule') await session?.close(); else await session?.unload(); }
    }, { config: { ...config, capsuleFilename: config.capsulePath ? path.basename(config.capsulePath) : null }, texts: reference.input.texts });
    if (computeCanonicalSha256(parseManifest(JSON.stringify(report.raw.manifest))) !== computeCanonicalSha256(parseManifest(JSON.stringify(manifest)))) {
      throw new Error('Loaded manifest differs from frozen candidate.');
    }
    report.observation = { input: reference.input, embeddingContract: reference.embeddingContract, outputs: report.raw.outputs };
    report.result = evaluateEmbeddingReference(reference, report.observation);
    report.initialExecutionIdentity = report.raw.initialExecutionIdentity;
    report.boundary.signedCapsuleExecution = config.mode === 'capsule' && report.raw.receipts.length === reference.input.texts.length * (config.repeatRuns + 1);
    report.passed = report.result.passed;
    report.stage = 'complete';
  } catch (error) { report.error = { name: error.name, message: error.message, stack: error.stack }; }
  finally {
    clearTimeout(timer);
    await fs.writeFile(path.join(config.outputDir, 'qualification.json'), JSON.stringify(report, null, 2));
    const errors = [];
    for (const resource of [browser, server]) {
      try { await resource?.close(); } catch (error) { errors.push(error.message); }
    }
    report.cleanup = { passed: !errors.length, errors };
    report.passed = report.passed && !errors.length;
    await fs.writeFile(path.join(config.outputDir, 'qualification.json'), JSON.stringify(report, null, 2));
  }
  if (!report.passed) throw new Error(`Embedding qualification failed: ${report.error?.message ?? 'source comparison'}; retained at ${config.outputDir}`);
  return report;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  if (args.length !== 1) throw new Error('Usage: node tools/qualify-embedding-browser.js <config.json>');
  qualifyEmbeddingBrowser(JSON.parse(await fs.readFile(args[0], 'utf8')))
    .then(report => console.log(JSON.stringify({ passed: report.passed, checks: report.result.checks.length, outputDir: report.config.outputDir })))
    .catch(error => { console.error(error.message); process.exitCode = 1; });
}
