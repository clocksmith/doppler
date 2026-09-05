#!/usr/bin/env node

import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../src/tooling/node-browser-command-runner.js';
import { hashStableJson } from '../src/tooling/program-bundle/materialize.js';
import { hashBytesSha256 } from '../src/formats/canonical-hash.js';
import { parseManifest } from '../src/formats/rdrr/parsing.js';
import { evaluateSequenceReference, validateSequenceReference } from './lib/sequence-model-qualification.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

export async function qualifySequenceModelBrowser(config) {
  for (const field of ['modelDir', 'referencePath', 'outputPath', 'browserExecutablePath']) {
    if (typeof config?.[field] !== 'string' || !config[field].trim()) throw new Error(`Browser qualification requires ${field}.`);
  }
  if (!Array.isArray(config.browserArgs) || config.browserArgs.some((arg) => typeof arg !== 'string')
    || !Number.isSafeInteger(config.timeoutMs) || config.timeoutMs < 1) {
    throw new Error('Browser qualification requires browserArgs and timeoutMs.');
  }
  const manifestBytes = await readFile(resolve(config.modelDir, 'manifest.json'));
  const manifest = parseManifest(manifestBytes.toString('utf8'));
  const referenceBytes = await readFile(config.referencePath);
  const reference = validateSequenceReference(JSON.parse(referenceBytes.toString('utf8')));
  const logs = [];
  const report = {
    schema: 'doppler.sequenceModelQualification.v1', passed: false,
    generatedAt: new Date().toISOString(),
    model: { modelId: manifest.modelId, manifestHash: hashBytesSha256(manifestBytes), artifactIdentity: manifest.artifactIdentity,
      quantization: manifest.quantization, architecture: manifest.architecture, sequence: manifest.inference.sequence, session: manifest.inference.session },
    reference: { path: relative(ROOT, resolve(config.referencePath)), digest: hashBytesSha256(referenceBytes),
      source: reference.source, input: reference.input, tolerances: reference.tolerances },
    runtime: { surface: 'browser-webgpu', sourceRevision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim(),
      sourceDirty: Boolean(execFileSync('git', ['status', '--short'], { cwd: ROOT, encoding: 'utf8' }).trim()),
      executionGraphHash: hashStableJson(manifest.inference.execution), browserExecutablePath: config.browserExecutablePath,
      browserArgs: config.browserArgs },
    stage: 'browser-launch', logs,
  };
  let server;
  let browser;
  try {
    server = await createStaticFileServer({ rootDir: ROOT, host: '127.0.0.1',
      staticMounts: [{ urlPrefix: '/__model', rootDir: resolve(config.modelDir) }] });
    browser = await chromium.launch({ executablePath: config.browserExecutablePath, args: config.browserArgs, headless: true });
    report.runtime.browserVersion = browser.version();
    const context = await browser.newContext();
    await context.route('**/*', (route) => {
      const url = new URL(route.request().url());
      if (url.origin !== server.baseUrl) return route.abort();
      if (url.pathname === '/qualification') return route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>Sequence qualification</title>' });
      return url.pathname.startsWith('/src/') || url.pathname.startsWith('/__model/') ? route.continue() : route.abort();
    });
    const page = await context.newPage();
    page.setDefaultTimeout(config.timeoutMs);
    page.on('console', (message) => logs.push({ type: message.type(), text: message.text() }));
    page.on('pageerror', (error) => logs.push({ type: 'pageerror', text: error.message }));
    await page.goto(`${server.baseUrl}/qualification`);
    report.stage = 'browser-execution';
    const timeout = setTimeout(() => browser.close(), config.timeoutMs);
    let observation;
    try {
      observation = await page.evaluate(async ({ input, manifest }) => {
        const { load } = await import('/src/client/doppler-api.browser.js');
        const { observeInitialExecutionIdentity } = await import('/src/config/initial-execution-identity.js');
        const { getKernelCapabilities } = await import('/src/gpu/device.js');
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter || adapter.isFallbackAdapter === true) throw new Error('Physical WebGPU adapter required; fallback is not qualification.');
        const model = await load({ url: `${location.origin}/__model` }, { runtimeConfig: { inference: { session: manifest.inference.session } } });
        try {
          const initialExecutionIdentity = observeInitialExecutionIdentity(model.advanced.getResolvedRuntimeSession());
          const options = { includeTokenEmbeddings: true, includeLogits: false };
          const result = await model.encodeSequence(input.sequence, options);
          const capabilities = getKernelCapabilities();
          return { manifest: model.manifest, initialExecutionIdentity, options,
            isFallbackAdapter: adapter.isFallbackAdapter ?? null, adapterInfo: capabilities.adapterInfo,
            capabilities: { hasF16: capabilities.hasF16, hasSubgroups: capabilities.hasSubgroups, maxBufferSize: capabilities.maxBufferSize },
            result: { tokens: Array.from(result.tokens), embeddingDim: result.embeddingDim, logits: result.logits,
              pooledEmbedding: Array.from(result.pooledEmbedding), tokenEmbeddings: Array.from(result.tokenEmbeddings), phase: result.phase } };
        } finally { await model.unload(); }
      }, { input: reference.input, manifest });
    } finally { clearTimeout(timeout); }
    if (hashStableJson(parseManifest(JSON.stringify(observation.manifest))) !== hashStableJson(manifest)) {
      throw new Error('Browser loaded a manifest different from the frozen qualification input.');
    }
    const result = { ...observation.result, pooledEmbedding: new Float32Array(observation.result.pooledEmbedding),
      tokenEmbeddings: new Float32Array(observation.result.tokenEmbeddings) };
    const evaluation = evaluateSequenceReference({ manifest: observation.manifest, result, reference });
    Object.assign(report.runtime, { adapterInfo: observation.adapterInfo, capabilities: observation.capabilities,
      isFallbackAdapter: observation.isFallbackAdapter, initialExecutionIdentity: observation.initialExecutionIdentity });
    report.result = { options: observation.options, embeddingDim: result.embeddingDim, tokenCount: result.tokens.length,
      checks: evaluation.checks, outputDigests: evaluation.outputDigests, phase: result.phase, loraQualification: null };
    report.passed = evaluation.passed;
    report.stage = 'complete';
  } catch (error) { report.error = { name: error.name, message: error.message }; }
  finally {
    for (const cleanup of [() => browser?.close(), () => server?.close()]) {
      try { await cleanup(); } catch (error) { report.passed = false; report.cleanupError = error.message; }
    }
    await mkdir(dirname(resolve(config.outputPath)), { recursive: true });
    await writeFile(config.outputPath, `${JSON.stringify(report, null, 2)}\n`);
  }
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const main = async () => {
    if (process.argv.length !== 4 || process.argv[2] !== '--config') throw new Error('Usage: node tools/qualify-sequence-model-browser.js --config <path>');
    const config = JSON.parse(await readFile(process.argv[3], 'utf8'));
    const report = await qualifySequenceModelBrowser(config);
    console.log(JSON.stringify({ passed: report.passed, stage: report.stage, error: report.error ?? null, outputPath: config.outputPath }));
    if (!report.passed) process.exitCode = 1;
  };
  main().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
