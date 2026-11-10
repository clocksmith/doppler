#!/usr/bin/env node

import http from 'node:http';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { checkLiveActivity, checkActivationPlayback, checkTokenGeometry, checkRealActivityPlayback, checkFirstLoadLayout, checkLayerMap } from '../tests/demo/browser-activation.js';
import { checkDemoControls } from '../tests/demo/browser-controls.js';
import { checkBootLayout, checkAutomaticReplay, checkPendingExample, checkPendingInitialReplay, checkWithoutWebGpu } from '../tests/demo/browser-run-coherence.js';
import {
  SHELL_MANIFEST_DIGEST,
} from '../demo/generated-shell-manifest.js';
import {
  DEMO_CONTRACT_RECEIPT_SCHEMA,
  validateDemoContractReceipt,
} from '../src/tooling/demo-receipts.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MIME = new Map([
  ['.css', 'text/css'],
  ['.html', 'text/html'],
  ['.js', 'text/javascript'],
  ['.json', 'application/json'],
  ['.png', 'image/png'],
  ['.svg', 'image/svg+xml'],
]);
const STUB_MODULE = `
const controls = globalThis.__demoContract = { calls: [], loads: [], unloads: 0, removals: 0, blockNext: false, resolveOnAbort: false, failLoad: false };
const policies = new Map([
  ['demo/always-on', { id: 'demo/always-on', modifiesExecution: false, performanceRepresentative: true }],
  ['demo/guided-quality', { id: 'demo/guided-quality', modifiesExecution: true, performanceRepresentative: false }],
  ['demo/deep-xray', { id: 'demo/deep-xray', modifiesExecution: true, performanceRepresentative: false }],
  ['demo/model-activity', { id: 'demo/model-activity', modifiesExecution: true, performanceRepresentative: false }],
  ['demo/model-activity-timing', { id: 'demo/model-activity-timing', modifiesExecution: true, performanceRepresentative: false }],
]);
const model = {
  modelId: 'contract-model',
  loaded: true,
  manifestHash: 'sha256:${'a'.repeat(64)}',
  persistentCache: { backend: 'opfs', state: 'verified-hit', fromCache: true },
  async unload() { controls.unloads++; },
  resetGenerationState() {
    if (controls.failReset) throw new Error('Contract reset failure');
    controls.resets = (controls.resets ?? 0) + 1;
  },
  advanced: {
    decodeTokenIds(ids) {
      return ids.map((id) => ({ 1: 'Contract', 2: ' generation', 3: ' passed.', 4: '\\nAnother line.', 5: ' 🙂' })[id] ?? '').join('');
    },
    getStopTokenIds() { return [9]; },
  },
  inspect: {
    async generate(prompt, options = {}) {
      controls.calls.push({ prompt: structuredClone(prompt), policyId: options.policyId, generation: options.generation, inputTokenIds: options.inputTokenIds ?? null });
      let tokenIds = [];
      controls.emitActivation = event => options.onEvent?.(event);
      controls.emitTokens = (ids) => {
        for (const tokenId of ids) {
          // Probability-capturing policies attach the live token record, as the engine does.
          const token = options.policyId && options.policyId !== 'demo/always-on'
            ? { index: tokenIds.length, tokenId, text: model.advanced.decodeTokenIds([tokenId]), probability: .95,
              topCandidates: [{ tokenId, text: model.advanced.decodeTokenIds([tokenId]), probability: .95 }, { tokenId: 5, text: ' 🙂', probability: .1 }] }
            : null;
          options.onEvent?.({ type: 'token', tokenId, index: tokenIds.length, ...(token ? { token } : {}) });
          tokenIds.push(tokenId);
        }
      };
      if (controls.streamNext) {
        controls.streamNext = false;
        await new Promise((resolve, reject) => {
          controls.completeStream = resolve;
          controls.failStream = () => reject(new Error('Contract stream failure'));
          options.generation.signal.addEventListener('abort', () => reject(new DOMException('Stopped', 'AbortError')), { once: true });
        });
      } else {
        // A branch continues after its exact prefix; a new prompt answers in full.
        controls.emitTokens(controls.blockNext ? [1] : options.inputTokenIds ? [4] : [1, 2, 3]);
      }
      if (controls.blockNext) {
        controls.blockNext = false;
        await new Promise((resolve, reject) => {
          options.generation.signal.addEventListener('abort', () => controls.resolveOnAbort ? resolve() : reject(new DOMException('Stopped', 'AbortError')), { once: true });
        });
      }
      const policy = policies.get(options.policyId) ?? policies.get('demo/always-on');
      return {
        schema: 'doppler.model-inspection-receipt/v1',
        policy,
        fingerprint: {
          schema: 'doppler.comparison-fingerprint/v1',
          fullDigest: 'sha256:${'b'.repeat(64)}',
          qualityDigest: 'sha256:${'c'.repeat(64)}',
          performanceDigest: 'sha256:${'d'.repeat(64)}',
          identity: { execution: { backend: 'mocked-contract' }, adapter: {} },
        },
        outputText: model.advanced.decodeTokenIds(tokenIds),
        inputTokenIds: options.inputTokenIds ?? [100, 101],
        generatedTokenIds: tokenIds,
        wallTimingMs: 1,
        performanceRepresentative: policy.performanceRepresentative,
        tokens: policy.id === 'demo/always-on' ? [] : tokenIds.map((tokenId,index) => ({index,tokenId,text:model.advanced.decodeTokenIds([tokenId]),probability:index ? .2 : .95,topCandidates:[{tokenId,text:model.advanced.decodeTokenIds([tokenId]),probability:index ? .2 : .95},...(tokenId === 5 ? [] : [{tokenId:5,text:' 🙂',probability:.1}])]})),
        quality: policy.id === 'demo/guided-quality' ? { words: [{ text: 'Contract', rollingPerplexity: 2, summedSurprisal: 1, cumulativePerplexity: 2, tokenCount: 1, rollingWindow: { size: 1, unit: 'word' } }] } : null,
        generationEvidence: { stats: { tokensGenerated: 3, prefillTimeMs: 10, decodeTimeMs: 20, decodeTokens: 2, prefillProfileSteps: [], decodeProfileSteps: policy.id === 'demo/deep-xray' ? [{step:1,timings:{rmsnorm:.1,fused_ffn:1.96},totalMs:2.06}] : [], gpuTimeDecodeMs: policy.id === 'demo/deep-xray' ? 2.06 : null } },
      };
    },
  },
};
export const dr = {
  async listModelDetails() { return [{ modelId: 'contract-model', label: 'Contract model', sizeBytes: 418518521, modes: ['text'], classification: { tasks: ['generation'] } }, { modelId: 'second-model', label: 'Second model', modes: ['text'], classification: { tasks: ['generation'] } }, { modelId: 'embedding-only', modes: ['embedding'], classification: { tasks: ['embedding'] } }]; },
  async listPersistentModels() { return []; },
  async load(modelId, options) {
    controls.loads.push({ modelId, ...options });
    if (controls.failLoad) throw new Error('Contract load failure');
    return { ...model, modelId };
  },
  async removePersistentModel() { controls.removals++; return true; },
};
export const doppler = dr;
export const DOPPLER_VERSION = 'contract';
export default dr;
`;

function safePath(urlPath) {
  const decoded = decodeURIComponent(urlPath === '/' ? '/demo/index.html' : urlPath);
  const resolved = path.resolve(ROOT, `.${decoded}`);
  if (!resolved.startsWith(`${ROOT}${path.sep}`)) return null;
  return resolved;
}

async function startServer() {
  const server = http.createServer(async (request, response) => {
    const target = safePath(new URL(request.url, 'http://localhost').pathname);
    if (!target) {
      response.writeHead(403).end();
      return;
    }
    try {
      const body = await fs.readFile(target);
      response.writeHead(200, {
        'content-type': MIME.get(path.extname(target)) ?? 'application/octet-stream',
        'cache-control': 'no-store',
      });
      response.end(body);
    } catch {
      response.writeHead(404).end();
    }
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  return {
    server,
    origin: `http://127.0.0.1:${server.address().port}`,
  };
}

async function main() {
  const { server, origin: localOrigin } = await startServer();
  const origin = process.env.DOPPLER_DEMO_ORIGIN || localOrigin;
  const entrypoint = process.env.DOPPLER_DEMO_ENTRYPOINT || '/demo/index.html';
  const modulePath = entrypoint.startsWith('/doppler/') ? '/doppler/src/index-browser.js' : '/src/index-browser.js';
  const browser = await chromium.launch({ headless: true, channel: process.env.DOPPLER_BROWSER_CHANNEL || undefined });
  const page = await browser.newPage();
  const fatalConsoleErrors = [];
  page.on('pageerror', (error) => fatalConsoleErrors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') fatalConsoleErrors.push(message.text());
  });
  // The model is a stub, so WebGPU is too: an adapter that answers, whatever this Chromium build has.
  await page.addInitScript(() => {
    Object.defineProperty(Navigator.prototype, 'gpu', { configurable: true,
      get: () => ({ requestAdapter: async () => ({ limits: { maxBufferSize: 2 ** 31, maxStorageBufferBindingSize: 2 ** 31 } }) }) });
  });
  await page.route(`${origin}${modulePath}`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/javascript',
      body: STUB_MODULE,
    });
  });

  const journey = {
    catalogRendered: false,
    modelSelected: false,
    modelLoaded: false,
    generationCompleted: false,
  };
  let testedShellDigest = SHELL_MANIFEST_DIGEST;
  try {
    await page.addInitScript(() => { localStorage.setItem('doppler.demo.activity-example', 'true'); });
    await page.goto(`${origin}${entrypoint}`, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => {
      const select = document.querySelector('#model-select');
      return select && !select.disabled && select.options.length > 0;
    });
    journey.catalogRendered = true;
    await page.waitForFunction(() => document.querySelector('#boot-overlay').hidden);
    assert.deepEqual(await page.evaluate(() => [__demoContract.loads.length, __demoContract.calls.length]), [0, 0], 'Opening the example does not download or execute a model');
    assert.equal(await page.locator('#model-select').isVisible(), true, 'Model selection is visible by the composer');
    assert.equal(await page.locator('#xray-shell').isVisible(), false);
    await checkAutomaticReplay(page);
    await checkLayerMap(page);
    assert.equal(await page.locator('#run-btn').isEnabled(), false, 'The first prompt has already been submitted in replay');
    assert.equal(await page.locator('#model-select option[value="embedding-only"]').count(), 0);
    await checkFirstLoadLayout(page);
    await page.locator('.token-chip').first().click();
    assert.equal(await page.locator('.token-inspector-badge').textContent(), '"The"');
    assert.equal(await page.locator('.token-candidates-list').isVisible(), true);
    await checkTokenGeometry(page);
    assert.match(await page.locator('.token-inspector-note').textContent(), /Only a live answer can branch/);
    assert.equal(await page.locator('.token-candidate-branch').first().isDisabled(), true, 'A recorded sample cannot branch');
    if (!await page.locator('#model-select').isVisible()) {
      await page.locator('#settings-toggle').click();
    }
    await page.selectOption('#model-select', 'contract-model');
    assert.match(await page.locator('#model-select-detail').textContent(), /First use downloads about 419 MB\. Your browser keeps a copy if it can\./);

    journey.modelSelected = await page.$eval(
      '#model-select',
      (element) => element.value === 'contract-model'
    );
    await page.click('#model-select-action');
    await page.waitForFunction(() => document.querySelector('#output-phase')?.textContent === 'Ready');
    journey.modelLoaded = await page.$eval(
      '#model-select-action',
      (element) => element.textContent === 'Loaded'
    );
    if (await page.locator('#settings-panel').isVisible()) await page.locator('#settings-toggle').click();
    await page.fill('#prompt-input', 'Run the demo contract.');
    await page.click('#run-btn');
    await page.waitForFunction(
      () => document.querySelector('#output-phase').textContent.startsWith('Complete')
        && document.querySelector('#output-text').textContent === 'Contract generation passed.'
    );
    journey.generationCompleted = true;
    await checkDemoControls(page);
    await checkActivationPlayback(page);
    await checkLiveActivity(page);
    if (process.env.DOPPLER_ACTIVITY_SAMPLE === '1') await checkRealActivityPlayback(page);
    // Returning users load a stored default model before the ready surface opens.
    const cachedPage = await browser.newPage({ serviceWorkers: 'block' });
    await cachedPage.addInitScript(() => {
      localStorage.setItem('doppler.demo.activity-example', 'true');
      Object.defineProperty(Navigator.prototype, 'gpu', { configurable: true,
        get: () => ({ requestAdapter: async () => ({ limits: { maxBufferSize: 2 ** 31, maxStorageBufferBindingSize: 2 ** 31 } }) }) });
    });
    await cachedPage.route(`${origin}${modulePath}`, route => route.fulfill({ status: 200, contentType: 'text/javascript',
      body: STUB_MODULE.replaceAll("'contract-model'", "'gemma-3-270m-it-q4k-ehf16-af32'")
        .replace('async listPersistentModels() { return []; }', "async listPersistentModels() { return [{modelId:'gemma-3-270m-it-q4k-ehf16-af32'}]; }") }));
    await cachedPage.goto(`${origin}${entrypoint}`, { waitUntil: 'networkidle' });
    await cachedPage.waitForFunction(() => document.querySelector('#boot-overlay').hidden);
    assert.equal(await cachedPage.evaluate(() => __demoContract.loads.length), 1, 'The cached model loads during boot');
    await checkAutomaticReplay(cachedPage);
    await checkBootLayout(cachedPage);
    await checkPendingExample(cachedPage);
    await checkPendingInitialReplay(cachedPage);
    await cachedPage.close();
    // A browser without WebGPU, whatever this Chromium build provides.
    const plainPage = await browser.newPage({ serviceWorkers: 'block' });
    await plainPage.addInitScript(() => {
      localStorage.setItem('doppler.demo.activity-example', 'true');
      Object.defineProperty(Navigator.prototype, 'gpu', { configurable: true, get: () => undefined });
    });
    await plainPage.route(`${origin}${modulePath}`, route => route.fulfill({ status: 200, contentType: 'text/javascript', body: STUB_MODULE }));
    await plainPage.goto(`${origin}${entrypoint}`, { waitUntil: 'networkidle' });
    await checkWithoutWebGpu(plainPage);
    await plainPage.screenshot({ path: '.cache/demo-first-replay/no-webgpu-1280.png', fullPage: true });
    await plainPage.close();
    // WebGPU is present but gives no adapter, as in a browser without a usable GPU.
    const noAdapterPage = await browser.newPage({ serviceWorkers: 'block' });
    await noAdapterPage.addInitScript(() => {
      localStorage.setItem('doppler.demo.activity-example', 'true');
      Object.defineProperty(Navigator.prototype, 'gpu', { configurable: true, get: () => ({ requestAdapter: async () => null }) });
    });
    await noAdapterPage.route(`${origin}${modulePath}`, route => route.fulfill({ status: 200, contentType: 'text/javascript', body: STUB_MODULE }));
    await noAdapterPage.goto(`${origin}${entrypoint}`, { waitUntil: 'networkidle' });
    await checkWithoutWebGpu(noAdapterPage, { hasApi: true });
    await noAdapterPage.close();
    testedShellDigest = await page.evaluate(async (url) => (await import(url)).SHELL_MANIFEST_DIGEST,
      `${origin}${modulePath.replace('/src/index-browser.js', '/demo/generated-shell-manifest.js')}`);
    await page.waitForFunction(async () => Boolean((await navigator.serviceWorker.getRegistration())?.active), null, { timeout: 60000 });
  } finally {
    await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }

  const passed = Object.values(journey).every(Boolean) && fatalConsoleErrors.length === 0;
  if (entrypoint !== '/demo/index.html') {
    // Hosted wrappers have their own source and cache identity; they must not
    // impersonate the canonical demo receipt used by the goal matrix.
    console.log(JSON.stringify({
      schema: 'doppler.demo-hosted-controls-check/v1',
      status: passed ? 'passed' : 'failed',
      createdAtUtc: new Date().toISOString(),
      entrypoint,
      executionClass: 'mocked-contract',
      journey,
      shellManifestDigest: testedShellDigest,
      fatalConsoleErrors,
    }, null, 2));
    if (!passed) process.exitCode = 1;
    return;
  }
  const receipt = validateDemoContractReceipt({
    schema: DEMO_CONTRACT_RECEIPT_SCHEMA,
    status: passed ? 'passed' : 'failed',
    createdAtUtc: new Date().toISOString(),
    entrypoint,
    executionClass: 'mocked-contract',
    journey,
    shellManifestDigest: testedShellDigest,
    fatalConsoleErrors,
  });
  console.log(JSON.stringify(receipt, null, 2));
  if (!passed) process.exitCode = 1;
}

await main();
