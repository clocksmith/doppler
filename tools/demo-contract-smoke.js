#!/usr/bin/env node

import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { checkDemoControls } from '../tests/demo/browser-controls.js';
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
  },
  inspect: {
    async generate(prompt, options = {}) {
      controls.calls.push({ prompt, policyId: options.policyId, generation: options.generation });
      let tokenIds = [];
      controls.emitTokens = (ids) => {
        for (const tokenId of ids) {
          options.onEvent?.({ type: 'token', tokenId, index: tokenIds.length });
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
        controls.emitTokens(controls.blockNext ? [1] : [1, 2, 3]);
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
        generatedTokenIds: tokenIds,
        wallTimingMs: 1,
        performanceRepresentative: policy.performanceRepresentative,
        tokens: [],
        quality: policy.id === 'demo/guided-quality' ? { words: [{ text: 'Contract', rollingPerplexity: 2, summedSurprisal: 1, cumulativePerplexity: 2, tokenCount: 1, rollingWindow: { size: 1, unit: 'word' } }] } : null,
        generationEvidence: { stats: { tokensGenerated: 3, decodeTimeMs: 1 } },
      };
    },
  },
};
export const dr = {
  async listModelDetails() { return [{ modelId: 'contract-model', label: 'Contract model' }, { modelId: 'second-model', label: 'Second model' }]; },
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
  await page.addInitScript(() => {
    if (!navigator.gpu) {
      Object.defineProperty(navigator, 'gpu', {
        configurable: true,
        value: {},
      });
    }
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
    await page.goto(`${origin}${entrypoint}`, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => {
      const select = document.querySelector('#model-select');
      return select && !select.disabled && select.options.length > 0;
    });
    journey.catalogRendered = true;
    await page.selectOption('#model-select', 'contract-model');
    journey.modelSelected = await page.$eval(
      '#model-select',
      (element) => element.value === 'contract-model'
    );
    await page.click('#model-select-action');
    await page.waitForFunction(() => document.querySelector('#status-text')?.textContent === 'Ready');
    journey.modelLoaded = await page.$eval(
      '#model-select-action',
      (element) => element.textContent === 'Loaded'
    );
    await page.fill('#prompt-input', 'Run the demo contract.');
    await page.click('#run-btn');
    await page.waitForFunction(
      () => document.querySelector('#output-phase').textContent.startsWith('Complete')
        && document.querySelector('#output-text').textContent === 'Contract generation passed.'
    );
    journey.generationCompleted = true;
    await checkDemoControls(page);
    testedShellDigest = await page.evaluate(async (url) => (await import(url)).SHELL_MANIFEST_DIGEST,
      `${origin}${modulePath.replace('/src/index-browser.js', '/demo/generated-shell-manifest.js')}`);
    await page.evaluate(() => navigator.serviceWorker.ready);
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
