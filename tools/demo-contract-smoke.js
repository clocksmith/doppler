#!/usr/bin/env node

import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
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
  async unload() {},
  inspect: {
    async generate(prompt, options = {}) {
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
        outputText: 'Contract generation passed.',
        generatedTokenIds: [1, 2, 3],
        wallTimingMs: 1,
        performanceRepresentative: policy.performanceRepresentative,
        tokens: [],
        quality: null,
        generationEvidence: { stats: { tokensGenerated: 3, decodeTimeMs: 1 } },
      };
    },
  },
};
export const dr = {
  async listModelDetails() { return [{ modelId: 'contract-model', label: 'Contract model' }]; },
  async listPersistentModels() { return []; },
  async load() { return model; },
  async removePersistentModel() { return true; },
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
  const { server, origin } = await startServer();
  const browser = await chromium.launch({ headless: true });
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
  await page.route(`${origin}/src/index-browser.js`, async (route) => {
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
  try {
    await page.goto(`${origin}/demo/index.html`, { waitUntil: 'networkidle' });
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
      () => Array.from(document.querySelectorAll('.chat-message--assistant .chat-message-text'))
        .some((element) => element.textContent === 'Contract generation passed.')
    );
    journey.generationCompleted = true;
    await page.evaluate(() => navigator.serviceWorker.ready);
  } finally {
    await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }

  const passed = Object.values(journey).every(Boolean) && fatalConsoleErrors.length === 0;
  const receipt = validateDemoContractReceipt({
    schema: DEMO_CONTRACT_RECEIPT_SCHEMA,
    status: passed ? 'passed' : 'failed',
    createdAtUtc: new Date().toISOString(),
    entrypoint: '/demo/index.html',
    executionClass: 'mocked-contract',
    journey,
    shellManifestDigest: SHELL_MANIFEST_DIGEST,
    fatalConsoleErrors,
  });
  console.log(JSON.stringify(receipt, null, 2));
  if (!passed) process.exitCode = 1;
}

await main();
