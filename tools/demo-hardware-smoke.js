#!/usr/bin/env node

import http from 'node:http';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import {
  DEMO_HARDWARE_RECEIPT_SCHEMA,
  validateDemoHardwareReceipt,
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
const DEFAULT_MODEL_ID = 'gemma-3-270m-it-q4k-ehf16-af32';
const DEFAULT_PROMPT = 'Answer with one word: sky color?';

function parseArgs(argv) {
  const options = {
    modelId: DEFAULT_MODEL_ID,
    prompt: DEFAULT_PROMPT,
    allowCapabilitySkip: false,
    headless: true,
    profileDir: null,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    const value = () => {
      const next = argv[index + 1];
      if (!next || next.startsWith('--')) throw new Error(`Missing value for ${token}`);
      index += 1;
      return next;
    };
    if (token === '--model-id') options.modelId = value();
    else if (token === '--prompt') options.prompt = value();
    else if (token === '--profile-dir') options.profileDir = path.resolve(value());
    else if (token === '--allow-capability-skip') options.allowCapabilitySkip = true;
    else if (token === '--headed') options.headless = false;
    else throw new Error(`Unknown argument: ${token}`);
  }
  return options;
}

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

function capabilityFailure(error) {
  return /webgpu|gpu adapter|requestadapter|unsupported gpu feature|shader-f16/iu.test(
    String(error?.message ?? error)
  );
}

function normalizeDigest(value) {
  const digest = String(value ?? '').toLowerCase();
  return digest.startsWith('sha256:') ? digest : `sha256:${digest}`;
}

function executionClass(adapter) {
  const text = JSON.stringify(adapter).toLowerCase();
  return /swiftshader|llvmpipe|software/iu.test(text)
    ? 'software-webgpu'
    : 'hardware-webgpu';
}

async function configureDeterministicGeneration(page, prompt) {
  await page.evaluate(() => {
    const values = {
      'set-temperature': '0',
      'set-top-k': '1',
      'set-top-p': '1',
      'set-max-tokens': '1',
    };
    for (const [id, value] of Object.entries(values)) {
      const element = document.getElementById(id);
      if (!element) throw new Error(`Missing demo generation control ${id}`);
      element.value = value;
      element.dispatchEvent(new Event('input', { bubbles: true }));
      element.dispatchEvent(new Event('change', { bubbles: true }));
    }
  });
  await page.fill('#prompt-input', prompt);
}

async function runDemoGeneration(page, prompt) {
  await configureDeterministicGeneration(page, prompt);
  await page.click('#run-btn');
  await page.waitForFunction(
    () => (
      globalThis.__DOPPLER_DEMO_EVIDENCE__?.generatedTokenIds?.length > 0
      || document.querySelector('#output-phase')?.textContent?.startsWith('Error:')
    ),
    null,
    { timeout: 300_000 }
  );
  const phase = await page.$eval('#output-phase', (element) => element.textContent ?? '');
  if (phase.startsWith('Error:')) {
    throw new Error(`Demo generation failed: ${phase}`);
  }
  return page.evaluate(() => JSON.parse(JSON.stringify(globalThis.__DOPPLER_DEMO_EVIDENCE__)));
}

async function waitForLoadedModel(page, modelId, clickIfNeeded) {
  await page.waitForFunction(
    (expected) => Array.from(document.querySelectorAll('#model-select option'))
      .some((option) => option.value === expected),
    modelId,
    { timeout: 60_000 }
  );
  await page.selectOption('#model-select', modelId);
  if (clickIfNeeded) await page.click('#model-select-action');
  await page.waitForFunction(
    () => {
      const action = document.querySelector('#model-select-action')?.textContent;
      const status = document.querySelector('#status-text')?.textContent ?? '';
      return action === 'Loaded' || status.startsWith('Load failed');
    },
    null,
    { timeout: 600_000 }
  );
  const loadState = await page.evaluate(() => ({
    action: document.querySelector('#model-select-action')?.textContent ?? '',
    status: document.querySelector('#status-text')?.textContent ?? '',
    detail: document.querySelector('#model-select-detail')?.textContent ?? '',
  }));
  if (loadState.action !== 'Loaded') {
    throw new Error(`Demo model load failed: ${loadState.status} (${loadState.detail})`);
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const { server, origin } = await startServer();
  const temporaryProfile = options.profileDir
    ? null
    : await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-demo-hardware-'));
  const profileDir = options.profileDir ?? temporaryProfile;
  const context = await chromium.launchPersistentContext(profileDir, {
    headless: options.headless,
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--use-angle=vulkan',
      '--disable-vulkan-surface',
    ],
  });
  let onlineEvidence = null;
  let offlineEvidence = null;
  let allPagesClosed = false;
  let upgradeChecked = false;
  let partialCacheFailureChecked = false;
  let browserIdentity = {};
  let caught = null;

  try {
    const page = context.pages()[0] ?? await context.newPage();
    await page.goto(`${origin}/demo/index.html`, { waitUntil: 'domcontentloaded' });
    await waitForLoadedModel(page, options.modelId, true);
    onlineEvidence = await runDemoGeneration(page, options.prompt);
    browserIdentity = await page.evaluate(() => ({
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      language: navigator.language,
    }));

    upgradeChecked = await page.evaluate(async () => {
      const obsolete = await caches.open('doppler-demo-shell-obsolete');
      await obsolete.put('/obsolete', new Response('obsolete'));
      const registrations = await navigator.serviceWorker.getRegistrations();
      await Promise.all(registrations.map((registration) => registration.unregister()));
      return true;
    });
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.evaluate(() => navigator.serviceWorker.ready);
    upgradeChecked = upgradeChecked && !await page.evaluate(
      async () => (await caches.keys()).includes('doppler-demo-shell-obsolete')
    );

    partialCacheFailureChecked = await page.evaluate(async () => {
      const { CACHE_NAME } = await import('/demo/generated-shell-manifest.js');
      const cache = await caches.open(CACHE_NAME);
      return cache.delete('/demo/assets/pwa/screenshot-desktop.png');
    });

    await Promise.all(context.pages().map((openPage) => openPage.close()));
    allPagesClosed = context.pages().length === 0;
    await context.setOffline(true);
    const offlinePage = await context.newPage();
    await offlinePage.goto(`${origin}/demo/index.html`, {
      waitUntil: 'domcontentloaded',
      timeout: 60_000,
    });
    await waitForLoadedModel(offlinePage, options.modelId, false);
    offlineEvidence = await runDemoGeneration(offlinePage, options.prompt);
  } catch (error) {
    caught = error;
  } finally {
    await context.close();
    await new Promise((resolve) => server.close(resolve));
    if (temporaryProfile) {
      await fs.rm(temporaryProfile, { recursive: true, force: true });
    }
  }

  if (caught) {
    if (options.allowCapabilitySkip && capabilityFailure(caught)) {
      console.log(JSON.stringify({
        schema: 'doppler.demo-hardware-capability-skip/v1',
        status: 'capability-skip',
        reason: String(caught.message ?? caught),
      }, null, 2));
      return;
    }
    throw caught;
  }

  const adapter = onlineEvidence.fingerprint.identity.adapter;
  const receipt = validateDemoHardwareReceipt({
    schema: DEMO_HARDWARE_RECEIPT_SCHEMA,
    status: 'passed',
    createdAtUtc: new Date().toISOString(),
    executionClass: executionClass(adapter),
    browser: browserIdentity,
    adapter,
    artifact: {
      modelId: options.modelId,
      manifestHash: normalizeDigest(onlineEvidence.generationEvidence.runtimeProfile.model.manifestHash),
    },
    online: {
      outputText: onlineEvidence.outputText,
      tokenIds: onlineEvidence.generatedTokenIds,
      transcriptHash: onlineEvidence.generationEvidence.transcriptHash,
    },
    offline: {
      outputText: offlineEvidence.outputText,
      tokenIds: offlineEvidence.generatedTokenIds,
      transcriptHash: offlineEvidence.generationEvidence.transcriptHash,
    },
    lifecycle: {
      allPagesClosed,
      networkDisabled: true,
      persistentCacheRestored: offlineEvidence.generationEvidence != null,
      upgradeChecked,
      partialCacheFailureChecked,
    },
    fingerprint: onlineEvidence.fingerprint,
  });
  console.log(JSON.stringify(receipt, null, 2));
}

await main();
