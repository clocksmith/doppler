#!/usr/bin/env node
// Physical Mac GPU execution under requested mobile limits; not phone qualification.
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const root = process.cwd();
const output = path.resolve(process.env.DOPPLER_MOBILE_REPORT ?? 'reports/gemma3-mobile-binding-20261110');
const modelId = 'gemma-3-270m-it-q4k-ehf16-af32';
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.wgsl': 'text/plain', '.svg': 'image/svg+xml' };
await fs.mkdir(output, { recursive: true });
const server = http.createServer(async (req, res) => {
  const name = new URL(req.url, 'http://localhost').pathname;
  const file = path.resolve(root, '.' + decodeURIComponent(name));
  if (!file.startsWith(root + path.sep) || name.includes('/.')) {
    res.writeHead(403).end();
    return;
  }
  try {
    const data = await fs.readFile(file);
    res.writeHead(200, { 'content-type': mime[path.extname(file)] ?? 'application/octet-stream', 'cache-control': 'no-store' });
    res.end(data);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = process.env.DOPPLER_DEMO_URL ?? `http://127.0.0.1:${server.address().port}/demo/index.html`;
const prefix = process.env.DOPPLER_DEMO_PREFIX ?? '';
const context = await chromium.launchPersistentContext(path.join(root, '.cache/gemma3-mobile-binding-browser'), {
  channel: 'chrome', headless: true,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal'],
  viewport: { width: 390, height: 844 },
});
const results = [];
try {
  for (const constrained of [false, true]) {
    const page = await context.newPage();
    const logs = [];
    const name = constrained ? 'mobile-limits' : 'baseline';
    page.on('console', message => {
      logs.push(message.type() + ': ' + message.text());
      if (/Error|fail|split GPU|Embedding|GPU\]/.test(message.text())) console.log(message.text().slice(0, 300));
    });
    page.on('pageerror', error => logs.push('pageerror: ' + error.message));
    try {
      if (constrained) await page.addInitScript(() => {
        const original = navigator.gpu.requestAdapter.bind(navigator.gpu);
        navigator.gpu.requestAdapter = async (...args) => {
          const adapter = await original(...args);
          if (!adapter) return adapter;
          const request = adapter.requestDevice.bind(adapter);
          adapter.requestDevice = descriptor => request({
            ...descriptor,
            requiredLimits: {
              ...descriptor?.requiredLimits,
              maxStorageBufferBindingSize: 134217728,
              maxBufferSize: 268435456,
              maxStorageBuffersPerShaderStage: 8,
            },
          });
          return adapter;
        };
      });
      await page.goto(url, { waitUntil: 'domcontentloaded' });
      await page.evaluate(async base => {
        globalThis.testState = (await import(`${base}/demo/ui/state.js`)).state;
      }, prefix);
      // Keep polling synchronous: an async predicate can appear truthy before its result.
      await page.waitForFunction(() => globalThis.testState.phase === 'ready' && !globalThis.testState.modelBusy, null, { timeout: 240000 });
      const cachedStartup = await page.evaluate(() => !!globalThis.testState.model);
      if (constrained) assert.equal(cachedStartup, true, 'Second page must reload the cached model during startup');
      if (!cachedStartup) {
        await page.evaluate(id => {
          document.querySelector('#model-picker').open = true;
          const select = document.querySelector('#model-select');
          select.value = id;
          select.dispatchEvent(new Event('change'));
          document.querySelector('#model-select-action').click();
        }, modelId);
      }
      await page.waitForFunction(() => {
        const state = globalThis.testState;
        return !!state.model || (!state.modelBusy && document.querySelector('#output-phase')?.textContent?.includes('failed'));
      }, null, { timeout: 240000 });
      const loaded = await page.evaluate(() => ({
        id: globalThis.testState.modelId,
        phase: document.querySelector('#output-phase')?.textContent,
      }));
      assert.equal(loaded.id, modelId, loaded.phase);
      await page.fill('#prompt-input', 'Answer with one word: what is the capital of France?');
      await page.evaluate(() => {
        Object.assign(globalThis.testState.settings, { temperature: 0, topK: 1, topP: 1, maxTokens: 8 });
      });
      await page.click('#run-btn');
      await page.waitForFunction(() => !globalThis.testState.generating && (
        !!globalThis.__DOPPLER_DEMO_EVIDENCE__ || document.querySelector('#output-phase')?.textContent?.startsWith('Error:')
      ), null, { timeout: 180000 });
      const result = await page.evaluate(async base => {
        const { getDevice, getKernelCapabilities } = await import(`${base}/src/gpu/device.js`);
        const device = getDevice();
        const evidence = globalThis.__DOPPLER_DEMO_EVIDENCE__;
        return {
          artifact: evidence?.fingerprint?.identity?.artifact ?? null,
          generatedTokenIds: evidence?.generatedTokenIds ?? [],
          phase: document.querySelector('#output-phase')?.textContent,
          limits: {
            binding: device.limits.maxStorageBufferBindingSize,
            buffers: device.limits.maxStorageBuffersPerShaderStage,
            maxBuffer: device.limits.maxBufferSize,
          },
          adapter: getKernelCapabilities().adapterInfo,
        };
      }, prefix);
      assert.ok(result.generatedTokenIds.length > 0, result.phase);
      if (constrained) {
        assert.equal(result.limits.binding, 134217728);
        assert.equal(result.limits.buffers, 8);
      }
      results.push({ constrained, cachedStartup, ...result });
      await page.screenshot({ path: path.join(output, `${name}.png`), fullPage: true });
      console.log(JSON.stringify({ constrained, limits: result.limits, tokens: result.generatedTokenIds }));
    } finally {
      await fs.writeFile(path.join(output, `${name}.log`), logs.join('\n'));
      await page.close();
    }
  }
  assert.deepEqual(results[1].generatedTokenIds, results[0].generatedTokenIds);
  await fs.writeFile(path.join(output, 'summary.json'), JSON.stringify({
    classification: 'Mac Chrome physical GPU with constrained device limits, not an actual phone test',
    url, passed: true, results,
  }, null, 2) + '\n');
} finally {
  await context.close();
  await new Promise(resolve => server.close(resolve));
}
