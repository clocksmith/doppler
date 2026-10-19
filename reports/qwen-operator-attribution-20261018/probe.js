// Bounded diagnostic over a frozen source snapshot. No optimization or speedup claim.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../../src/tooling/node-browser-command-runner.js';
const [contractPath, output] = process.argv.slice(2);
const config = JSON.parse(await fs.readFile(contractPath));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
assert.equal(hash(await fs.readFile(path.join(config.modelDirectory, 'manifest.json'))), config.modelHash);
for (const [file, digest] of Object.entries(config.sourceFiles)) {
  assert.equal(hash(await fs.readFile(path.join(config.sourceRoot, file))), digest, file);
}
assert.equal(hash(await fs.readFile(new URL(import.meta.url))), config.probeHash);
const receipt = { contractHash: hash(await fs.readFile(contractPath)), contract: config, rows: [],
  scope: 'Three-request diagnostic; profiled passes alter submission shape; no speedup estimate.', passed: false };
await fs.writeFile(output, JSON.stringify(receipt), { flag: 'wx' });
const save = () => fs.writeFile(output, JSON.stringify(receipt, null, 2) + '\n');
let server, browser, timer;
try {
  server = await createStaticFileServer({ rootDir: config.sourceRoot, host: '127.0.0.1', port: 0,
    staticMounts: [{ urlPrefix: '/model', rootDir: config.modelDirectory }] });
  browser = await chromium.launch({ channel: 'chrome', headless: true,
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--use-angle=metal'] });
  receipt.browser = browser.version();
  timer = setTimeout(() => void browser.close(), config.timeoutMs);
  const page = await browser.newPage();
  receipt.errors = [];
  page.on('pageerror', error => receipt.errors.push(String(error)));
  await page.exposeFunction('checkpoint', async row => { receipt.rows.push(row); await save(); });
  await page.goto(server.baseUrl);
  receipt.result = await page.evaluate(async config => {
    globalThis.__DOPPLER_KERNEL_BASE_PATH__ = '/src/gpu/kernels/';
    const { load } = await import('/src/index.js');
    const { getDevice, destroyDevice } = await import('/src/gpu/device.js');
    let handle;
    const rows = [];
    try {
      const opening = performance.now();
      handle = await load({ url: new URL('/model/', location.href).href }, { runtimeConfig: config.runtimeConfig });
      await checkpoint({ stage: 'open', elapsedMs: performance.now() - opening,
        adapter: { vendor: getDevice().adapterInfo?.vendor ?? null }, timestampQuery: getDevice().features.has('timestamp-query') });
      for (const profile of [false, true, false]) {
        handle.resetGenerationState();
        const tokens = [], pieces = [], started = performance.now();
        for await (const text of handle.generate(config.prompt, { ...config.generation, profile,
          executionObserver: profile, signal: AbortSignal.timeout(config.requestTimeoutMs), onToken: token => tokens.push(token) })) pieces.push(text);
        const row = { stage: profile ? 'profile' : rows.length ? 'repeat' : 'control',
          text: pieces.join(''), tokens, elapsedMs: performance.now() - started, stats: handle.advanced.getStats() };
        rows.push(row); await checkpoint(row);
        if (row.text !== config.expected.text || JSON.stringify(row.tokens) !== JSON.stringify(config.expected.tokens)) {
          throw new Error('Output differs from retained control; attribution cannot continue');
        }
      }
      return { exactOutputs: rows.every(row => row.text === rows[0].text && JSON.stringify(row.tokens) === JSON.stringify(rows[0].tokens)) };
    } finally { await handle?.unload(); destroyDevice(); }
  }, config);
  assert(receipt.result.exactOutputs, 'Profiled and unprofiled tokens differ');
  assert.equal(receipt.errors.length, 0);
  receipt.passed = true;
} catch (error) { receipt.failure = String(error.stack || error); process.exitCode = 1; }
finally {
  clearTimeout(timer);
  await browser?.close(); await server?.close();
  receipt.cleanup = 'settled'; await save();
}
console.log(JSON.stringify({ passed: receipt.passed, failure: receipt.failure, rows: receipt.rows.map(row => ({ stage: row.stage, elapsedMs: row.elapsedMs })) }));
