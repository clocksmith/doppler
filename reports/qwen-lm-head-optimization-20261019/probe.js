import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../../src/tooling/node-browser-command-runner.js';

const [contractPath, output] = process.argv.slice(2);
const contract = JSON.parse(await fs.readFile(contractPath, 'utf8'));

console.log('Running LM head optimization verification probe on Chrome Metal...');
let server, browser, timer;
try {
  server = await createStaticFileServer({
    rootDir: contract.sourceRoot,
    host: '127.0.0.1',
    port: 0,
    staticMounts: [{ urlPrefix: '/model', rootDir: contract.modelDirectory }],
  });

  browser = await chromium.launch({
    channel: 'chrome',
    headless: true,
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--use-angle=metal'],
  });

  timer = setTimeout(() => {
    console.error('Timed out after 120s');
    browser.close();
  }, contract.timeoutMs || 120000);

  const page = await browser.newPage();
  await page.goto(server.baseUrl);

  const evalResult = await page.evaluate(async (config) => {
    globalThis.__DOPPLER_KERNEL_BASE_PATH__ = '/src/gpu/kernels/';
    const { load } = await import('/src/index.js');
    const { getDevice, destroyDevice } = await import('/src/gpu/device.js');

    let handle;
    const rows = [];
    try {
      handle = await load(
        { url: new URL('/model/', location.href).href },
        { runtimeConfig: config.runtimeConfig }
      );

      for (const profile of [false, true, false]) {
        handle.resetGenerationState();
        const tokens = [];
        const pieces = [];
        const reqStart = performance.now();

        for await (const text of handle.generate(config.prompt, {
          ...config.generation,
          profile,
          executionObserver: profile,
          signal: AbortSignal.timeout(config.requestTimeoutMs),
          onToken: (tok) => tokens.push(tok),
        })) {
          pieces.push(text);
        }

        const reqElapsed = performance.now() - reqStart;
        const stage = profile ? 'profile' : rows.length ? 'repeat' : 'control';
        const fullText = pieces.join('');
        const stats = handle.advanced.getStats();

        rows.push({
          stage,
          text: fullText,
          tokens,
          tokenCount: tokens.length,
          elapsedMs: reqElapsed,
          stats,
        });

        if (fullText !== config.expected.text || JSON.stringify(tokens) !== JSON.stringify(config.expected.tokens)) {
          throw new Error(`Output mismatch at stage ${stage}`);
        }
      }

      return {
        rows,
        exactMatch: rows.every(r => r.text === config.expected.text && JSON.stringify(r.tokens) === JSON.stringify(config.expected.tokens)),
      };
    } finally {
      await handle?.unload();
      destroyDevice();
    }
  }, contract);

  assert(evalResult.exactMatch, 'Outputs must match expected control');
  if (output) {
    await fs.writeFile(output, JSON.stringify(evalResult, null, 2) + '\n');
  }
  console.log(JSON.stringify({ passed: true, rows: evalResult.rows.map(r => ({ stage: r.stage, elapsedMs: r.elapsedMs })) }));
} finally {
  clearTimeout(timer);
  await browser?.close();
  await server?.close();
}
