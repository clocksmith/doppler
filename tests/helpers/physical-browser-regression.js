import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../../src/tooling/node-browser-command-runner.js';
import { resolveBrowserArgs } from '../../tools/demo-hardware-smoke.js';

export async function runPhysicalBrowserRegression({ modulePath, exportName, sourcePaths, defaultOutput, scope }) {
  assert(process.argv.includes('--physical'), 'Pass --physical to run this GPU regression.');
  const outputIndex = process.argv.indexOf('--output');
  if (outputIndex >= 0) assert(process.argv[outputIndex + 1], '--output requires a path');
  const output = resolve(outputIndex >= 0 ? process.argv[outputIndex + 1] : defaultOutput);
  const server = await createStaticFileServer({ rootDir: process.cwd(), host: '127.0.0.1' });
  let browser;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true, args: resolveBrowserArgs() });
    const page = await browser.newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(server.baseUrl + '/package.json');
    const result = await page.evaluate(async ({ modulePath, exportName }) => {
      const { initDevice, getDevice, destroyDevice } = await import('/src/gpu/device.js');
      const regression = await import(modulePath);
      await initDevice();
      try {
        const info = getDevice().adapterInfo;
        const adapter = Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(key => [key, info?.[key] || '']));
        if (!adapter.vendor && !adapter.architecture) throw new Error('GPU identity required');
        if (/swiftshader|llvmpipe|software/i.test(JSON.stringify(adapter))) throw new Error('Physical GPU required');
        return { adapter, cases: await regression[exportName]() };
      } finally { destroyDevice(); }
    }, { modulePath, exportName });
    assert.deepEqual(errors, []);
    const sources = {};
    for (const path of [...sourcePaths, 'tests/helpers/physical-browser-regression.js']) {
      sources[path] = createHash('sha256').update(await readFile(path)).digest('hex');
    }
    await mkdir(dirname(output), { recursive: true });
    await writeFile(output, JSON.stringify({ scope, browser: browser.version(), ...result, sources, errors }, null, 2));
    console.log(JSON.stringify({ output, ...result, errors }));
  } finally {
    try { await browser?.close(); } finally { await server.close(); }
  }
}
