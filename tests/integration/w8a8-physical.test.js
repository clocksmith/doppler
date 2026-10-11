import assert from 'node:assert/strict';
import { readFile, writeFile, mkdtemp, rm, cp, mkdir } from 'node:fs/promises';
import { createServer } from 'node:http';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { resolve, join, sep } from 'node:path';
import { tmpdir } from 'node:os';
import { chromium } from 'playwright';

const output = process.argv[2];
if (!output) console.log('w8a8-physical: SKIP (explicit result path required)');
else {
  const exec = promisify(execFile), hash = bytes => createHash('sha256').update(bytes).digest('hex');
  const directory = await mkdtemp(join(tmpdir(), 'doppler-w8a8-physical-'));
  let server, browser;
  const receipt = { scope: 'Physical W8A8 operator and ownership checks against independent rational reference; not model qualification', files: {} };
  try {
    receipt.commit = (await exec('git', ['rev-parse', 'HEAD'])).stdout.trim();
    receipt.sourceStatus = (await exec('git', ['status', '--short'])).stdout;
    await cp('src', join(directory, 'src'), { recursive: true });
    await mkdir(join(directory, 'tests/kernels/reference'), { recursive: true });
    for (const file of ['tests/kernels/w8a8-physical.js', 'tests/kernels/reference/w8a8.js']) await cp(file, join(directory, file));
    server = createServer(async (request, response) => {
      try {
        const pathname = new URL(request.url, 'http://localhost').pathname;
        if (pathname === '/') { response.end('<!doctype html>'); return; }
        const file = resolve(directory, '.' + pathname); assert(file.startsWith(directory + sep));
        const bytes = await readFile(file); receipt.files[pathname] = hash(bytes);
        response.setHeader('Content-Type', pathname.endsWith('.js') ? 'text/javascript' : 'application/json');
        response.end(bytes);
      } catch (error) { response.statusCode = 404; response.end(error.message); }
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    browser = await chromium.launch({ channel: 'chrome', headless: true,
      args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'] });
    receipt.browser = browser.version();
    const system = await browser.newBrowserCDPSession(); receipt.systemInfo = await system.send('SystemInfo.getInfo'); await system.detach();
    const page = await browser.newPage(); await page.goto(`http://127.0.0.1:${server.address().port}`);
    receipt.result = await page.evaluate(async () => {
      globalThis.__DOPPLER_KERNEL_BASE_PATH__ = '/src/gpu/kernels/';
      return (await import('/tests/kernels/w8a8-physical.js')).auditW8A8();
    });
    await writeFile(output, JSON.stringify(receipt, null, 2) + '\n');
    assert(receipt.result.qualified, receipt.result.failure);
    console.log(JSON.stringify({ commit: receipt.commit, browser: receipt.browser, result: receipt.result }));
  } catch (error) { receipt.failure = error.stack; await writeFile(output, JSON.stringify(receipt, null, 2) + '\n'); throw error; }
  finally { await browser?.close(); if (server) await new Promise(resolve => server.close(resolve)); await rm(directory, { recursive: true, force: true }); }
}
