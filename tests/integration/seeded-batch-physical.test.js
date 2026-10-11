import assert from 'node:assert/strict';
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { resolve, join, sep } from 'node:path';
import { tmpdir } from 'node:os';
import { chromium } from 'playwright';

const output = process.argv[2];
if (!output) console.log('seeded-batch-physical: SKIP (explicit result path required)');
else {
  const exec = promisify(execFile), hash = bytes => createHash('sha256').update(bytes).digest('hex');
  const directory = await mkdtemp(join(tmpdir(), 'doppler-seeded-physical-'));
  let server, browser;
  const receipt = { scope: 'Real GPU sampler and production batch control with fixed tensor fixtures', files: {} };
  try {
    receipt.commit = (await exec('git', ['rev-parse', 'HEAD'])).stdout.trim();
    const archive = join(directory, 'source.tar');
    await exec('git', ['archive', '--output', archive, 'HEAD']); await exec('tar', ['-xf', archive, '-C', directory]);
    if (process.argv[3] === '--candidate-decode-runtime') {
      const path = 'src/inference/pipelines/text/generator/decode-runtime.js';
      const candidate = await readFile(path);
      receipt.candidate = { path, baseSha256: hash(await readFile(join(directory, path))), sha256: hash(candidate) };
      await writeFile(join(directory, path), candidate);
    } else assert(process.argv[3] === undefined, 'Unknown physical audit option');
    const helper = await readFile(new URL('../kernels/seeded-batch-physical.js', import.meta.url));
    receipt.helperSha256 = hash(helper);
    server = createServer(async (request, response) => {
      try {
        const path = new URL(request.url, 'http://localhost').pathname;
        if (path === '/') { response.end('<!doctype html>'); return; }
        response.setHeader('Content-Type', path.endsWith('.js') ? 'text/javascript' : 'application/json');
        if (path === '/audit.js') { response.end(helper); return; }
        if (path === '/fixture/embed.js') { response.end('export const embed = async () => globalThis.__samplingFixture.embed();'); return; }
        if (path === '/fixture/logits.js') { response.end('export * from "/src/inference/pipelines/text/logits/index.js"; export const recordLogitsGPU = async () => globalThis.__samplingFixture.logits();'); return; }
        const file = resolve(directory, '.' + path); assert(file.startsWith(directory + sep));
        let bytes = await readFile(file); receipt.files[path] = hash(bytes);
        if (path.endsWith('/generator-steps.js')) {
          bytes = Buffer.from(bytes.toString().replace("from './embed.js'", "from '/fixture/embed.js'")
            .replace("from './logits/index.js'", "from '/fixture/logits.js'"));
          receipt.transformedBatchOwnerSha256 = hash(bytes);
        }
        response.end(bytes);
      } catch (error) { response.statusCode = 404; response.end(error.message); }
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    browser = await chromium.launch({ channel: 'chrome', headless: true,
      args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'] });
    receipt.browser = browser.version();
    const system = await browser.newBrowserCDPSession();
    receipt.systemInfo = await system.send('SystemInfo.getInfo');
    await system.detach();
    const page = await browser.newPage(); await page.goto(`http://127.0.0.1:${server.address().port}`);
    receipt.result = await page.evaluate(async () => {
      globalThis.__DOPPLER_KERNEL_BASE_PATH__ = '/src/gpu/kernels/';
      return (await import('/audit.js')).auditSeededBatches();
    });
    await writeFile(output, JSON.stringify(receipt, null, 2) + '\n');
    assert(receipt.result.qualified, receipt.result.failure);
    console.log(JSON.stringify({ commit: receipt.commit, browser: receipt.browser, ...receipt.result }));
  } catch (error) {
    receipt.failure = error.stack; await writeFile(output, JSON.stringify(receipt, null, 2) + '\n'); throw error;
  } finally { await browser?.close(); if (server) await new Promise(resolve => server.close(resolve)); await rm(directory, { recursive: true, force: true }); }
}
