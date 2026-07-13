import { createServer } from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, extname, resolve, sep } from 'node:path';
import { execFileSync } from 'node:child_process';

import { sha256BytesHex } from '../../src/utils/sha256.js';

const BROWSER_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-dawn-features=allow_unsafe_apis',
  '--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE',
  '--use-angle=vulkan',
  '--disable-vulkan-surface',
]);

function parseArgs(argv, defaultOutput) {
  const args = { output: defaultOutput };
  for (let index = 0; index < argv.length; index += 2) {
    if (argv[index] !== '--output' || !argv[index + 1]) {
      throw new Error(`${argv[index]} requires a value.`);
    }
    args.output = argv[index + 1];
  }
  return args;
}

function contentType(path) {
  if (extname(path) === '.js') return 'text/javascript; charset=utf-8';
  if (extname(path) === '.json') return 'application/json; charset=utf-8';
  if (extname(path) === '.wgsl') return 'text/plain; charset=utf-8';
  return 'text/html; charset=utf-8';
}

async function createStaticServer(root) {
  const server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url, 'http://127.0.0.1');
      const relativePath = decodeURIComponent(url.pathname).replace(/^\/+/, '');
      if (!relativePath) {
        response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
        response.end('<!doctype html><meta charset="utf-8"><title>Doppler oracle</title>');
        return;
      }
      const path = resolve(root, relativePath);
      if (path !== root && !path.startsWith(`${root}${sep}`)) {
        throw new Error('Path escapes repository root.');
      }
      const bytes = await readFile(path);
      response.writeHead(200, { 'Content-Type': contentType(path) });
      response.end(bytes);
    } catch (error) {
      response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      response.end(error?.message || String(error));
    }
  });
  await new Promise((accept, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', accept);
  });
  return server;
}

async function fileHash(root, path) {
  return sha256BytesHex(new Uint8Array(await readFile(resolve(root, path))));
}

export async function runBrowserOracle(options) {
  const {
    argv,
    root,
    defaultOutput,
    modulePath,
    exportName,
    sourcePaths,
  } = options;
  const args = parseArgs(argv, defaultOutput);
  const server = await createStaticServer(root);
  let browser = null;
  try {
    const { chromium } = await import('playwright');
    browser = await chromium.launch({ headless: true, args: [...BROWSER_ARGS] });
    const page = await browser.newPage();
    const address = server.address();
    const baseUrl = `http://127.0.0.1:${address.port}`;
    await page.goto(baseUrl);
    const oracle = await page.evaluate(async ({ moduleUrl, functionName }) => {
      const module = await import(moduleUrl);
      return module[functionName]();
    }, { moduleUrl: `${baseUrl}/${modulePath}`, functionName: exportName });
    const hashes = {};
    for (const [name, path] of Object.entries(sourcePaths)) {
      hashes[name] = await fileHash(root, path);
    }
    const receipt = {
      ...oracle,
      sourceRevision: execFileSync('git', ['rev-parse', 'HEAD'], {
        cwd: root,
        encoding: 'utf8',
      }).trim(),
      sourceDirty: execFileSync('git', ['status', '--short'], {
        cwd: root,
        encoding: 'utf8',
      }).trim().length > 0,
      sourceHashes: hashes,
    };
    const outputPath = resolve(root, args.output);
    await mkdir(dirname(outputPath), { recursive: true });
    await writeFile(outputPath, `${JSON.stringify(receipt, null, 2)}\n`, 'utf8');
    console.log(JSON.stringify({ ok: receipt.passed, outputPath, receipt }, null, 2));
    if (!receipt.passed) process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
    await new Promise((accept) => server.close(accept));
  }
}
