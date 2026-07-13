#!/usr/bin/env node

import { createServer } from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, extname, resolve, sep } from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { sha256BytesHex } from '../src/utils/sha256.js';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const DEFAULT_OUTPUT = 'reports/training/native-parity/split-gate-up-lora-oracle.json';
const BROWSER_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-dawn-features=allow_unsafe_apis',
  '--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE',
  '--use-angle=vulkan',
  '--disable-vulkan-surface',
]);

function parseArgs(argv) {
  const args = { output: DEFAULT_OUTPUT };
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

async function createStaticServer() {
  const server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url, 'http://127.0.0.1');
      const relativePath = decodeURIComponent(url.pathname).replace(/^\/+/, '');
      if (!relativePath) {
        response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
        response.end('<!doctype html><meta charset="utf-8"><title>Doppler oracle</title>');
        return;
      }
      const path = resolve(ROOT, relativePath);
      if (path !== ROOT && !path.startsWith(`${ROOT}${sep}`)) {
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

async function fileHash(path) {
  return sha256BytesHex(new Uint8Array(await readFile(resolve(ROOT, path))));
}

async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const server = await createStaticServer();
  let browser = null;
  try {
    const { chromium } = await import('playwright');
    browser = await chromium.launch({ headless: true, args: [...BROWSER_ARGS] });
    const page = await browser.newPage();
    const address = server.address();
    const baseUrl = `http://127.0.0.1:${address.port}`;
    await page.goto(baseUrl);
    const oracle = await page.evaluate(async (moduleUrl) => {
      const module = await import(moduleUrl);
      return module.runSplitGateUpLoraOracle();
    }, `${baseUrl}/tests/training/browser/split-gate-up-lora-oracle.js`);
    const receipt = {
      ...oracle,
      sourceRevision: execFileSync('git', ['rev-parse', 'HEAD'], {
        cwd: ROOT,
        encoding: 'utf8',
      }).trim(),
      sourceDirty: execFileSync('git', ['status', '--short'], {
        cwd: ROOT,
        encoding: 'utf8',
      }).trim().length > 0,
      sourceHashes: {
        studentFixture: await fileHash('src/experimental/training/distillation/student-fixture.js'),
        autograd: await fileHash('src/experimental/training/autograd.js'),
        lora: await fileHash('src/experimental/training/lora.js'),
        oracle: await fileHash('tests/training/browser/split-gate-up-lora-oracle.js'),
      },
    };
    const outputPath = resolve(ROOT, args.output);
    await mkdir(dirname(outputPath), { recursive: true });
    await writeFile(outputPath, `${JSON.stringify(receipt, null, 2)}\n`, 'utf8');
    console.log(JSON.stringify({ ok: receipt.passed, outputPath, receipt }, null, 2));
    if (!receipt.passed) process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
    await new Promise((accept) => server.close(accept));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
