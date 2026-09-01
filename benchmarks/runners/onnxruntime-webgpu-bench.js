#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

function usage() {
  return [
    'Usage: node benchmarks/runners/onnxruntime-webgpu-bench.js --model <model.onnx> --inputs <inputs.json> --workload <id> [options]',
    '',
    'Options:',
    '  --model <path>               ONNX model file. Required.',
    '  --inputs <path>              JSON input tensor map. Required.',
    '  --workload <id>              Shared workload identity. Required.',
    '  --warmup <n>                 Warmup executions. Default: 3.',
    '  --runs <n>                   Timed executions. Default: 15.',
    '  --cache-mode <cold|warm>     Cache declaration. Default: warm.',
    '  --browser-executable <path>  Chromium executable path.',
    '  --json                       Accepted for runner parity.',
  ].join('\n');
}

function parseArgs(argv) {
  const flags = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--help' || token === '-h') return { help: true };
    if (token === '--json') {
      flags.json = true;
      continue;
    }
    if (!token.startsWith('--')) throw new Error(`Unexpected positional argument: ${token}`);
    const value = argv[index + 1];
    if (value == null || value.startsWith('--')) throw new Error(`Missing value for ${token}`);
    flags[token.slice(2)] = value;
    index += 1;
  }
  return flags;
}

function positiveInteger(value, label, fallback) {
  const parsed = value == null ? fallback : Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) throw new Error(`${label} must be a non-negative integer`);
  return parsed;
}

function existingFile(value, label) {
  const resolved = path.resolve(String(value || ''));
  if (!value || !fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
    throw new Error(`${label} must be an existing file: ${resolved}`);
  }
  return resolved;
}

function sha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function serve(root, modelPath, inputPath) {
  const modelBytes = fs.readFileSync(modelPath);
  const inputBytes = fs.readFileSync(inputPath);
  const server = http.createServer((request, response) => {
    const requestPath = new URL(request.url, 'http://127.0.0.1').pathname;
    let body;
    let contentType = 'text/html; charset=utf-8';
    if (requestPath === '/runner.html') body = fs.readFileSync(path.join(root, 'benchmarks/runners/onnxruntime-webgpu-runner.html'));
    else if (requestPath === '/model.onnx') { body = modelBytes; contentType = 'application/octet-stream'; }
    else if (requestPath === '/inputs.json') { body = inputBytes; contentType = 'application/json'; }
    else if (requestPath.startsWith('/node_modules/')) {
      const relative = requestPath.slice('/node_modules/'.length);
      const resolved = path.resolve(root, 'node_modules', relative);
      if (!resolved.startsWith(path.resolve(root, 'node_modules') + path.sep) || !fs.existsSync(resolved)) {
        response.writeHead(404); response.end(); return;
      }
      body = fs.readFileSync(resolved);
      contentType = resolved.endsWith('.mjs') || resolved.endsWith('.js') ? 'text/javascript; charset=utf-8' : 'application/octet-stream';
    } else { response.writeHead(404); response.end(); return; }
    response.writeHead(200, { 'content-type': contentType, 'cache-control': 'no-store', 'access-control-allow-origin': '*' });
    response.end(body);
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port })));
}

function percentile(values, ratio) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * ratio) - 1)];
}

async function main(argv = process.argv.slice(2)) {
  const flags = parseArgs(argv);
  if (flags.help) { console.log(usage()); return; }
  const modelPath = existingFile(flags.model, '--model');
  const inputPath = existingFile(flags.inputs, '--inputs');
  const workloadId = String(flags.workload || '').trim();
  if (!workloadId) throw new Error('--workload is required');
  const warmupRuns = positiveInteger(flags.warmup, '--warmup', 3);
  const timedRuns = positiveInteger(flags.runs, '--runs', 15);
  if (timedRuns < 1) throw new Error('--runs must be at least 1');
  const cacheMode = String(flags['cache-mode'] || 'warm');
  if (!['cold', 'warm'].includes(cacheMode)) throw new Error('--cache-mode must be cold or warm');
  const { server, port } = await serve(ROOT, modelPath, inputPath);
  try {
    const { chromium } = await import('playwright');
    const launchOptions = {
      headless: true,
      args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist', '--disable-vulkan-surface'],
    };
    if (flags['browser-executable']) launchOptions.executablePath = existingFile(flags['browser-executable'], '--browser-executable');
    const browser = await chromium.launch(launchOptions);
    try {
      const page = await browser.newPage();
      await page.goto(`http://127.0.0.1:${port}/runner.html?workload=${encodeURIComponent(workloadId)}&warmup=${warmupRuns}&runs=${timedRuns}&cacheMode=${cacheMode}`, { waitUntil: 'load' });
      const result = await page.evaluate(async () => {
        if (!globalThis.__ortBenchReady) throw new Error(globalThis.__ortBenchError || 'ORT runner failed to initialize');
        return globalThis.__runOrtBench();
      });
      const output = {
        schemaVersion: 1,
        workloadId,
        cacheMode,
        model: { path: path.relative(ROOT, modelPath).split(path.sep).join('/'), sha256: sha256(modelPath) },
        inputs: { path: path.relative(ROOT, inputPath).split(path.sep).join('/'), sha256: sha256(inputPath) },
        runtime: result.runtime,
        hardware: result.hardware,
        correctness: result.correctness,
        timing: result.timing,
        statistics: {
          raw: result.timing.samples,
          p50: percentile(result.timing.samples, 0.50),
          p95: percentile(result.timing.samples, 0.95),
          p99: percentile(result.timing.samples, 0.99),
        },
      };
      console.log(JSON.stringify(output, null, 2));
    } finally { await browser.close(); }
  } finally { await new Promise((resolve) => server.close(resolve)); }
}

main().catch((error) => { console.error(`[onnxruntime-webgpu-bench] ${error.message}`); process.exitCode = 1; });
