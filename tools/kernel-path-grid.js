#!/usr/bin/env node


import { readdir, readFile, writeFile, mkdir } from 'fs/promises';
import { join, resolve } from 'path';
import { spawn } from 'child_process';
import process from 'process';
import { loadConfig } from '../cli/config/index.js';

const PROJECT_ROOT = resolve(new URL('.', import.meta.url).pathname, '..');
const KERNEL_PATH_DIR = join(PROJECT_ROOT, 'src', 'config', 'presets', 'kernel-paths');

function printHelp() {
  console.log(`
Kernel Path Grid Benchmark

Usage:
  doppler --config <ref>

Config requirements:
  model (string, required)
  cli (object, required)
  tools.kernelPathGrid.outputDir (string)
  tools.kernelPathGrid.profileDirBase (string|null)
  tools.kernelPathGrid.kernelPrefix (string|null)
  tools.kernelPathGrid.kernelPaths (string[]|null)
  tools.kernelPathGrid.includeF32A (boolean)

Notes:
  Kernel selection is config-only; this tool writes a config file per kernel path.
  Provide either kernelPrefix or kernelPaths (one required).
`);
}

function parseArgs(argv) {
  const opts = { config: null, help: false };
  let i = 0;
  while (i < argv.length) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      opts.help = true;
      i++;
      continue;
    }
    if (arg === '--config' || arg === '-c') {
      opts.config = argv[i + 1] || null;
      i += 2;
      continue;
    }
    if (!arg.startsWith('-') && !opts.config) {
      opts.config = arg;
      i++;
      continue;
    }
    console.error(`Unknown argument: ${arg}`);
    opts.help = true;
    break;
  }
  return opts;
}

function sanitize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/-+/g, '-');
}

function normalizeWeightTag(value) {
  if (!value) return null;
  const lower = String(value).toLowerCase();
  if (lower === 'q4_k_m' || lower === 'q4k' || lower === 'q4') return 'q4k';
  if (lower === 'fp16' || lower === 'f16' || lower === 'float16') return 'f16';
  return lower;
}

function inferKernelWeightType(id) {
  if (id.includes('-q4k-')) return 'q4k';
  if (id.includes('-f16-')) return 'f16';
  return null;
}

function assertObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
}

function assertString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string`);
  }
}

function assertStringArray(value, label) {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array`);
  }
  for (const entry of value) {
    if (typeof entry !== 'string' || entry.trim() === '') {
      throw new Error(`${label} entries must be non-empty strings`);
    }
  }
}

function assertStringOrNull(value, label) {
  if (value === null) return;
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string or null`);
  }
}

function assertBoolean(value, label) {
  if (typeof value !== 'boolean') {
    throw new Error(`${label} must be a boolean`);
  }
}

function assertNumber(value, label, { min = null } = {}) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    throw new Error(`${label} must be a number`);
  }
  if (min !== null && value < min) {
    throw new Error(`${label} must be >= ${min}`);
  }
}

function assertIntent(value) {
  const allowed = new Set(['calibrate', 'investigate']);
  if (!allowed.has(value)) {
    throw new Error('runtime.shared.tooling.intent must be "calibrate" or "investigate"');
  }
}

async function loadKernelPathIds(prefix, explicitList) {
  if (explicitList?.length) return explicitList;
  const files = await readdir(KERNEL_PATH_DIR);
  const ids = [];
  for (const file of files) {
    if (!file.endsWith('.json')) continue;
    const data = await readFile(join(KERNEL_PATH_DIR, file), 'utf-8');
    const json = JSON.parse(data);
    if (json?.id) ids.push(json.id);
  }
  ids.sort();
  if (!prefix) return ids;
  return ids.filter((id) => id.startsWith(prefix));
}

function buildProfileDir(base, id) {
  if (!base) return null;
  if (base.includes('{id}')) {
    return base.replace('{id}', sanitize(id));
  }
  return `${base}-${sanitize(id)}`;
}

async function runBench(opts, kernelId, runtimeConfig) {
  const outputDir = resolve(PROJECT_ROOT, opts.outputDir);
  await mkdir(outputDir, { recursive: true });
  const outputFile = join(outputDir, `grid_${sanitize(opts.model)}_${sanitize(kernelId)}.json`);
  const configFile = join(outputDir, `kernel_path_${sanitize(kernelId)}.json`);
  const profileDir = buildProfileDir(opts.profileDirBase, kernelId);
  const runtime = {
    ...runtimeConfig,
    inference: {
      ...runtimeConfig.inference,
      kernelPath: kernelId,
    },
  };
  const configPayload = {
    model: opts.model,
    cli: {
      ...opts.cli,
      command: 'bench',
      suite: 'inference',
      output: outputFile,
      profileDir,
    },
    runtime,
  };
  await writeFile(configFile, JSON.stringify(configPayload, null, 2), 'utf-8');

  const args = [
    'cli/index.js',
    '--config',
    configFile,
  ];

  console.log(`\n==> ${kernelId}`);
  console.log(`    output: ${outputFile}`);

  const result = await new Promise((resolvePromise) => {
    const child = spawn(process.execPath, args, {
      stdio: 'inherit',
      cwd: PROJECT_ROOT,
      env: process.env,
    });
    child.on('close', (code) => resolvePromise(code ?? 1));
  });

  let metrics = null;
  try {
    const json = JSON.parse(await readFile(outputFile, 'utf-8'));
    metrics = json.metrics || null;
  } catch {
    // Ignore parse errors; report via status
  }

  return {
    kernelId,
    outputFile,
    status: result === 0 ? 'ok' : 'fail',
    metrics,
  };
}

function formatNumber(value) {
  if (value == null || Number.isNaN(value)) return 'n/a';
  return Math.round(value).toString();
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printHelp();
    process.exit(0);
  }
  if (!args.config) {
    console.error('Error: --config is required');
    console.error('Run with --help for usage.');
    process.exit(1);
  }

  const loadedConfig = await loadConfig(args.config);
  const raw = loadedConfig.raw ?? {};
  assertString(raw.model, 'model');

  assertObject(raw.cli, 'cli');
  const cli = raw.cli;
  const requiredCli = [
    'command', 'suite', 'baseUrl', 'noServer', 'headless', 'minimized',
    'reuseBrowser', 'cdpEndpoint', 'timeout', 'retries', 'profileDir',
    'output', 'html', 'compare', 'filter',
  ];
  for (const key of requiredCli) {
    if (!(key in cli)) throw new Error(`cli.${key} is required`);
  }
  assertString(cli.command, 'cli.command');
  assertStringOrNull(cli.suite, 'cli.suite');
  assertString(cli.baseUrl, 'cli.baseUrl');
  assertBoolean(cli.noServer, 'cli.noServer');
  assertBoolean(cli.headless, 'cli.headless');
  assertBoolean(cli.minimized, 'cli.minimized');
  assertBoolean(cli.reuseBrowser, 'cli.reuseBrowser');
  assertStringOrNull(cli.cdpEndpoint, 'cli.cdpEndpoint');
  assertNumber(cli.timeout, 'cli.timeout', { min: 1 });
  assertNumber(cli.retries, 'cli.retries', { min: 0 });
  assertStringOrNull(cli.profileDir, 'cli.profileDir');
  assertStringOrNull(cli.output, 'cli.output');
  assertStringOrNull(cli.html, 'cli.html');
  assertStringOrNull(cli.compare, 'cli.compare');
  assertStringOrNull(cli.filter, 'cli.filter');

  const runtimeConfig = loadedConfig.runtime;
  const intent = runtimeConfig?.shared?.tooling?.intent ?? null;
  if (!intent) {
    throw new Error('runtime.shared.tooling.intent is required');
  }
  assertIntent(intent);

  assertObject(raw.tools, 'tools');
  assertObject(raw.tools.kernelPathGrid, 'tools.kernelPathGrid');
  const tool = raw.tools.kernelPathGrid;
  assertString(tool.outputDir, 'tools.kernelPathGrid.outputDir');
  assertStringOrNull(tool.profileDirBase, 'tools.kernelPathGrid.profileDirBase');
  assertStringOrNull(tool.kernelPrefix, 'tools.kernelPathGrid.kernelPrefix');
  if (tool.kernelPaths !== null && tool.kernelPaths !== undefined) {
    assertStringArray(tool.kernelPaths, 'tools.kernelPathGrid.kernelPaths');
  }
  assertBoolean(tool.includeF32A, 'tools.kernelPathGrid.includeF32A');

  if (!tool.kernelPrefix && (!tool.kernelPaths || tool.kernelPaths.length === 0)) {
    throw new Error('tools.kernelPathGrid.kernelPrefix or kernelPaths is required');
  }

  const opts = {
    model: raw.model,
    cli,
    outputDir: tool.outputDir,
    profileDirBase: tool.profileDirBase,
    kernelPrefix: tool.kernelPrefix,
    kernelPaths: tool.kernelPaths ?? null,
    includeF32A: tool.includeF32A,
  };

  const kernelIds = await loadKernelPathIds(opts.kernelPrefix, opts.kernelPaths);
  if (!kernelIds.length) {
    console.error('No kernel paths found.');
    process.exit(1);
  }

  const manifestPath = join(PROJECT_ROOT, 'models', opts.model, 'manifest.json');
  let modelWeightType = null;
  try {
    const manifest = JSON.parse(await readFile(manifestPath, 'utf-8'));
    modelWeightType = normalizeWeightTag(manifest?.quantizationInfo?.weights || manifest?.quantization);
  } catch (err) {
    console.warn(`Warning: Failed to read manifest at ${manifestPath}: ${ (err).message}`);
  }

  const filteredKernelIds = kernelIds.filter((id) => {
    if (!opts.includeF32A && id.includes('f32a')) return false;
    if (!modelWeightType) return true;
    const kernelWeightType = inferKernelWeightType(id);
    if (!kernelWeightType) return true;
    return kernelWeightType === modelWeightType;
  });

  const skipped = kernelIds.filter((id) => !filteredKernelIds.includes(id));

  if (!filteredKernelIds.length) {
    console.error('No compatible kernel paths found after filtering.');
    process.exit(1);
  }

  console.log(`\nKernel grid for model: ${opts.model}`);
  console.log(`Base config: ${loadedConfig.chain.join(' -> ')}`);
  console.log(`Kernel paths: ${filteredKernelIds.join(', ')}`);
  if (skipped.length) {
    console.log(`Skipped: ${skipped.join(', ')}`);
  }

  const results = [];
  for (const kernelId of filteredKernelIds) {
    const result = await runBench(opts, kernelId, runtimeConfig);
    results.push(result);
  }

  console.log('\nSummary:');
  console.log('kernelPath | status | ttft_ms | prefill_tok_s | decode_tok_s | output');
  for (const result of results) {
    const metrics = result.metrics;
    console.log([
      result.kernelId,
      result.status,
      formatNumber(metrics?.ttft_ms),
      formatNumber(metrics?.prefill_tokens_per_sec),
      formatNumber(metrics?.decode_tokens_per_sec),
      result.outputFile,
    ].join(' | '));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
