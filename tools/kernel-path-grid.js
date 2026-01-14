#!/usr/bin/env node
/**
 * Kernel path benchmark grid runner.
 *
 * Runs `npm run bench -- inference` sequentially across kernel paths
 * and summarizes TTFT/prefill/decode throughput from saved JSON output.
 */

import { readdir, readFile, mkdir } from 'fs/promises';
import { join, resolve } from 'path';
import { spawn } from 'child_process';
import process from 'process';

const PROJECT_ROOT = resolve(new URL('.', import.meta.url).pathname, '..');
const KERNEL_PATH_DIR = join(PROJECT_ROOT, 'src', 'config', 'presets', 'kernel-paths');

const DEFAULTS = {
  model: 'gemma-2-2b-it-wf16',
  prompt: 'short',
  maxTokens: 128,
  runs: 3,
  retries: 2,
  noServer: true,
  profileDirBase: '.benchmark-grid-cache',
  outputDir: 'tests/results',
  kernelPrefix: null,
  kernelPaths: null,
  includeF32A: false,
};

function printHelp() {
  console.log(`
Kernel Path Grid Benchmark

Usage:
  node tools/kernel-path-grid.js [options]

Options:
  --model <id>            Model id (default: ${DEFAULTS.model})
  --prompt <name>         Prompt name (default: ${DEFAULTS.prompt})
  --max-tokens <n>        Max new tokens (default: ${DEFAULTS.maxTokens})
  --runs <n>              Timed runs (default: ${DEFAULTS.runs})
  --retries <n>           Retries per run (default: ${DEFAULTS.retries})
  --no-server             Use Playwright local routing (default: on)
  --profile-dir-base <p>  Profile dir base (default: ${DEFAULTS.profileDirBase})
  --output-dir <p>        Output directory (default: ${DEFAULTS.outputDir})
  --kernel-prefix <p>     Filter kernel path ids by prefix
  --kernel-paths <ids>    Comma-separated kernel path ids (overrides prefix)
  --include-f32a          Include f32 activation kernel paths
  --help                  Show help

Examples:
  node tools/kernel-path-grid.js --model gemma-2-2b-it-wf16 --kernel-prefix gemma2
  node tools/kernel-path-grid.js --kernel-paths gemma2-f16-f16a,gemma2-f16-f32a
`);
}

function parseArgs(argv) {
  const opts = { ...DEFAULTS };
  const args = [...argv];

  while (args.length) {
    const arg = args.shift();
    if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    }
    if (arg === '--no-server') {
      opts.noServer = true;
      continue;
    }
    if (arg === '--include-f32a') {
      opts.includeF32A = true;
      continue;
    }
    if (!arg || !arg.startsWith('--')) {
      console.error(`Unknown argument: ${arg}`);
      printHelp();
      process.exit(1);
    }
    const key = arg.slice(2);
    const value = args.shift();
    if (value == null) {
      console.error(`Missing value for --${key}`);
      process.exit(1);
    }
    switch (key) {
      case 'model':
        opts.model = value;
        break;
      case 'prompt':
        opts.prompt = value;
        break;
      case 'max-tokens':
        opts.maxTokens = Number(value);
        break;
      case 'runs':
        opts.runs = Number(value);
        break;
      case 'retries':
        opts.retries = Number(value);
        break;
      case 'profile-dir-base':
        opts.profileDirBase = value;
        break;
      case 'output-dir':
        opts.outputDir = value;
        break;
      case 'kernel-prefix':
        opts.kernelPrefix = value;
        break;
      case 'kernel-paths':
        opts.kernelPaths = value.split(',').map((v) => v.trim()).filter(Boolean);
        break;
      default:
        console.error(`Unknown option: --${key}`);
        process.exit(1);
    }
  }

  if (Number.isNaN(opts.maxTokens) || opts.maxTokens <= 0) {
    console.error('Invalid --max-tokens value');
    process.exit(1);
  }
  if (Number.isNaN(opts.runs) || opts.runs <= 0) {
    console.error('Invalid --runs value');
    process.exit(1);
  }
  if (Number.isNaN(opts.retries) || opts.retries < 0) {
    console.error('Invalid --retries value');
    process.exit(1);
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

function deriveKernelPrefix(modelId) {
  const lower = modelId.toLowerCase();
  if (lower.includes('gemma-2') || lower.includes('gemma2')) return 'gemma2';
  if (lower.includes('gemma-3') || lower.includes('gemma3')) return 'gemma3';
  if (lower.includes('llama-3') || lower.includes('llama3')) return 'llama3';
  if (lower.includes('qwen3')) return 'qwen3';
  return null;
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

async function runBench(opts, kernelId) {
  const outputDir = resolve(PROJECT_ROOT, opts.outputDir);
  await mkdir(outputDir, { recursive: true });
  const outputFile = join(outputDir, `grid_${sanitize(opts.model)}_${sanitize(kernelId)}.json`);
  const profileDir = buildProfileDir(opts.profileDirBase, kernelId);

  const args = [
    'run',
    'bench',
    '--',
    'inference',
    '--model',
    opts.model,
    '--prompt',
    opts.prompt,
    '--max-tokens',
    String(opts.maxTokens),
    '--runs',
    String(opts.runs),
    '--retries',
    String(opts.retries),
    '--kernel-path',
    kernelId,
    '--output',
    outputFile,
  ];

  if (opts.noServer) args.push('--no-server');
  if (profileDir) args.push('--profile-dir', profileDir);

  console.log(`\n==> ${kernelId}`);
  console.log(`    output: ${outputFile}`);

  const result = await new Promise((resolvePromise) => {
    const child = spawn('npm', args, {
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
  const opts = parseArgs(process.argv.slice(2));
  if (!opts.kernelPrefix && !opts.kernelPaths?.length) {
    opts.kernelPrefix = deriveKernelPrefix(opts.model);
  }

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
    console.warn(`Warning: Failed to read manifest at ${manifestPath}: ${/** @type {Error} */ (err).message}`);
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
  console.log(`Kernel paths: ${filteredKernelIds.join(', ')}`);
  if (skipped.length) {
    console.log(`Skipped: ${skipped.join(', ')}`);
  }

  const results = [];
  for (const kernelId of filteredKernelIds) {
    const result = await runBench(opts, kernelId);
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
