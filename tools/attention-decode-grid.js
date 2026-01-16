#!/usr/bin/env node
/**
 * Attention decode microbench across kv lengths for specific kernel variants.
 */

import { writeFile, mkdir } from 'fs/promises';
import { resolve, join } from 'path';
import process from 'process';
import { createBrowserContext, setupPage } from '../cli/helpers/utils.js';
import { loadConfig } from '../cli/config/index.js';

const PROJECT_ROOT = resolve(new URL('.', import.meta.url).pathname, '..');

const DEFAULTS = {
  variants: [
    'attention_decode_chunked_f16.wgsl',
    'attention_streaming_f16.wgsl',
  ],
  kvLens: [128, 256, 512, 1024, 1536, 2048],
  headDim: 256,
  numHeads: 8,
  numKVHeads: 4,
  config: 'bench',
  outputDir: 'tests/results',
  headless: true,
  noServer: true,
  baseUrl: 'http://localhost:8080',
  timeout: 120000,
};

function printHelp() {
  console.log(`
Attention Decode Microbench

Usage:
  node tools/attention-decode-grid.js [options]

Options:
  --variants <list>     Comma-separated kernel files
  --kv-lens <list>      Comma-separated kv lengths
  --head-dim <n>        Head dimension (default: ${DEFAULTS.headDim})
  --num-heads <n>       Number of attention heads (default: ${DEFAULTS.numHeads})
  --num-kv-heads <n>    Number of KV heads (default: ${DEFAULTS.numKVHeads})
  --config <ref>        Runtime config preset or path (default: ${DEFAULTS.config})
  --output-dir <p>      Output directory (default: ${DEFAULTS.outputDir})
  --headed              Run headed (default: headless)
  --server              Use dev server instead of local routing
  --base-url <url>      Base URL (default: ${DEFAULTS.baseUrl})
  --timeout <ms>        Timeout in ms (default: ${DEFAULTS.timeout})
  --help                Show help
`);
}

function parseList(value) {
  return value
    .split(',')
    .map((v) => v.trim())
    .filter(Boolean);
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
    if (arg === '--headed') {
      opts.headless = false;
      continue;
    }
    if (arg === '--server') {
      opts.noServer = false;
      continue;
    }
    if (!arg.startsWith('--')) {
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
      case 'variants':
        opts.variants = parseList(value);
        break;
      case 'kv-lens':
        opts.kvLens = parseList(value).map(Number).filter((n) => Number.isFinite(n));
        break;
      case 'head-dim':
        opts.headDim = Number(value);
        break;
      case 'num-heads':
        opts.numHeads = Number(value);
        break;
      case 'num-kv-heads':
        opts.numKVHeads = Number(value);
        break;
      case 'config':
        opts.config = value;
        break;
      case 'output-dir':
        opts.outputDir = value;
        break;
      case 'base-url':
        opts.baseUrl = value;
        break;
      case 'timeout':
        opts.timeout = Number(value);
        break;
      default:
        console.error(`Unknown option: --${key}`);
        process.exit(1);
    }
  }

  if (!opts.kvLens.length) {
    console.error('No kv lengths specified.');
    process.exit(1);
  }

  return opts;
}

function sanitize(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/-+/g, '-');
}

function summarize(results) {
  const summary = {};
  for (const variant of results) {
    const rows = variant.results.map((entry) => ({
      kvLen: entry.kvLen,
      medianMs: entry.stats.medianMs,
      gflops: entry.stats.gflops,
    }));
    summary[variant.kernel] = rows;
  }
  return summary;
}

async function run() {
  const opts = parseArgs(process.argv.slice(2));
  const loadedConfig = await loadConfig(opts.config);
  const benchmarkRun = loadedConfig.runtime.shared.benchmark.run;
  const outputDir = resolve(PROJECT_ROOT, opts.outputDir);
  await mkdir(outputDir, { recursive: true });

  const cliOpts = {
    headless: opts.headless,
    minimized: false,
    reuseBrowser: false,
    cdpEndpoint: 'http://localhost:9222',
    timeout: opts.timeout,
    verbose: true,
    quiet: false,
    noServer: opts.noServer,
    baseUrl: opts.baseUrl,
    profileDir: null,
    platform: null,
  };

  const context = await createBrowserContext(cliOpts, { scope: 'test' });
  const page = await setupPage(context, cliOpts);

  const url = `${opts.baseUrl}/doppler/tests/harness.html?mode=kernels`;
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.testHarness && typeof window.testHarness.benchmarkAttentionDecodeVariant === 'function',
    null,
    { timeout: opts.timeout }
  );

  const results = [];
  for (const kernel of opts.variants) {
    const payload = {
      kernel,
      kvLens: opts.kvLens,
      headDim: opts.headDim,
      numHeads: opts.numHeads,
      numKVHeads: opts.numKVHeads,
      warmupRuns: benchmarkRun.warmupRuns,
      timedRuns: benchmarkRun.timedRuns,
    };
    const result = await page.evaluate(async (params) => {
      return window.testHarness.benchmarkAttentionDecodeVariant(null, params);
    }, payload);
    results.push(result);
  }

  await context.close();

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const outputFile = join(
    outputDir,
    `attention_decode_grid_${sanitize(opts.headDim)}_${timestamp}.json`
  );
  await writeFile(outputFile, JSON.stringify({
    schemaVersion: 1,
    timestamp: new Date().toISOString(),
    config: {
      headDim: opts.headDim,
      numHeads: opts.numHeads,
      numKVHeads: opts.numKVHeads,
      kvLens: opts.kvLens,
      warmupRuns: benchmarkRun.warmupRuns,
      timedRuns: benchmarkRun.timedRuns,
      variants: opts.variants,
    },
    results,
    summary: summarize(results),
  }, null, 2));

  console.log(`Saved results: ${outputFile}`);
  for (const variant of results) {
    console.log(`\n${variant.kernel}`);
    for (const row of variant.results) {
      console.log(`  kv=${row.kvLen} median=${row.stats.medianMs.toFixed(3)}ms gflops=${row.stats.gflops.toFixed(2)}`);
    }
  }
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
