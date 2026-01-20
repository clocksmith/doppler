#!/usr/bin/env node


import { writeFile, mkdir } from 'fs/promises';
import { resolve, join } from 'path';
import process from 'process';
import { createBrowserContext, setupPage } from '../cli/helpers/utils.js';
import { loadConfig } from '../cli/config/index.js';

const PROJECT_ROOT = resolve(new URL('.', import.meta.url).pathname, '..');

function printHelp() {
  console.log(`
Attention Decode Microbench

Usage:
  doppler --config <ref>

Config requirements:
  tools.attentionDecodeGrid.variants (string[])
  tools.attentionDecodeGrid.kvLens (number[])
  tools.attentionDecodeGrid.headDim (number)
  tools.attentionDecodeGrid.numHeads (number)
  tools.attentionDecodeGrid.numKVHeads (number)
  tools.attentionDecodeGrid.outputDir (string)
  cli.baseUrl (string)
  cli.noServer (boolean)
  cli.headless (boolean)
  cli.timeout (number)

Options:
  --config, -c <ref>    Config preset or path
  --help, -h            Show help
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

function assertString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string`);
  }
}

function assertNumber(value, label) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    throw new Error(`${label} must be a number`);
  }
}

function assertNumberArray(value, label) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${label} must be a non-empty array`);
  }
  for (const entry of value) {
    if (typeof entry !== 'number' || Number.isNaN(entry)) {
      throw new Error(`${label} entries must be numbers`);
    }
  }
}

function assertStringArray(value, label) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${label} must be a non-empty array`);
  }
  for (const entry of value) {
    if (typeof entry !== 'string' || entry.trim() === '') {
      throw new Error(`${label} entries must be non-empty strings`);
    }
  }
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
  const parsed = parseArgs(process.argv.slice(2));
  if (parsed.help) {
    printHelp();
    process.exit(0);
  }
  if (!parsed.config) {
    console.error('Error: --config is required');
    printHelp();
    process.exit(1);
  }

  const loadedConfig = await loadConfig(parsed.config);
  const raw = loadedConfig.raw ?? {};
  const toolConfig = raw.tools?.attentionDecodeGrid;
  const cli = raw.cli ?? null;

  if (!toolConfig || typeof toolConfig !== 'object') {
    throw new Error('tools.attentionDecodeGrid is required in config');
  }
  if (!cli) {
    throw new Error('cli is required in config');
  }

  assertStringArray(toolConfig.variants, 'tools.attentionDecodeGrid.variants');
  assertNumberArray(toolConfig.kvLens, 'tools.attentionDecodeGrid.kvLens');
  assertNumber(toolConfig.headDim, 'tools.attentionDecodeGrid.headDim');
  assertNumber(toolConfig.numHeads, 'tools.attentionDecodeGrid.numHeads');
  assertNumber(toolConfig.numKVHeads, 'tools.attentionDecodeGrid.numKVHeads');
  assertString(toolConfig.outputDir, 'tools.attentionDecodeGrid.outputDir');
  assertString(cli.baseUrl, 'cli.baseUrl');
  assertNumber(cli.timeout, 'cli.timeout');
  const runtimeConfig = loadedConfig.runtime;
  const configChain = loadedConfig.chain ?? null;
  const benchmarkRun = runtimeConfig.shared.benchmark.run;
  const outputDir = resolve(PROJECT_ROOT, toolConfig.outputDir);
  await mkdir(outputDir, { recursive: true });

  const cliOpts = {
    headless: cli.headless,
    minimized: cli.minimized,
    reuseBrowser: false,
    cdpEndpoint: cli.cdpEndpoint,
    timeout: cli.timeout,
    verbose: true,
    quiet: false,
    noServer: cli.noServer,
    baseUrl: cli.baseUrl,
    profileDir: null,
    platform: null,
  };

  const context = await createBrowserContext(cliOpts, { scope: 'test' });
  const page = await setupPage(context, cliOpts);

  const harness = runtimeConfig.shared?.harness;
  if (!harness) {
    throw new Error('runtime.shared.harness is required for attention grid benchmarks.');
  }
  Object.assign(harness, {
    mode: 'kernels',
    autorun: false,
    skipLoad: false,
    modelId: null,
  });

  const params = new URLSearchParams();
  params.set('runtimeConfig', JSON.stringify(runtimeConfig));
  if (configChain) {
    params.set('configChain', JSON.stringify(configChain));
  }

  const url = `${cli.baseUrl}/doppler/tests/harness.html?${params.toString()}`;
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.testHarness && typeof window.testHarness.benchmarkAttentionDecodeVariant === 'function',
    null,
    { timeout: cli.timeout }
  );

  const results = [];
  for (const kernel of toolConfig.variants) {
    const payload = {
      kernel,
      kvLens: toolConfig.kvLens,
      headDim: toolConfig.headDim,
      numHeads: toolConfig.numHeads,
      numKVHeads: toolConfig.numKVHeads,
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
    `attention_decode_grid_${sanitize(toolConfig.headDim)}_${timestamp}.json`
  );
  await writeFile(outputFile, JSON.stringify({
    schemaVersion: 1,
    timestamp: new Date().toISOString(),
    config: {
      headDim: toolConfig.headDim,
      numHeads: toolConfig.numHeads,
      numKVHeads: toolConfig.numKVHeads,
      kvLens: toolConfig.kvLens,
      warmupRuns: benchmarkRun.warmupRuns,
      timedRuns: benchmarkRun.timedRuns,
      variants: toolConfig.variants,
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
