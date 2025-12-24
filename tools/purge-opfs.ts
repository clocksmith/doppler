#!/usr/bin/env node
/**
 * Purge a model from OPFS cache (browser storage).
 * Requires a browser context because OPFS is origin-scoped.
 */

import { resolve } from 'path';
import type { CLIOptions } from './cli/types.js';
import { ensureServerRunning, createBrowserContext, setupPage } from './cli/utils.js';

interface PurgeOptions {
  model: string | null;
  baseUrl: string;
  headless: boolean;
  noServer: boolean;
  profileDir: string | null;
  verbose: boolean;
  help: boolean;
}

function parseArgs(argv: string[]): PurgeOptions {
  const opts: PurgeOptions = {
    model: null,
    baseUrl: 'http://localhost:8080',
    headless: true,
    noServer: false,
    profileDir: null,
    verbose: false,
    help: false,
  };

  let i = 0;
  while (i < argv.length) {
    const arg = argv[i];
    switch (arg) {
      case '--help':
      case '-h':
        opts.help = true;
        break;
      case '--model':
      case '-m':
        opts.model = argv[++i] || null;
        break;
      case '--base-url':
      case '-u':
        opts.baseUrl = argv[++i] || opts.baseUrl;
        break;
      case '--headless':
        opts.headless = true;
        break;
      case '--headed':
        opts.headless = false;
        break;
      case '--no-server':
        opts.noServer = true;
        break;
      case '--profile-dir':
        opts.profileDir = argv[++i] || null;
        break;
      case '--verbose':
      case '-v':
        opts.verbose = true;
        break;
      default:
        if (!arg.startsWith('-') && !opts.model) {
          opts.model = arg;
        }
        break;
    }
    i++;
  }

  return opts;
}

function printHelp(): void {
  console.log(`
Purge a model from OPFS cache.

Usage:
  node purge-opfs.js --model <model-id> [options]

Options:
  --model, -m <id>      Model ID to delete (required)
  --base-url, -u <url>  Base URL (default: http://localhost:8080)
  --no-server           Serve assets from disk via Playwright routing
  --profile-dir <path>  Persistent browser profile dir (OPFS cache)
  --headless            Run headless (default)
  --headed              Run with visible browser window
  --verbose, -v         Verbose logging
  --help, -h            Show this help
`);
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) {
    printHelp();
    process.exit(0);
  }
  if (!opts.model) {
    console.error('Error: --model is required');
    process.exit(1);
  }

  if (!opts.noServer) {
    await ensureServerRunning(opts.baseUrl, opts.verbose);
  } else {
    console.log('No-server mode enabled (serving assets from disk)...');
  }

  const cliOptions: CLIOptions = {
    command: 'test',
    suite: 'quick',
    model: opts.model,
    baseUrl: opts.baseUrl,
    noServer: opts.noServer,
    headless: opts.headless,
    verbose: opts.verbose,
    filter: null,
    timeout: 60000,
    output: null,
    html: null,
    warmup: 0,
    runs: 1,
    maxTokens: 1,
    temperature: 0.7,
    prompt: 'xs',
    text: null,
    file: null,
    compare: null,
    trace: null,
    traceLayers: null,
    debugLayers: null,
    profileDir: opts.profileDir,
    retries: 0,
    quiet: true,
    help: false,
    gpuProfile: false,
    computePrecision: null,
    q4kMatmul: null,
    f16Matmul: null,
    attentionPrefill: null,
    attentionDecode: null,
    attentionKernel: null,
    kernelHints: null,
    layer: null,
    tokens: null,
    kernel: null,
  };

  const context = await createBrowserContext(cliOptions, { scope: 'test' });
  const page = await setupPage(context, cliOptions);

  try {
    const url = `${opts.baseUrl}/d`;
    await page.goto(url, { timeout: 30000 });
    const deleted = await page.evaluate(async (modelId: string) => {
      const mod = await import('/doppler/dist/storage/shard-manager.js');
      await mod.initOPFS();
      return mod.deleteModel(modelId);
    }, opts.model);

    if (deleted) {
      console.log(`OPFS purge complete: ${opts.model}`);
    } else {
      console.log(`No OPFS entry found for: ${opts.model}`);
    }
  } finally {
    await context.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
