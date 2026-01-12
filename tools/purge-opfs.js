#!/usr/bin/env node
/**
 * Purge a model from OPFS cache (browser storage).
 * Requires a browser context because OPFS is origin-scoped.
 */

import { resolve } from 'path';
import { ensureServerRunning, createBrowserContext, setupPage } from '../cli/helpers/utils.js';
import { DEFAULT_OPFS_PATH_CONFIG } from '../src/config/schema/loading.schema.js';

/**
 * @param {string[]} argv
 * @returns {import('./purge-opfs.js').PurgeOptions}
 */
function parseArgs(argv) {
  /** @type {import('./purge-opfs.js').PurgeOptions} */
  const opts = {
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

/**
 * @returns {void}
 */
function printHelp() {
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

/**
 * @returns {Promise<void>}
 */
async function main() {
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

  /** @type {import('./cli/types.js').CLIOptions} */
  const cliOptions = {
    command: 'test',
    suite: 'quick',
    model: opts.model,
    baseUrl: opts.baseUrl,
    noServer: opts.noServer,
    headless: opts.headless,
    minimized: false,
    reuseBrowser: true,
    cdpEndpoint: 'http://localhost:9222',
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
    kernelProfile: null,
    kernelPath: null,
    perf: false,
    debug: false,
    layer: null,
    tokens: null,
    kernel: null,
  };

  const context = await createBrowserContext(cliOptions, { scope: 'test' });
  const page = await setupPage(context, cliOptions);

  try {
    const url = `${opts.baseUrl}/d`;
    await page.goto(url, { timeout: 30000 });
    const opfsRootDir = DEFAULT_OPFS_PATH_CONFIG.opfsRootDir || 'doppler-models';
    const deleted = await page.evaluate(async (/** @type {{ modelId: string; rootDir: string }} */ params) => {
      const { modelId, rootDir } = params;
      // Access OPFS directly in browser context
      const root = await navigator.storage.getDirectory();
      const tryDelete = async (dirHandle, entryName) => {
        try {
          await dirHandle.removeEntry(entryName, { recursive: true });
          return true;
        } catch {
          return false;
        }
      };

      if (await tryDelete(root, modelId)) {
        return true;
      }

      try {
        const modelsDir = await root.getDirectoryHandle(rootDir, { create: false });
        return await tryDelete(modelsDir, modelId);
      } catch {
        return false;
      }
    }, { modelId: opts.model, rootDir: opfsRootDir });

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
