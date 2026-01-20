#!/usr/bin/env node


import { resolve } from 'path';
import { loadConfig } from '../cli/config/index.js';
import { ensureServerRunning, createBrowserContext, setupPage } from '../cli/helpers/utils.js';
import { DEFAULT_OPFS_PATH_CONFIG } from '../src/config/schema/loading.schema.js';


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


function printHelp() {
  console.log(`
Purge a model from OPFS cache.

Usage:
  doppler --config <ref>

Config requirements:
  model (string, required)
  cli.baseUrl (string, required)
  cli.noServer (boolean, required)
  cli.headless (boolean, required)
  cli.profileDir (string|null, required)

Options:
  --config, -c <ref>    Config preset or path
  --help, -h            Show this help
`);
}


async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) {
    printHelp();
    process.exit(0);
  }
  if (!opts.config) {
    console.error('Error: --config is required');
    process.exit(1);
  }

  const loaded = await loadConfig(opts.config);
  const raw = loaded.raw ?? {};
  const runtime = loaded.runtime ?? {};
  const modelId = raw.model;
  const cli = raw.cli ?? null;

  if (typeof modelId !== 'string' || modelId.trim() === '') {
    throw new Error('model is required in config');
  }
  if (!cli) {
    throw new Error('cli is required in config');
  }

  const verbose = runtime.shared?.debug?.logLevel?.defaultLogLevel === 'verbose';

  if (!cli.noServer) {
    await ensureServerRunning(cli.baseUrl, verbose);
  } else {
    console.log('No-server mode enabled (serving assets from disk)...');
  }

  
  const cliOptions = {
    command: 'test',
    suite: 'quick',
    model: modelId,
    baseUrl: cli.baseUrl,
    noServer: cli.noServer,
    headless: cli.headless,
    minimized: cli.minimized,
    reuseBrowser: cli.reuseBrowser,
    cdpEndpoint: cli.cdpEndpoint,
    verbose,
    timeout: cli.timeout,
    profileDir: cli.profileDir,
    quiet: true,
    help: false,
    perf: false,
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
    const deleted = await page.evaluate(async (params) => {
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
    }, { modelId, rootDir: opfsRootDir });

    if (deleted) {
      console.log(`OPFS purge complete: ${modelId}`);
    } else {
      console.log(`No OPFS entry found for: ${modelId}`);
    }
  } finally {
    await context.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
