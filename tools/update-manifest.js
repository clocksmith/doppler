#!/usr/bin/env node


import { readFile, writeFile, stat } from 'fs/promises';
import { resolve, join } from 'path';


function parseArgs(args) {
  
  const options = {
    input: null,
    kernelPath: null,
    clearKernelPath: false,
    q4kLayout: null,
    defaultWeightLayout: null,
    allowUnsafe: false,
    dryRun: false,
    help: false,
  };

  let i = 0;
  while (i < args.length) {
    const arg = args[i];
    switch (arg) {
      case '--help':
      case '-h':
        options.help = true;
        break;
      case '--kernel-path': {
        const raw = args[++i] || '';
        if (!raw) break;
        if (raw.trim().startsWith('{')) {
          try {
            const parsed = JSON.parse(raw);
            if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
              throw new Error('kernel path must be a JSON object');
            }
            options.kernelPath =  (parsed);
          } catch (err) {
            throw new Error(`Failed to parse --kernel-path JSON: ${ (err).message}`);
          }
        } else {
          options.kernelPath = raw;
        }
        break;
      }
      case '--clear-kernel-path':
        options.clearKernelPath = true;
        break;
      case '--q4k-layout':
        options.q4kLayout = args[++i] || null;
        break;
      case '--default-weight-layout':
        options.defaultWeightLayout = args[++i] || null;
        break;
      case '--allow-unsafe':
        options.allowUnsafe = true;
        break;
      case '--dry-run':
        options.dryRun = true;
        break;
      default:
        if (!arg.startsWith('-') && !options.input) {
          options.input = arg;
        }
        break;
    }
    i++;
  }

  return options;
}


function printHelp() {
  console.log(`
Update DOPPLER manifest settings (no shard changes).

Usage:
  node update-manifest.js <model-dir|manifest.json> [options]

Safe options:
  --kernel-path <id|json>   Build-time manifest edit (not a runtime override)
  --clear-kernel-path
  --dry-run

Unsafe options (require --allow-unsafe):
  --q4k-layout <flat|row_wise|column_wise>
  --default-weight-layout <row|column>

Examples:
  node update-manifest.js ./models/gemma-1b-q4-row --kernel-path gemma2-q4k-fused
  node update-manifest.js ./models/gemma-1b-q4-row --kernel-path '{"id":"gemma2-q4k-dequant-f16"}'
`);
}


async function resolveManifestPath(input) {
  const resolved = resolve(input);
  const stats = await stat(resolved);
  if (stats.isDirectory()) {
    return join(resolved, 'manifest.json');
  }
  return resolved;
}


async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    process.exit(0);
  }

  if (!options.input) {
    console.error('Error: model directory or manifest.json path is required.');
    process.exit(1);
  }

  const manifestPath = await resolveManifestPath(options.input);
  const manifestJson = await readFile(manifestPath, 'utf-8');
  const manifest = JSON.parse(manifestJson);
  let changed = false;

  if (options.clearKernelPath) {
    if (manifest.optimizations?.kernelPath) {
      delete manifest.optimizations.kernelPath;
      changed = true;
    }
  }

  if (options.kernelPath !== null) {
    manifest.optimizations = manifest.optimizations || {};
    manifest.optimizations.kernelPath = options.kernelPath;
    changed = true;
  }

  if (options.q4kLayout) {
    if (!options.allowUnsafe) {
      console.warn('Skipping --q4k-layout (use --allow-unsafe to force).');
    } else {
      manifest.config = manifest.config || {};
      manifest.config.q4kLayout = options.q4kLayout;
      changed = true;
    }
  }

  if (options.defaultWeightLayout) {
    if (!options.allowUnsafe) {
      console.warn('Skipping --default-weight-layout (use --allow-unsafe to force).');
    } else {
      manifest.defaultWeightLayout = options.defaultWeightLayout;
      changed = true;
    }
  }

  if (!changed) {
    console.log('No changes requested.');
    return;
  }

  if (options.dryRun) {
    console.log(`Dry run: changes applied in memory for ${manifestPath}`);
    return;
  }

  await writeFile(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`Updated manifest: ${manifestPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
