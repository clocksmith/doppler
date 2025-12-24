#!/usr/bin/env node
/**
 * Update manifest settings without touching shards.
 *
 * Safe edits (default):
 * - optimizations.kernelHints
 * - optimizations.attentionKernel
 *
 * Unsafe edits (require --allow-unsafe):
 * - config.q4kLayout
 * - defaultWeightLayout
 */

import { readFile, writeFile, stat } from 'fs/promises';
import { resolve, join } from 'path';

interface UpdateOptions {
  input: string | null;
  computePrecision: string | null;
  q4kMatmul: string | null;
  f16Matmul: string | null;
  attentionPrefill: string | null;
  attentionDecode: string | null;
  attentionKernel: string | null;
  tunedDevice: string | null;
  benchmarkTokPerSec: number | null;
  clearKernelHints: boolean;
  q4kLayout: string | null;
  defaultWeightLayout: string | null;
  allowUnsafe: boolean;
  dryRun: boolean;
  help: boolean;
}

function parseArgs(args: string[]): UpdateOptions {
  const options: UpdateOptions = {
    input: null,
    computePrecision: null,
    q4kMatmul: null,
    f16Matmul: null,
    attentionPrefill: null,
    attentionDecode: null,
    attentionKernel: null,
    tunedDevice: null,
    benchmarkTokPerSec: null,
    clearKernelHints: false,
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
      case '--compute-precision':
        options.computePrecision = args[++i] || null;
        break;
      case '--q4k-matmul':
        options.q4kMatmul = args[++i] || null;
        break;
      case '--f16-matmul':
        options.f16Matmul = args[++i] || null;
        break;
      case '--attention-prefill':
        options.attentionPrefill = args[++i] || null;
        break;
      case '--attention-decode':
        options.attentionDecode = args[++i] || null;
        break;
      case '--attention-kernel':
        options.attentionKernel = args[++i] || null;
        break;
      case '--tuned-device':
        options.tunedDevice = args[++i] || null;
        break;
      case '--benchmark-tokps':
        options.benchmarkTokPerSec = Number(args[++i]);
        break;
      case '--clear-kernel-hints':
        options.clearKernelHints = true;
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

function printHelp(): void {
  console.log(`
Update DOPPLER manifest settings (no shard changes).

Usage:
  node update-manifest.js <model-dir|manifest.json> [options]

Safe options:
  --compute-precision <auto|f16|f32>
  --q4k-matmul <auto|fused_q4k|dequant_f16|dequant_f32>
  --f16-matmul <auto|gemv_subgroup>
  --attention-prefill <auto|tiled_large|tiled_small|streaming>
  --attention-decode <auto|tiled_large|tiled_small|streaming>
  --attention-kernel <auto|tiled_large|tiled_small|streaming>
  --tuned-device <string>
  --benchmark-tokps <number>
  --clear-kernel-hints
  --dry-run

Unsafe options (require --allow-unsafe):
  --q4k-layout <flat|row_wise|column_wise>
  --default-weight-layout <row|column>

Examples:
  node update-manifest.js ./models/gemma-1b-q4-row --q4k-matmul fused_q4k
  node update-manifest.js ./models/gemma-1b-q4-row --compute-precision f16 --attention-decode streaming
`);
}

async function resolveManifestPath(input: string): Promise<string> {
  const resolved = resolve(input);
  const stats = await stat(resolved);
  if (stats.isDirectory()) {
    return join(resolved, 'manifest.json');
  }
  return resolved;
}

async function main(): Promise<void> {
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

  if (options.clearKernelHints) {
    if (manifest.optimizations?.kernelHints) {
      delete manifest.optimizations.kernelHints;
      changed = true;
    }
  }

  const hasHintUpdates =
    options.computePrecision ||
    options.q4kMatmul ||
    options.f16Matmul ||
    options.attentionPrefill ||
    options.attentionDecode ||
    options.tunedDevice ||
    options.benchmarkTokPerSec !== null;

  if (hasHintUpdates) {
    manifest.optimizations = manifest.optimizations || {};
    manifest.optimizations.kernelHints = manifest.optimizations.kernelHints || {};

    if (options.computePrecision) {
      manifest.optimizations.kernelHints.computePrecision = options.computePrecision;
    }
    if (options.q4kMatmul) {
      manifest.optimizations.kernelHints.q4kMatmul = options.q4kMatmul;
    }
    if (options.f16Matmul) {
      manifest.optimizations.kernelHints.f16Matmul = options.f16Matmul;
    }
    if (options.attentionPrefill) {
      manifest.optimizations.kernelHints.attentionPrefill = options.attentionPrefill;
    }
    if (options.attentionDecode) {
      manifest.optimizations.kernelHints.attentionDecode = options.attentionDecode;
    }
    if (options.tunedDevice) {
      manifest.optimizations.kernelHints.tunedDevice = options.tunedDevice;
    }
    if (options.benchmarkTokPerSec !== null) {
      manifest.optimizations.kernelHints.benchmarkTokPerSec = options.benchmarkTokPerSec;
    }
    changed = true;
  }

  if (options.attentionKernel) {
    manifest.optimizations = manifest.optimizations || {};
    if (options.attentionKernel === 'auto') {
      delete manifest.optimizations.attentionKernel;
    } else {
      manifest.optimizations.attentionKernel = options.attentionKernel;
    }
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
