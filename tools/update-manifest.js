#!/usr/bin/env node


import { readFile, writeFile, stat } from 'fs/promises';
import { resolve, join } from 'path';
import { loadConfig } from '../cli/config/index.js';


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
Update DOPPLER manifest settings (no shard changes).

Usage:
  doppler --config <ref>

Config requirements:
  tools.updateManifest.input (string, required)
  tools.updateManifest.kernelPath (string|object|null)
  tools.updateManifest.clearKernelPath (boolean)
  tools.updateManifest.q4kLayout (string|null)
  tools.updateManifest.defaultWeightLayout (string|null)
  tools.updateManifest.allowUnsafe (boolean)
  tools.updateManifest.dryRun (boolean)

Notes:
  - kernelPath is a build-time manifest edit (not a runtime override)
  - q4kLayout/defaultWeightLayout require allowUnsafe=true
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

function assertKernelPath(value, label) {
  if (value === null) return;
  if (typeof value === 'string') {
    if (value.trim() === '') {
      throw new Error(`${label} must be a non-empty string, object, or null`);
    }
    return;
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be a non-empty string, object, or null`);
  }
}

function assertQ4KLayout(value, label) {
  if (value === null) return;
  assertString(value, label);
  const normalized = value.toLowerCase();
  if (normalized !== 'row' && normalized !== 'col') {
    throw new Error(`${label} must be "row", "col", or null`);
  }
}

function assertWeightLayout(value, label) {
  if (value === null) return;
  assertString(value, label);
  const normalized = value.toLowerCase();
  if (normalized !== 'row' && normalized !== 'column') {
    throw new Error(`${label} must be "row", "column", or null`);
  }
}


async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    process.exit(0);
  }

  if (!options.config) {
    console.error('Error: --config is required.');
    process.exit(1);
  }

  const loaded = await loadConfig(options.config);
  const raw = loaded.raw ?? {};
  const toolConfig = raw.tools?.updateManifest;

  if (!toolConfig || typeof toolConfig !== 'object') {
    throw new Error('tools.updateManifest is required in config');
  }

  const required = [
    'input',
    'kernelPath',
    'clearKernelPath',
    'q4kLayout',
    'defaultWeightLayout',
    'allowUnsafe',
    'dryRun',
  ];
  for (const key of required) {
    if (!(key in toolConfig)) {
      throw new Error(`tools.updateManifest.${key} is required in config`);
    }
  }

  assertString(toolConfig.input, 'tools.updateManifest.input');
  assertKernelPath(toolConfig.kernelPath, 'tools.updateManifest.kernelPath');
  assertBoolean(toolConfig.clearKernelPath, 'tools.updateManifest.clearKernelPath');
  assertQ4KLayout(toolConfig.q4kLayout, 'tools.updateManifest.q4kLayout');
  assertWeightLayout(toolConfig.defaultWeightLayout, 'tools.updateManifest.defaultWeightLayout');
  assertBoolean(toolConfig.allowUnsafe, 'tools.updateManifest.allowUnsafe');
  assertBoolean(toolConfig.dryRun, 'tools.updateManifest.dryRun');

  const manifestPath = await resolveManifestPath(toolConfig.input);
  const manifestJson = await readFile(manifestPath, 'utf-8');
  const manifest = JSON.parse(manifestJson);
  let changed = false;

  if (toolConfig.clearKernelPath) {
    if (manifest.optimizations?.kernelPath) {
      delete manifest.optimizations.kernelPath;
      changed = true;
    }
  }

  if (toolConfig.kernelPath !== null) {
    manifest.optimizations = manifest.optimizations || {};
    manifest.optimizations.kernelPath = toolConfig.kernelPath;
    changed = true;
  }

  if (toolConfig.q4kLayout) {
    if (!toolConfig.allowUnsafe) {
      console.warn('Skipping q4kLayout (set tools.updateManifest.allowUnsafe=true to apply).');
    } else {
      manifest.quantizationInfo = manifest.quantizationInfo || {};
      manifest.quantizationInfo.layout = toolConfig.q4kLayout;
      changed = true;
    }
  }

  if (toolConfig.defaultWeightLayout) {
    if (!toolConfig.allowUnsafe) {
      console.warn('Skipping defaultWeightLayout (set tools.updateManifest.allowUnsafe=true to apply).');
    } else {
      manifest.defaultWeightLayout = toolConfig.defaultWeightLayout;
      changed = true;
    }
  }

  if (!changed) {
    console.log('No changes requested.');
    return;
  }

  if (toolConfig.dryRun) {
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
