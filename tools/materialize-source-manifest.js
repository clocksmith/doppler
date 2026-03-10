#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';

import { materializeSourceRuntimeManifest } from '../src/tooling/source-runtime-materializer.js';
import { resolveNodeSourceRuntimeBundle } from '../src/tooling/node-source-runtime.js';

function fail(message) {
  console.error(`[materialize-source-manifest] ${message}`);
  process.exit(1);
}

function parseArgs(argv) {
  const args = {
    inputPath: null,
    modelId: null,
    manifestPath: null,
    dryRun: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--model-id') {
      args.modelId = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--manifest' || arg === '--manifest-path') {
      args.manifestPath = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--dry-run') {
      args.dryRun = true;
      continue;
    }
    if (arg.startsWith('-')) {
      fail(`Unknown flag: ${arg}`);
    }
    if (!args.inputPath) {
      args.inputPath = arg;
      continue;
    }
    fail(`Unexpected positional argument: ${arg}`);
  }

  if (!args.inputPath) {
    fail(
      'Usage: node tools/materialize-source-manifest.js <source-path> ' +
      '[--model-id <id>] [--manifest <manifest.json>] [--dry-run]'
    );
  }

  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(args.inputPath);
  const stats = await fs.stat(inputPath).catch((error) => {
    fail(`Failed to stat inputPath "${inputPath}": ${error.message}`);
  });
  const artifactDir = stats.isDirectory() ? inputPath : path.dirname(inputPath);
  const manifestPath = path.resolve(args.manifestPath || path.join(artifactDir, 'manifest.json'));

  const bundle = await resolveNodeSourceRuntimeBundle({
    inputPath,
    modelId: args.modelId,
    verifyHashes: true,
  });
  if (!bundle) {
    fail(
      `No direct-source model detected at "${inputPath}". ` +
      'Expected a Safetensors directory, existing direct-source artifact, or .gguf file.'
    );
  }

  const materializedManifest = materializeSourceRuntimeManifest(bundle.manifest, artifactDir);
  const manifestJson = `${JSON.stringify(materializedManifest, null, 2)}\n`;

  if (args.dryRun) {
    process.stdout.write(manifestJson);
    return;
  }

  await fs.writeFile(manifestPath, manifestJson, 'utf8');
  console.log(
    `[materialize-source-manifest] wrote ${path.relative(process.cwd(), manifestPath)} ` +
    `for ${materializedManifest.modelId}`
  );
}

await main();
