#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  createProgramBundleCliDefaults,
  writeProgramBundle,
} from '../src/tooling/program-bundle.js';

function usage() {
  return [
    'Usage:',
    '  node tools/export-program-bundle.js --manifest <path> --reference-report <path> --out <path> [--conversion-config <path>]',
    '  node tools/export-program-bundle.js --config <path|json>',
    '',
    'Config fields:',
    '  manifestPath, modelDir, referenceReportPath, conversionConfigPath, runtimeConfigPath, outputPath, bundleId, createdAtUtc',
  ].join('\n');
}

function parseArgs(argv) {
  const flags = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--help' || token === '-h') {
      flags.help = true;
      continue;
    }
    if (!token.startsWith('--')) {
      throw new Error(`Unsupported positional argument "${token}".`);
    }
    const key = token.slice(2);
    const value = argv[index + 1];
    if (value === undefined || value.startsWith('--')) {
      throw new Error(`Missing value for --${key}.`);
    }
    flags[key] = value;
    index += 1;
  }
  return flags;
}

async function readJsonInput(value) {
  const normalized = String(value || '').trim();
  if (!normalized) {
    throw new Error('--config must be a JSON object or path.');
  }
  if (normalized.startsWith('{')) {
    return JSON.parse(normalized);
  }
  const raw = await fs.readFile(path.resolve(normalized), 'utf8');
  return JSON.parse(raw);
}

async function buildOptions(flags) {
  const defaults = createProgramBundleCliDefaults(import.meta.url);
  if (flags.config) {
    const config = await readJsonInput(flags.config);
    if (!config || typeof config !== 'object' || Array.isArray(config)) {
      throw new Error('--config must resolve to a JSON object.');
    }
    return {
      ...defaults,
      ...config,
      outputPath: config.outputPath ?? config.out ?? null,
    };
  }
  return {
    ...defaults,
    manifestPath: flags.manifest ?? null,
    modelDir: flags['model-dir'] ?? null,
    referenceReportPath: flags['reference-report'] ?? null,
    conversionConfigPath: flags['conversion-config'] ?? null,
    runtimeConfigPath: flags['runtime-config'] ?? null,
    outputPath: flags.out ?? null,
    bundleId: flags['bundle-id'] ?? null,
    createdAtUtc: flags['created-at'] ?? null,
  };
}

async function main() {
  const flags = parseArgs(process.argv.slice(2));
  if (flags.help) {
    console.log(usage());
    return;
  }
  const options = await buildOptions(flags);
  const result = await writeProgramBundle(options);
  console.log(JSON.stringify({
    ok: true,
    outputPath: path.relative(process.cwd(), result.outputPath),
    modelId: result.bundle.modelId,
    bundleId: result.bundle.bundleId,
    executionGraphHash: result.bundle.sources.executionGraph.hash,
    artifactCount: result.bundle.artifacts.length,
    wgslModuleCount: result.bundle.wgslModules.length,
  }, null, 2));
}

function isMainModule(metaUrl) {
  const entryPath = process.argv[1];
  return entryPath && path.resolve(fileURLToPath(metaUrl)) === path.resolve(entryPath);
}

if (isMainModule(import.meta.url)) {
  main().catch((error) => {
    console.error(`[program-bundle:export] ${error.message}`);
    process.exit(1);
  });
}
