#!/usr/bin/env node

/**
 * Doppler Forge: Ahead-of-Time (AOT) Model Pack Compiler
 *
 * Transforms model manifests, conversion configs, and reference reports into
 * sealed, self-contained Doppler Packs (Program Bundles) containing only the
 * reachable WGSL kernel closure, execution-v1 DAG, artifact hashes, and
 * preflighted memory profiles.
 *
 * @module tools/forge-model-pack
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import {
  createProgramBundleCliDefaults,
  writeProgramBundle,
  verifyClosedProgramBundle,
} from '../src/tooling/program-bundle.js';

export const FORGE_VERSION = '1.0.0';

export function usage() {
  return [
    'Doppler Forge: Ahead-of-Time (AOT) Model Pack Compiler',
    '',
    'Usage:',
    '  node tools/forge-model-pack.js --manifest <path> --reference-report <path> --out <path> [--conversion-config <path>]',
    '  node tools/forge-model-pack.js --config <path|json>',
    '',
    'Flags:',
    '  --manifest <path>           Path to model manifest.json',
    '  --reference-report <path>   Path to reference report.json containing execution transcript',
    '  --conversion-config <path>  Path to conversion configuration json',
    '  --runtime-config <path>     Path to runtime configuration json',
    '  --model-dir <path>          Model directory root (defaults to manifest directory)',
    '  --out <path>                Output path for compiled .program-bundle.json / .pack',
    '  --bundle-id <string>        Explicit bundle ID override',
    '  --created-at <iso8601>      Explicit creation timestamp override',
    '  --config <path|json>        Inline JSON or config file containing all options',
    '  --json                      Emit machine-readable JSON output',
    '  --help, -h                  Show this help message',
  ].join('\n');
}

export function parseArgs(argv) {
  const flags = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--help' || token === '-h') {
      flags.help = true;
      continue;
    }
    if (token === '--json') {
      flags.json = true;
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

export async function readJsonInput(value) {
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

export async function buildForgeOptions(flags, metaUrl = import.meta.url) {
  const defaults = createProgramBundleCliDefaults(metaUrl);
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

/**
 * Compiles an AOT specialized Doppler Pack from manifest and reference inputs.
 *
 * @param {object} options
 * @returns {Promise<object>} Forge receipt summary
 */
export async function forgeModelPack(options) {
  const writeResult = await writeProgramBundle(options);
  const bundle = writeResult.bundle;

  // Verify that the written pack satisfies the closed-program contract
  const verification = await verifyClosedProgramBundle(writeResult.outputPath, bundle);
  if (!verification || verification.ok !== true) {
    throw new Error('Forge pack verification failed: bundle does not meet closed-program invariants.');
  }

  const relativeOut = path.relative(process.cwd(), writeResult.outputPath);
  return {
    ok: true,
    forgeVersion: FORGE_VERSION,
    outputPath: relativeOut.startsWith('..') ? writeResult.outputPath : relativeOut,
    absoluteOutputPath: writeResult.outputPath,
    modelId: bundle.modelId,
    bundleId: bundle.bundleId,
    schema: bundle.schema,
    schemaVersion: bundle.schemaVersion,
    createdAtUtc: bundle.createdAtUtc,
    executionGraphHash: bundle.sources.executionGraph.hash,
    artifactCount: bundle.artifacts.length,
    wgslModuleCount: bundle.wgslModules.length,
    reachableKernelDigests: bundle.wgslModules.map((m) => m.digest),
    packagedFiles: bundle.package?.files?.length ?? 0,
    referencePromptHash: bundle.referenceTranscript?.prompt?.hash ?? null,
  };
}

export async function main(argv = process.argv.slice(2)) {
  const flags = parseArgs(argv);
  if (flags.help) {
    console.log(usage());
    return;
  }
  const options = await buildForgeOptions(flags);
  const receipt = await forgeModelPack(options);

  if (flags.json) {
    console.log(JSON.stringify(receipt, null, 2));
  } else {
    console.log('✔ Doppler Forge: Pack compiled successfully');
    console.log(`  Model ID:             ${receipt.modelId}`);
    console.log(`  Bundle ID:            ${receipt.bundleId}`);
    console.log(`  Execution Graph Hash: ${receipt.executionGraphHash}`);
    console.log(`  Reachable WGSLs:      ${receipt.wgslModuleCount} modules`);
    console.log(`  Bundled Artifacts:    ${receipt.artifactCount} artifacts`);
    console.log(`  Output Pack:          ${receipt.outputPath}`);
  }
}

function isMainModule(metaUrl) {
  const entryPath = process.argv[1];
  return entryPath && path.resolve(fileURLToPath(metaUrl)) === path.resolve(entryPath);
}

if (isMainModule(import.meta.url)) {
  main().catch((error) => {
    console.error(`[doppler-forge] ${error.message}`);
    process.exit(1);
  });
}
