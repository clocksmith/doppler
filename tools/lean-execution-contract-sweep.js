#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { runLeanExecutionContractForManifest } from '../src/tooling/lean-execution-contract-runner.js';

function parseArgs(argv) {
  const args = {
    root: 'models',
    json: false,
    check: true,
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--root') {
      args.root = argv[index + 1] ?? args.root;
      index += 1;
      continue;
    }
    if (arg === '--json') {
      args.json = true;
      continue;
    }
    if (arg === '--no-check') {
      args.check = false;
      continue;
    }
    if (arg === '--help' || arg === '-h') {
      args.help = true;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function usage() {
  return [
    'Usage:',
    '  node tools/lean-execution-contract-sweep.js [--root <dir>] [--json] [--no-check]',
  ].join('\n');
}

async function collectManifestPaths(rootDir) {
  const output = [];
  async function walk(currentDir) {
    let entries;
    try {
      entries = await fs.readdir(currentDir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const absolute = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        await walk(absolute);
        continue;
      }
      if (entry.isFile() && entry.name === 'manifest.json') {
        output.push(absolute);
      }
    }
  }
  await walk(rootDir);
  output.sort((left, right) => left.localeCompare(right));
  return output;
}

function isLeanExecutionContractManifest(manifest) {
  return manifest
    && typeof manifest === 'object'
    && manifest.modelType === 'transformer'
    && manifest.architecture
    && typeof manifest.architecture === 'object'
    && manifest.inference
    && typeof manifest.inference === 'object';
}

async function runSweep(rootDir, options = {}) {
  const manifestPaths = await collectManifestPaths(rootDir);
  const results = [];
  for (const manifestPath of manifestPaths) {
    const relativePath = path.relative(process.cwd(), manifestPath);
    try {
      const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
      if (!isLeanExecutionContractManifest(manifest)) {
        results.push({
          path: relativePath,
          modelId: manifest?.modelId ?? null,
          status: 'skipped',
          reason: 'manifest is not a transformer execution-contract candidate',
        });
        continue;
      }
      const result = runLeanExecutionContractForManifest(manifest, {
        rootDir: process.cwd(),
        check: options.check !== false,
      });
      results.push({
        path: relativePath,
        modelId: result.facts?.modelId ?? manifest.modelId ?? null,
        status: result.ok ? 'pass' : 'fail',
        toolchainRef: result.toolchainRef,
      });
    } catch (error) {
      results.push({
        path: relativePath,
        modelId: null,
        status: 'error',
        reason: error instanceof Error ? error.message : String(error),
      });
    }
  }
  return {
    schemaVersion: 1,
    source: 'doppler',
    root: rootDir,
    ok: results.every((entry) => entry.status === 'pass' || entry.status === 'skipped'),
    totals: {
      manifests: results.length,
      passed: results.filter((entry) => entry.status === 'pass').length,
      skipped: results.filter((entry) => entry.status === 'skipped').length,
      failed: results.filter((entry) => entry.status === 'fail').length,
      errors: results.filter((entry) => entry.status === 'error').length,
    },
    results,
  };
}

function printHuman(summary) {
  console.log(
    `[lean-execution-contract-sweep] manifests=${summary.totals.manifests} ` +
    `passed=${summary.totals.passed} skipped=${summary.totals.skipped} ` +
    `failed=${summary.totals.failed} errors=${summary.totals.errors}`
  );
  for (const result of summary.results) {
    const suffix = result.reason ? ` reason=${JSON.stringify(result.reason)}` : '';
    console.log(
      `[lean-execution-contract-sweep] ${result.status} ` +
      `model=${result.modelId ?? 'unknown'} path=${result.path}${suffix}`
    );
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }
  const rootDir = path.resolve(process.cwd(), args.root);
  const summary = await runSweep(rootDir, { check: args.check });
  if (args.json) {
    console.log(JSON.stringify(summary, null, 2));
  } else {
    printHuman(summary);
  }
  if (!summary.ok) {
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}

export {
  collectManifestPaths,
  isLeanExecutionContractManifest,
  runSweep,
};
