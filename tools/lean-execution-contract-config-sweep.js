#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { runLeanExecutionContractForManifest } from '../src/tooling/lean-execution-contract-runner.js';
import {
  inferConversionConfigModelId,
  resolveMaterializedManifestFromConversionConfig,
} from '../src/tooling/conversion-config-materializer.js';

function parseArgs(argv) {
  const args = {
    configRoot: 'tools/configs/conversion',
    manifestRoot: 'models',
    fixtureMap: 'tools/configs/conversion/lean-execution-contract-fixtures.json',
    json: false,
    check: true,
    requireManifestMatch: false,
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--config-root') {
      args.configRoot = argv[index + 1] ?? args.configRoot;
      index += 1;
      continue;
    }
    if (arg === '--manifest-root') {
      args.manifestRoot = argv[index + 1] ?? args.manifestRoot;
      index += 1;
      continue;
    }
    if (arg === '--fixture-map') {
      args.fixtureMap = argv[index + 1] ?? args.fixtureMap;
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
    if (arg === '--require-manifest-match') {
      args.requireManifestMatch = true;
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
    '  node tools/lean-execution-contract-config-sweep.js [--config-root <dir>] [--manifest-root <dir>] [--fixture-map <json>] [--json] [--no-check] [--require-manifest-match]',
  ].join('\n');
}

async function collectFiles(rootDir, predicate) {
  const output = [];
  async function walk(currentDir, isRoot) {
    let entries;
    try {
      entries = await fs.readdir(currentDir, { withFileTypes: true });
    } catch (error) {
      if (isRoot) {
        throw new Error(`Cannot read config root "${currentDir}": ${error.message}`);
      }
      return;
    }
    for (const entry of entries) {
      const absolute = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        await walk(absolute, false);
        continue;
      }
      if (entry.isFile() && predicate(entry.name)) {
        output.push(absolute);
      }
    }
  }
  await walk(rootDir, true);
  output.sort((left, right) => left.localeCompare(right));
  return output;
}

async function buildManifestIndex(rootDir) {
  const manifestPaths = await collectFiles(rootDir, (name) => name === 'manifest.json');
  const byModelId = new Map();
  for (const manifestPath of manifestPaths) {
    try {
      const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
      const modelId = typeof manifest?.modelId === 'string' ? manifest.modelId.trim() : '';
      if (!modelId) continue;
      byModelId.set(modelId, {
        manifestPath,
        manifest,
      });
    } catch {
      continue;
    }
  }
  return byModelId;
}

async function loadFixtureMap(filePath) {
  const resolvedPath = path.resolve(process.cwd(), filePath);
  const raw = await fs.readFile(resolvedPath, 'utf8');
  const payload = JSON.parse(raw);
  const mappings = Array.isArray(payload?.mappings) ? payload.mappings : [];
  const exclusions = Array.isArray(payload?.exclusions) ? payload.exclusions : [];
  const byConfigPath = new Map();
  const excludedByConfigPath = new Map();
  for (const entry of mappings) {
    const configPath = typeof entry?.configPath === 'string'
      ? path.normalize(entry.configPath)
      : '';
    const manifestPath = typeof entry?.manifestPath === 'string'
      ? path.normalize(entry.manifestPath)
      : '';
    if (!configPath || !manifestPath) continue;
    byConfigPath.set(configPath, manifestPath);
  }
  for (const entry of exclusions) {
    const configPath = typeof entry?.configPath === 'string'
      ? path.normalize(entry.configPath)
      : '';
    const reason = typeof entry?.reason === 'string'
      ? entry.reason.trim()
      : '';
    if (!configPath || !reason) continue;
    excludedByConfigPath.set(configPath, reason);
  }
  return { byConfigPath, excludedByConfigPath };
}

function isExecutionContractConfigCandidate(manifest) {
  return manifest
    && typeof manifest === 'object'
    && manifest.modelType !== 'diffusion'
    && manifest.modelType !== 'energy'
    && manifest.architecture
    && typeof manifest.architecture === 'object';
}

async function runSweep(options = {}) {
  const configRoot = path.resolve(process.cwd(), options.configRoot ?? 'tools/configs/conversion');
  const manifestRoot = path.resolve(process.cwd(), options.manifestRoot ?? 'models');
  const fixtureMapPath = options.fixtureMap ?? 'tools/configs/conversion/lean-execution-contract-fixtures.json';
  const resolvedFixtureMapPath = path.resolve(process.cwd(), fixtureMapPath);
  const configPaths = (await collectFiles(configRoot, (name) => name.endsWith('.json')))
    .filter((filePath) => path.resolve(filePath) !== resolvedFixtureMapPath);
  const manifestIndex = await buildManifestIndex(manifestRoot);
  const fixtureMap = await loadFixtureMap(fixtureMapPath);
  const results = [];

  for (const configPath of configPaths) {
    const relativeConfigPath = path.relative(process.cwd(), configPath);
    try {
      const excludedReason = fixtureMap.excludedByConfigPath.get(path.normalize(relativeConfigPath)) || null;
      if (excludedReason) {
        results.push({
          configPath: relativeConfigPath,
          modelId: inferConversionConfigModelId(configPath, JSON.parse(await fs.readFile(configPath, 'utf8'))),
          status: 'skipped',
          reason: excludedReason,
        });
        continue;
      }
      const rawConfig = JSON.parse(await fs.readFile(configPath, 'utf8'));
      const modelId = inferConversionConfigModelId(configPath, rawConfig);
      const mappedManifestPath = fixtureMap.byConfigPath.get(path.normalize(relativeConfigPath)) || null;
      let matched = manifestIndex.get(modelId) || null;
      if (!matched && mappedManifestPath) {
        const absoluteManifestPath = path.resolve(process.cwd(), mappedManifestPath);
        const manifest = JSON.parse(await fs.readFile(absoluteManifestPath, 'utf8'));
        matched = {
          manifestPath: absoluteManifestPath,
          manifest,
        };
      }
      if (!matched) {
        results.push({
          configPath: relativeConfigPath,
          modelId,
          status: options.requireManifestMatch ? 'error' : 'skipped',
          reason: 'no matching converted manifest fixture found',
        });
        continue;
      }
      if (!isExecutionContractConfigCandidate(matched.manifest)) {
        results.push({
          configPath: relativeConfigPath,
          modelId,
          manifestPath: path.relative(process.cwd(), matched.manifestPath),
          status: 'skipped',
          reason: 'matched manifest is outside execution-contract transformer scope',
        });
        continue;
      }
      const materializedManifest = resolveMaterializedManifestFromConversionConfig(rawConfig, matched.manifest);
      const leanResult = runLeanExecutionContractForManifest(materializedManifest, {
        rootDir: process.cwd(),
        check: options.check !== false,
      });
      results.push({
        configPath: relativeConfigPath,
        manifestPath: path.relative(process.cwd(), matched.manifestPath),
        modelId,
        status: leanResult.ok ? 'pass' : 'fail',
        toolchainRef: leanResult.toolchainRef,
      });
    } catch (error) {
      results.push({
        configPath: relativeConfigPath,
        modelId: null,
        status: 'error',
        reason: error instanceof Error ? error.message : String(error),
      });
    }
  }

  return {
    schemaVersion: 1,
    source: 'doppler',
    configRoot,
    manifestRoot,
    fixtureMapPath: resolvedFixtureMapPath,
    ok: results.every((entry) => entry.status === 'pass' || entry.status === 'skipped'),
    totals: {
      configs: results.length,
      passed: results.filter((entry) => entry.status === 'pass').length,
      skipped: results.filter((entry) => entry.status === 'skipped').length,
      explicitSkips: results.filter((entry) => entry.status === 'skipped' && entry.reason !== 'no matching converted manifest fixture found').length,
      failed: results.filter((entry) => entry.status === 'fail').length,
      errors: results.filter((entry) => entry.status === 'error').length,
    },
    results,
  };
}

function printHuman(summary) {
  console.log(
    `[lean-execution-contract-config-sweep] configs=${summary.totals.configs} ` +
    `passed=${summary.totals.passed} skipped=${summary.totals.skipped} ` +
    `failed=${summary.totals.failed} errors=${summary.totals.errors}`
  );
  for (const result of summary.results) {
    const suffix = result.reason ? ` reason=${JSON.stringify(result.reason)}` : '';
    console.log(
      `[lean-execution-contract-config-sweep] ${result.status} ` +
      `model=${result.modelId ?? 'unknown'} config=${result.configPath}` +
      `${result.manifestPath ? ` manifest=${result.manifestPath}` : ''}${suffix}`
    );
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }
  const summary = await runSweep(args);
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
  buildManifestIndex,
  runSweep,
};
