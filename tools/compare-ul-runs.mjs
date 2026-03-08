#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

function requireFlagValue(argv, index, flag) {
  const value = argv[index + 1];
  if (value == null || String(value).startsWith('--')) {
    throw new Error(`Missing value for ${flag}`);
  }
  return String(value);
}

export function parseArgs(argv) {
  const parsed = { left: null, right: null };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--left') {
      parsed.left = requireFlagValue(argv, i, '--left');
      i += 1;
      continue;
    }
    if (arg === '--right') {
      parsed.right = requireFlagValue(argv, i, '--right');
      i += 1;
      continue;
    }
    throw new Error(`Unknown flag: ${arg}`);
  }
  if (!parsed.left || !parsed.right) {
    throw new Error('Usage: node tools/compare-ul-runs.mjs --left <manifest.json> --right <manifest.json>');
  }
  return parsed;
}

async function readJson(pathValue) {
  const absolute = resolve(String(pathValue));
  const raw = await readFile(absolute, 'utf8');
  return { absolute, value: JSON.parse(raw) };
}

function compareScalar(left, right, key) {
  return {
    key,
    left: left?.[key] ?? null,
    right: right?.[key] ?? null,
    equal: (left?.[key] ?? null) === (right?.[key] ?? null),
  };
}

export function mainCompare(left, right) {
  return {
    schemaVersion: 1,
    stage: compareScalar(left, right, 'stage'),
    ulContractHash: compareScalar(left, right, 'ulContractHash'),
    manifestHash: compareScalar(left, right, 'manifestHash'),
    manifestContentHash: compareScalar(left, right, 'manifestContentHash'),
    configHash: compareScalar(left, right, 'configHash'),
    modelHash: compareScalar(left, right, 'modelHash'),
    datasetHash: compareScalar(left, right, 'datasetHash'),
    latentDatasetHash: {
      left: left?.latentDataset?.hash ?? null,
      right: right?.latentDataset?.hash ?? null,
      equal: (left?.latentDataset?.hash ?? null) === (right?.latentDataset?.hash ?? null),
    },
    latentDatasetCount: {
      left: left?.latentDataset?.count ?? null,
      right: right?.latentDataset?.count ?? null,
      delta: (left?.latentDataset?.count != null && right?.latentDataset?.count != null)
        ? left.latentDataset.count - right.latentDataset.count
        : null,
      incomplete: left?.latentDataset?.count == null || right?.latentDataset?.count == null,
    },
    metricCount: {
      left: left?.metrics?.count ?? null,
      right: right?.metrics?.count ?? null,
      delta: (left?.metrics?.count != null && right?.metrics?.count != null)
        ? left.metrics.count - right.metrics.count
        : null,
      incomplete: left?.metrics?.count == null || right?.metrics?.count == null,
    },
    stage1Dependency: {
      left: left?.stage1Dependency ?? null,
      right: right?.stage1Dependency ?? null,
    },
  };
}

async function main() {
  const args = parseArgs(process.argv);
  const left = await readJson(args.left);
  const right = await readJson(args.right);
  const comparison = mainCompare(left.value, right.value);
  process.stdout.write(`${JSON.stringify(comparison, null, 2)}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
