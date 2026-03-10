#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_CATALOG_FILE = path.join(REPO_ROOT, 'models', 'catalog.json');
const DEFAULT_OUTPUT_FILE = path.join(REPO_ROOT, 'src', 'client', 'doppler-registry.json');

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeStringList(values) {
  return Array.isArray(values)
    ? values
      .map((entry) => normalizeText(entry))
      .filter((entry) => entry.length > 0)
    : [];
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

export function parseArgs(argv) {
  const args = {
    check: false,
    catalogFile: DEFAULT_CATALOG_FILE,
    outputFile: DEFAULT_OUTPUT_FILE,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    const nextValue = () => {
      const candidate = argv[i + 1];
      if (candidate == null || String(candidate).startsWith('--')) {
        throw new Error(`Missing value for ${token}`);
      }
      i += 1;
      return path.resolve(REPO_ROOT, String(candidate).trim());
    };
    if (token === '--check') {
      args.check = true;
      continue;
    }
    if (token === '--catalog-file') {
      args.catalogFile = nextValue();
      continue;
    }
    if (token === '--output-file') {
      args.outputFile = nextValue();
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }

  return args;
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

function compareCatalogEntries(left, right) {
  const leftOrder = Number.isFinite(left?.sortOrder) ? left.sortOrder : Number.MAX_SAFE_INTEGER;
  const rightOrder = Number.isFinite(right?.sortOrder) ? right.sortOrder : Number.MAX_SAFE_INTEGER;
  if (leftOrder !== rightOrder) {
    return leftOrder - rightOrder;
  }
  return normalizeText(left?.modelId).localeCompare(normalizeText(right?.modelId));
}

function toQuickstartEntry(entry) {
  const modelId = normalizeText(entry?.modelId);
  if (!modelId) {
    throw new Error('quickstart catalog entry must include modelId');
  }
  const hf = isPlainObject(entry?.hf) ? entry.hf : null;
  const repoId = normalizeText(hf?.repoId);
  const revision = normalizeText(hf?.revision);
  const repoPath = normalizeText(hf?.path).replace(/^\/+/, '');
  if (!repoId || !revision || !repoPath) {
    throw new Error(
      `${modelId}: quickstart catalog entries require complete hf.repoId, hf.revision, and hf.path metadata`
    );
  }

  return {
    modelId,
    aliases: normalizeStringList(entry?.aliases),
    modes: normalizeStringList(entry?.modes),
    hf: {
      repoId,
      revision,
      path: repoPath,
    },
  };
}

export function buildQuickstartRegistryPayload(catalog) {
  if (!isPlainObject(catalog) || !Array.isArray(catalog.models)) {
    throw new Error('catalog payload must be an object with a models array');
  }
  const models = catalog.models
    .filter((entry) => entry?.quickstart === true)
    .sort(compareCatalogEntries)
    .map(toQuickstartEntry);

  return {
    version: 1,
    source: 'models/catalog.json',
    models,
  };
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const catalog = await readJson(args.catalogFile);
  const nextPayload = buildQuickstartRegistryPayload(catalog);
  const nextText = `${JSON.stringify(nextPayload, null, 2)}\n`;

  if (args.check) {
    const currentText = await fs.readFile(args.outputFile, 'utf8');
    if (currentText !== nextText) {
      throw new Error(
        `Quickstart registry is out of date at ${path.relative(REPO_ROOT, args.outputFile)}. ` +
        'Run npm run quickstart:sync'
      );
    }
    console.log(`[quickstart-registry] up to date (${nextPayload.models.length} models)`);
    return;
  }

  await fs.mkdir(path.dirname(args.outputFile), { recursive: true });
  await fs.writeFile(args.outputFile, nextText, 'utf8');
  console.log(`[quickstart-registry] wrote ${path.relative(REPO_ROOT, args.outputFile)} (${nextPayload.models.length} models)`);
}

if (process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href) {
  main().catch((error) => {
    console.error(`[quickstart-registry] ${error.message}`);
    process.exit(1);
  });
}
