#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  DEFAULT_EXTERNAL_SUPPORT_REGISTRY_PATH,
  ensureCatalogPayload,
} from './hf-registry-utils.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_EXTERNAL_REGISTRY = DEFAULT_EXTERNAL_SUPPORT_REGISTRY_PATH;
const DEFAULT_CATALOG_FILE = path.join(REPO_ROOT, 'models', 'catalog.json');

function parseArgs(argv) {
  const out = {
    check: false,
    externalRegistry: path.resolve(DEFAULT_EXTERNAL_REGISTRY),
    catalogFile: path.resolve(DEFAULT_CATALOG_FILE),
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const nextValue = () => {
      const value = String(argv[i + 1] || '').trim();
      if (!value) {
        throw new Error(`Missing value for ${arg}`);
      }
      i += 1;
      return value;
    };

    if (arg === '--check') {
      out.check = true;
      continue;
    }
    if (arg === '--external-registry') {
      out.externalRegistry = path.resolve(nextValue());
      continue;
    }
    if (arg === '--catalog-file') {
      out.catalogFile = path.resolve(nextValue());
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  return out;
}

export async function buildCatalogFromExternalRegistry(args) {
  const externalPayload = JSON.parse(await fs.readFile(args.externalRegistry, 'utf8'));
  ensureCatalogPayload(externalPayload, args.externalRegistry);

  const models = externalPayload.models.map((entry) => {
    const next = structuredClone(entry);
    delete next.external;
    return next;
  });

  return `${JSON.stringify({
    version: Number.isFinite(Number(externalPayload.version)) ? Number(externalPayload.version) : 1,
    lifecycleSchemaVersion: Number.isFinite(Number(externalPayload.lifecycleSchemaVersion))
      ? Number(externalPayload.lifecycleSchemaVersion)
      : 1,
    updatedAt: typeof externalPayload.updatedAt === 'string'
      ? externalPayload.updatedAt
      : new Date().toISOString().slice(0, 10),
    models,
  }, null, 2)}\n`;
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const nextCatalog = await buildCatalogFromExternalRegistry(args);

  if (args.check) {
    const currentCatalog = await fs.readFile(args.catalogFile, 'utf8');
    if (currentCatalog !== nextCatalog) {
      throw new Error(
        'Catalog is out of sync with external support registry. Run: node tools/sync-catalog-from-external-support.js'
      );
    }
    console.log(`[catalog-from-external] up to date (${args.catalogFile})`);
    return;
  }

  await fs.writeFile(args.catalogFile, nextCatalog, 'utf8');
  console.log(`[catalog-from-external] wrote ${args.catalogFile}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(`[catalog-from-external] ${error.message}`);
    process.exit(1);
  });
}
