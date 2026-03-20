#!/usr/bin/env node

import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  DEFAULT_HF_REGISTRY_URL,
  buildHostedRegistryPayload,
  buildManifestUrl,
  buildShardUrl,
  buildEntryRemoteBaseUrl,
  collectDuplicateModelIds,
  ensureCatalogPayload,
  fetchJson,
  findCatalogEntry,
  getEntryHfSpec,
  loadJsonFile,
  normalizeText,
  probeUrl,
  resolveDemoRegistryEntryBaseUrl,
  shouldDemoSurfaceRemoteRegistryEntry,
  validateLocalHfEntryShape,
} from './hf-registry-utils.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_CATALOG_FILE = path.join(REPO_ROOT, 'models', 'catalog.json');

export function parseArgs(argv) {
  const out = {
    catalogFile: DEFAULT_CATALOG_FILE,
    registryUrl: DEFAULT_HF_REGISTRY_URL,
    checkLocalCatalog: true,
    checkDemoRegistry: true,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--catalog-file') {
      const value = normalizeText(argv[i + 1]);
      if (!value) throw new Error('Missing value for --catalog-file');
      out.catalogFile = path.resolve(REPO_ROOT, value);
      i += 1;
      continue;
    }
    if (arg === '--registry-url') {
      const value = normalizeText(argv[i + 1]);
      if (!value) throw new Error('Missing value for --registry-url');
      out.registryUrl = value;
      i += 1;
      continue;
    }
    if (arg === '--local-only') {
      out.checkDemoRegistry = false;
      continue;
    }
    if (arg === '--remote-only') {
      out.checkLocalCatalog = false;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  return out;
}

async function verifyManifestAndShards(modelId, baseUrl) {
  const manifestUrl = buildManifestUrl(baseUrl);
  if (!manifestUrl) {
    throw new Error(`${modelId}: could not resolve manifest URL`);
  }
  const manifestProbe = await probeUrl(manifestUrl);
  if (!manifestProbe.ok) {
    throw new Error(`${modelId}: manifest missing at ${manifestUrl}`);
  }
  const manifest = await fetchJson(manifestUrl);
  const manifestModelId = normalizeText(manifest?.modelId);
  if (!manifestModelId) {
    throw new Error(`${modelId}: manifest at ${manifestUrl} is missing modelId`);
  }
  if (manifestModelId !== normalizeText(modelId)) {
    throw new Error(
      `${modelId}: manifest modelId "${manifestModelId}" does not match the approved support entry modelId`
    );
  }
  const shards = Array.isArray(manifest?.shards) ? manifest.shards : [];
  if (shards.length === 0) {
    throw new Error(`${modelId}: manifest at ${manifestUrl} does not declare shards`);
  }
  for (const shard of shards) {
    const shardUrl = buildShardUrl(baseUrl, shard);
    const shardProbe = await probeUrl(shardUrl);
    if (!shardProbe.ok) {
      throw new Error(`${modelId}: shard missing at ${shardUrl}`);
    }
  }
  return {
    manifestUrl,
    manifestModelId,
    shardCount: shards.length,
  };
}

export async function validateLocalHfCatalog(payload) {
  const catalog = ensureCatalogPayload(payload, 'support registry');
  const approvedRegistry = buildHostedRegistryPayload(catalog);
  const errors = [];
  const duplicates = collectDuplicateModelIds(catalog.models);
  if (duplicates.length > 0) {
    errors.push(`Duplicate support registry modelIds: ${duplicates.join(', ')}`);
  }

  const summaries = [];
  for (const entry of approvedRegistry.models) {
    const shapeErrors = validateLocalHfEntryShape(entry);
    if (shapeErrors.length > 0) {
      errors.push(...shapeErrors);
      continue;
    }
    const baseUrl = buildEntryRemoteBaseUrl(entry);
    try {
      const verified = await verifyManifestAndShards(entry.modelId, baseUrl);
      summaries.push({
        modelId: entry.modelId,
        manifestModelId: verified.manifestModelId,
        shardCount: verified.shardCount,
      });
    } catch (error) {
      errors.push(error.message);
    }
  }
  return { errors, summaries };
}

export async function validateRemoteRegistry(payload, registryUrl, localCatalog = null) {
  const registry = ensureCatalogPayload(payload, 'remote registry');
  const errors = [];
  const duplicates = collectDuplicateModelIds(registry.models);
  if (duplicates.length > 0) {
    errors.push(`Duplicate remote registry modelIds: ${duplicates.join(', ')}`);
  }

  const summaries = [];
  for (const entry of registry.models) {
    if (!shouldDemoSurfaceRemoteRegistryEntry(entry, registryUrl)) continue;
    const baseUrl = resolveDemoRegistryEntryBaseUrl(entry, registryUrl);
    try {
      const verified = await verifyManifestAndShards(entry.modelId, baseUrl);
      summaries.push({
        modelId: entry.modelId,
        manifestModelId: verified.manifestModelId,
        shardCount: verified.shardCount,
      });
    } catch (error) {
      errors.push(`${entry.modelId}: demo-visible registry entry is not fetchable (${error.message})`);
    }
  }

  if (localCatalog) {
    const expectedRegistry = buildHostedRegistryPayload(localCatalog);
    const expectedModels = Array.isArray(expectedRegistry.models) ? expectedRegistry.models : [];
    const expectedModelIds = new Set(expectedModels.map((entry) => normalizeText(entry?.modelId)));
    for (const localEntry of expectedModels) {
      const remoteEntry = findCatalogEntry(registry, localEntry.modelId);
      if (!remoteEntry) {
        errors.push(`${localEntry.modelId}: approved hosted entry is missing from remote registry`);
        continue;
      }
      const localHf = getEntryHfSpec(localEntry);
      const remoteHf = getEntryHfSpec(remoteEntry);
      if (localHf.repoId !== remoteHf.repoId || localHf.revision !== remoteHf.revision || localHf.path !== remoteHf.path) {
        errors.push(
          `${localEntry.modelId}: local HF spec (${localHf.repoId}@${localHf.revision}:${localHf.path}) does not match remote registry ` +
          `(${remoteHf.repoId}@${remoteHf.revision}:${remoteHf.path})`
        );
      }
    }
    for (const remoteEntry of registry.models) {
      const modelId = normalizeText(remoteEntry?.modelId);
      if (!modelId || expectedModelIds.has(modelId)) continue;
      errors.push(`${modelId}: remote registry contains a model that is not approved in the canonical support registry`);
    }
  }

  return { errors, summaries };
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const failures = [];
  const checks = [];
  let localCatalog = null;

  if (args.checkLocalCatalog) {
    localCatalog = await loadJsonFile(args.catalogFile, args.catalogFile);
    const localResult = await validateLocalHfCatalog(localCatalog);
    checks.push(`[hf-registry-check] catalog source: ${args.catalogFile}`);
    checks.push(`[hf-registry-check] approved HF entries verified: ${localResult.summaries.length}`);
    failures.push(...localResult.errors);
  }

  if (args.checkDemoRegistry) {
    const remoteRegistry = await fetchJson(args.registryUrl);
    const remoteResult = await validateRemoteRegistry(remoteRegistry, args.registryUrl, localCatalog);
    checks.push(`[hf-registry-check] remote demo-visible entries verified: ${remoteResult.summaries.length}`);
    failures.push(...remoteResult.errors);
  }

  if (failures.length > 0) {
    throw new Error(failures.join('\n'));
  }

  for (const line of checks) {
    console.log(line);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main()
    .then(() => {
      process.exit(0);
    })
    .catch((error) => {
      console.error(`[hf-registry-check] ${error.message}`);
      process.exit(1);
    });
}
