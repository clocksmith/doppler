#!/usr/bin/env node

import { createHash } from 'node:crypto';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  DEFAULT_HF_REGISTRY_URL,
  buildHostedRegistryPayload,
  buildManifestUrl,
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
} from '../src/tooling/hf-registry-utils.js';

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

function verifyManifestIdentity(modelId, manifest, entry) {
  const identity = manifest?.artifactIdentity;
  const expected = {
    sourceCheckpointId: normalizeText(entry?.sourceCheckpointId),
    weightPackId: normalizeText(entry?.weightPackId),
    manifestVariantId: normalizeText(entry?.manifestVariantId),
  };
  if (!expected.sourceCheckpointId && !expected.weightPackId && !expected.manifestVariantId) {
    return;
  }
  if (!identity || typeof identity !== 'object') {
    throw new Error(`${modelId}: manifest is missing artifactIdentity for hosted catalog entry`);
  }
  for (const [field, expectedValue] of Object.entries(expected)) {
    if (!expectedValue) continue;
    const actualValue = normalizeText(identity?.[field]);
    if (actualValue !== expectedValue) {
      throw new Error(
        `${modelId}: manifest artifactIdentity.${field} "${actualValue}" does not match catalog "${expectedValue}"`
      );
    }
  }
}

function buildArtifactUrl(baseUrl, filename) {
  const normalizedBaseUrl = normalizeText(baseUrl).replace(/\/+$/, '');
  const normalizedFilename = normalizeText(filename).replace(/^\/+/, '');
  if (!normalizedBaseUrl || !normalizedFilename) {
    return '';
  }
  return `${normalizedBaseUrl}/${normalizedFilename.split('/').map(encodeURIComponent).join('/')}`;
}

function collectRequiredArtifactFiles(manifest) {
  const files = [];
  const add = (value) => {
    const normalized = normalizeText(value).replace(/^\/+/, '');
    if (normalized && !files.includes(normalized)) {
      files.push(normalized);
    }
  };
  if (typeof manifest?.tokenizer?.file === 'string') {
    add(manifest.tokenizer.file);
  }
  if (typeof manifest?.tokenizer?.sentencepieceModel === 'string') {
    add(manifest.tokenizer.sentencepieceModel);
  }
  if (typeof manifest?.tensorsFile === 'string') {
    add(manifest.tensorsFile);
  }
  for (const shard of Array.isArray(manifest?.shards) ? manifest.shards : []) {
    add(shard?.filename);
  }
  return files;
}

function normalizeDigest(value) {
  const normalized = normalizeText(value).toLowerCase();
  if (!normalized) return '';
  return normalized.startsWith('sha256:') ? normalized.slice('sha256:'.length) : normalized;
}

function sha256Text(value) {
  return createHash('sha256').update(String(value)).digest('hex');
}

function resolveWeightsRefBaseUrl(baseUrl, artifactRoot) {
  const root = normalizeText(artifactRoot);
  if (!root) {
    throw new Error('weightsRef.artifactRoot is required');
  }
  const base = normalizeText(baseUrl).replace(/\/+$/, '');
  if (!base) {
    throw new Error('weightsRef.artifactRoot requires a manifest base URL');
  }
  return new URL(root, base.endsWith('/') ? base : `${base}/`).toString().replace(/\/+$/, '');
}

async function fetchManifestPayload(manifestUrl) {
  const response = await fetch(manifestUrl, {
    headers: { Connection: 'close' },
    redirect: 'follow',
    signal: AbortSignal.timeout(30000),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${manifestUrl}`);
  }
  const text = await response.text();
  return {
    text,
    manifest: JSON.parse(text),
  };
}

function assertWeightsRefIdentity(modelId, variantManifest, weightsManifest, weightsRef, storageBaseUrl) {
  const expectedWeightPackId = normalizeText(weightsRef?.weightPackId);
  if (!expectedWeightPackId) {
    throw new Error(`${modelId}: weightsRef.weightPackId is required`);
  }
  const variantWeightPackId = normalizeText(variantManifest?.artifactIdentity?.weightPackId);
  if (variantWeightPackId && variantWeightPackId !== expectedWeightPackId) {
    throw new Error(
      `${modelId}: weightsRef.weightPackId "${expectedWeightPackId}" does not match ` +
      `manifest artifactIdentity.weightPackId "${variantWeightPackId}"`
    );
  }
  const targetWeightPackId = normalizeText(weightsManifest?.artifactIdentity?.weightPackId);
  if (targetWeightPackId !== expectedWeightPackId) {
    throw new Error(
      `${modelId}: weightsRef target ${storageBaseUrl} has artifactIdentity.weightPackId ` +
      `"${targetWeightPackId}", expected "${expectedWeightPackId}"`
    );
  }
  const expectedShardSetHash = normalizeText(weightsRef?.shardSetHash);
  if (expectedShardSetHash) {
    const actualShardSetHash = normalizeText(weightsManifest?.artifactIdentity?.shardSetHash);
    if (actualShardSetHash !== expectedShardSetHash) {
      throw new Error(
        `${modelId}: weightsRef.shardSetHash "${expectedShardSetHash}" does not match ` +
        `target artifactIdentity.shardSetHash "${actualShardSetHash}"`
      );
    }
  }
}

async function resolveArtifactManifest(modelId, baseUrl, manifest) {
  if (manifest?.weightsRef == null) {
    return {
      baseUrl,
      manifest,
      weightsRef: false,
    };
  }
  const weightsRef = manifest.weightsRef;
  const storageBaseUrl = resolveWeightsRefBaseUrl(baseUrl, weightsRef.artifactRoot);
  const storageManifestUrl = buildManifestUrl(storageBaseUrl);
  const expectedDigest = normalizeDigest(weightsRef.manifestDigest);
  if (!expectedDigest) {
    throw new Error(`${modelId}: weightsRef.manifestDigest is required`);
  }
  const storageManifestProbe = await probeUrl(storageManifestUrl);
  if (!storageManifestProbe.ok) {
    throw new Error(`${modelId}: weightsRef manifest missing at ${storageManifestUrl}`);
  }
  const storageManifestPayload = await fetchManifestPayload(storageManifestUrl);
  const actualDigest = sha256Text(storageManifestPayload.text);
  if (actualDigest !== expectedDigest) {
    throw new Error(
      `${modelId}: weightsRef.manifestDigest mismatch for ${storageBaseUrl}; ` +
      `expected ${expectedDigest}, got ${actualDigest}`
    );
  }
  assertWeightsRefIdentity(modelId, manifest, storageManifestPayload.manifest, weightsRef, storageBaseUrl);
  return {
    baseUrl: storageBaseUrl,
    manifest: storageManifestPayload.manifest,
    weightsRef: true,
  };
}

async function verifyArtifactFiles(modelId, baseUrl, manifest) {
  const shards = Array.isArray(manifest?.shards) ? manifest.shards : [];
  if (shards.length === 0) {
    throw new Error(`${modelId}: resolved artifact manifest does not declare shards`);
  }
  for (const file of collectRequiredArtifactFiles(manifest)) {
    const fileUrl = buildArtifactUrl(baseUrl, file);
    const fileProbe = await probeUrl(fileUrl);
    if (!fileProbe.ok) {
      throw new Error(`${modelId}: artifact file missing at ${fileUrl}`);
    }
  }
  return shards.length;
}

async function verifyManifestAndShards(modelId, baseUrl, entry = null) {
  const manifestUrl = buildManifestUrl(baseUrl);
  if (!manifestUrl) {
    throw new Error(`${modelId}: could not resolve manifest URL`);
  }
  const manifestProbe = await probeUrl(manifestUrl);
  if (!manifestProbe.ok) {
    throw new Error(`${modelId}: manifest missing at ${manifestUrl}`);
  }
  const manifestPayload = await fetchManifestPayload(manifestUrl);
  const manifest = manifestPayload.manifest;
  const manifestModelId = normalizeText(manifest?.modelId);
  if (!manifestModelId) {
    throw new Error(`${modelId}: manifest at ${manifestUrl} is missing modelId`);
  }
  if (manifestModelId !== normalizeText(modelId)) {
    throw new Error(
      `${modelId}: manifest modelId "${manifestModelId}" does not match the approved support entry modelId`
    );
  }
  if ((manifest?.weightsRef != null) !== (entry?.weightsRefAllowed === true)) {
    throw new Error(`${modelId}: manifest weightsRef presence does not match catalog weightsRefAllowed`);
  }
  verifyManifestIdentity(modelId, manifest, entry);
  const artifact = await resolveArtifactManifest(modelId, baseUrl, manifest);
  const shardCount = await verifyArtifactFiles(modelId, artifact.baseUrl, artifact.manifest);
  return {
    manifestUrl,
    manifestModelId,
    shardCount,
    weightsRef: artifact.weightsRef,
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
      const verified = await verifyManifestAndShards(entry.modelId, baseUrl, entry);
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
      const verified = await verifyManifestAndShards(entry.modelId, baseUrl, entry);
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
      for (const field of ['sourceCheckpointId', 'weightPackId', 'manifestVariantId']) {
        const localValue = normalizeText(localEntry?.[field]);
        const remoteValue = normalizeText(remoteEntry?.[field]);
        if (localValue !== remoteValue) {
          errors.push(`${localEntry.modelId}: local ${field} "${localValue}" does not match remote registry "${remoteValue}"`);
        }
      }
      for (const field of ['artifactCompleteness', 'runtimePromotionState', 'weightsRefAllowed']) {
        if (localEntry?.[field] !== remoteEntry?.[field]) {
          errors.push(`${localEntry.modelId}: local ${field} ${JSON.stringify(localEntry?.[field])} does not match remote registry ${JSON.stringify(remoteEntry?.[field])}`);
        }
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
