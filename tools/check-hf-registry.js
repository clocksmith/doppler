#!/usr/bin/env node

import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { isDeepStrictEqual } from 'node:util';
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
const DEFAULT_REMOTE_DRIFT_ALLOWLIST_FILE = path.join(
  REPO_ROOT,
  'tools',
  'policies',
  'hf-registry-remote-drift-allowlist.json'
);
const REGISTRY_ENTRY_VERIFY_CONCURRENCY = 4;
const ARTIFACT_SIDECAR_PROBE_CONCURRENCY = 4;

async function mapWithConcurrency(items, concurrency, mapper) {
  if (!Array.isArray(items) || items.length === 0) return [];
  const workerCount = Math.max(1, Math.min(concurrency, items.length));
  const results = new Array(items.length);
  let nextIndex = 0;

  const runners = Array.from({ length: workerCount }, async () => {
    while (nextIndex < items.length) {
      const index = nextIndex++;
      results[index] = await mapper(items[index], index);
    }
  });
  await Promise.all(runners);
  return results;
}

export function parseArgs(argv) {
  const out = {
    catalogFile: DEFAULT_CATALOG_FILE,
    remoteDriftAllowlistFile: DEFAULT_REMOTE_DRIFT_ALLOWLIST_FILE,
    registryUrl: DEFAULT_HF_REGISTRY_URL,
    checkLocalCatalog: true,
    checkDemoRegistry: true,
    probeArtifacts: false,
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
    if (arg === '--remote-drift-allowlist') {
      const value = normalizeText(argv[i + 1]);
      if (!value) throw new Error('Missing value for --remote-drift-allowlist');
      out.remoteDriftAllowlistFile = path.resolve(REPO_ROOT, value);
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
    if (arg === '--probe-artifacts') {
      out.probeArtifacts = true;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  return out;
}

function buildRemoteDriftAllowlist(payload) {
  if (payload == null) {
    return new Set();
  }
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('remote drift allowlist payload must be an object');
  }
  if (payload.schemaVersion !== 1) {
    throw new Error('remote drift allowlist schemaVersion must be 1');
  }
  if (!Array.isArray(payload.entries)) {
    throw new Error('remote drift allowlist entries must be an array');
  }
  const modelIds = new Set();
  for (const entry of payload.entries) {
    const modelId = normalizeText(entry?.modelId);
    if (!modelId) {
      throw new Error('remote drift allowlist entries require modelId');
    }
    if (!normalizeText(entry?.reason)) {
      throw new Error(`${modelId}: remote drift allowlist entries require reason`);
    }
    modelIds.add(modelId);
  }
  return modelIds;
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

function collectRequiredArtifactSidecars(manifest) {
  const shardFilenames = new Set(
    (Array.isArray(manifest?.shards) ? manifest.shards : [])
      .map((shard) => normalizeText(shard?.filename).replace(/^\/+/, ''))
      .filter(Boolean)
  );
  return collectRequiredArtifactFiles(manifest).filter((file) => !shardFilenames.has(file));
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

async function readJsonFile(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
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
  for (const [index, shard] of shards.entries()) {
    if (!normalizeText(shard?.filename)) {
      throw new Error(`${modelId}: shard ${index} is missing filename`);
    }
    if (!Number.isFinite(shard?.size) || shard.size <= 0) {
      throw new Error(`${modelId}: shard ${index} is missing positive size`);
    }
    if (!normalizeDigest(shard?.sha256 || shard?.hash || shard?.blake3)) {
      throw new Error(`${modelId}: shard ${index} is missing digest`);
    }
  }
  const probeResults = await mapWithConcurrency(
    collectRequiredArtifactSidecars(manifest),
    ARTIFACT_SIDECAR_PROBE_CONCURRENCY,
    async (file) => {
      const fileUrl = buildArtifactUrl(baseUrl, file);
      const probe = await probeUrl(fileUrl);
      return {
        fileUrl,
        ok: probe.ok,
        error: probe.error?.message || null,
      };
    }
  );
  const missingFiles = probeResults.filter((result) => !result.ok);
  if (missingFiles.length > 0) {
    const missingUrls = missingFiles
      .map((result) => result.error ? `${result.fileUrl} (${result.error})` : result.fileUrl)
      .join(', ');
    throw new Error(`${modelId}: artifact sidecar missing at ${missingUrls}`);
  }
  return shards.length;
}

function buildMetadataSummary(entry) {
  return {
    modelId: entry.modelId,
    manifestModelId: normalizeText(entry.modelId),
    shardCount: null,
  };
}

async function verifyRegistryEntries(entries, resolveBaseUrl, formatError) {
  const results = await mapWithConcurrency(
    entries,
    REGISTRY_ENTRY_VERIFY_CONCURRENCY,
    async (entry) => {
      const baseUrl = resolveBaseUrl(entry);
      try {
        const verified = await verifyManifestAndShards(entry.modelId, baseUrl, entry);
        return {
          error: null,
          summary: {
            modelId: entry.modelId,
            manifestModelId: verified.manifestModelId,
            shardCount: verified.shardCount,
          },
        };
      } catch (error) {
        return {
          error: formatError(entry, error),
          summary: null,
        };
      }
    }
  );
  return {
    errors: results.map((result) => result.error).filter(Boolean),
    summaries: results.map((result) => result.summary).filter(Boolean),
  };
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

export async function validateLocalHfCatalog(payload, options = {}) {
  const catalog = ensureCatalogPayload(payload, 'support registry');
  const approvedRegistry = buildHostedRegistryPayload(catalog);
  const errors = [];
  const probeArtifacts = options.probeArtifacts === true;
  const duplicates = collectDuplicateModelIds(catalog.models);
  if (duplicates.length > 0) {
    errors.push(`Duplicate support registry modelIds: ${duplicates.join(', ')}`);
  }

  const entriesToVerify = [];
  for (const entry of approvedRegistry.models) {
    const shapeErrors = validateLocalHfEntryShape(entry);
    if (shapeErrors.length > 0) {
      errors.push(...shapeErrors);
      continue;
    }
    entriesToVerify.push(entry);
  }
  if (!probeArtifacts) {
    return {
      errors,
      summaries: entriesToVerify.map(buildMetadataSummary),
    };
  }
  const verification = await verifyRegistryEntries(
    entriesToVerify,
    buildEntryRemoteBaseUrl,
    (_entry, error) => error.message
  );
  errors.push(...verification.errors);
  const summaries = verification.summaries;
  return { errors, summaries };
}

export async function validateRemoteRegistry(payload, registryUrl, localCatalog = null, options = {}) {
  const registry = ensureCatalogPayload(payload, 'remote registry');
  const errors = [];
  const remoteDriftAllowlist = options.remoteDriftAllowlist instanceof Set
    ? options.remoteDriftAllowlist
    : buildRemoteDriftAllowlist(options.remoteDriftAllowlist);
  const probeArtifacts = options.probeArtifacts === true;
  const allowedDrifts = [];
  const expectedRegistry = localCatalog ? buildHostedRegistryPayload(localCatalog) : null;
  const expectedModels = Array.isArray(expectedRegistry?.models) ? expectedRegistry.models : [];
  const expectedModelIds = new Set(expectedModels.map((entry) => normalizeText(entry?.modelId)));
  const duplicates = collectDuplicateModelIds(registry.models);
  if (duplicates.length > 0) {
    errors.push(`Duplicate remote registry modelIds: ${duplicates.join(', ')}`);
  }

  const entriesToVerify = registry.models.filter((entry) => {
    if (!shouldDemoSurfaceRemoteRegistryEntry(entry, registryUrl)) return false;
    const modelId = normalizeText(entry?.modelId);
    return !localCatalog || expectedModelIds.has(modelId) || !remoteDriftAllowlist.has(modelId);
  });
  let summaries = entriesToVerify.map(buildMetadataSummary);
  if (probeArtifacts) {
    const verification = await verifyRegistryEntries(
      entriesToVerify,
      (entry) => resolveDemoRegistryEntryBaseUrl(entry, registryUrl),
      (entry, error) => `${entry.modelId}: demo-visible registry entry is not fetchable (${error.message})`
    );
    errors.push(...verification.errors);
    summaries = verification.summaries;
  }

  if (localCatalog) {
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
      if (!isDeepStrictEqual(localEntry?.classification, remoteEntry?.classification)) {
        errors.push(`${localEntry.modelId}: local classification does not match remote registry`);
      }
    }
    for (const remoteEntry of registry.models) {
      const modelId = normalizeText(remoteEntry?.modelId);
      if (!modelId || expectedModelIds.has(modelId)) continue;
      if (remoteDriftAllowlist.has(modelId)) {
        allowedDrifts.push(modelId);
        continue;
      }
      errors.push(`${modelId}: remote registry contains a model that is not approved in the canonical support registry`);
    }
  }

  return { errors, summaries, allowedDrifts };
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const failures = [];
  const checks = [];
  let localCatalog = null;

  if (args.checkLocalCatalog) {
    localCatalog = await loadJsonFile(args.catalogFile, args.catalogFile);
    const localResult = await validateLocalHfCatalog(localCatalog, {
      probeArtifacts: args.probeArtifacts,
    });
    checks.push(`[hf-registry-check] catalog source: ${args.catalogFile}`);
    checks.push(
      `[hf-registry-check] approved HF entries ${args.probeArtifacts ? 'probed' : 'checked'}: ` +
      `${localResult.summaries.length}`
    );
    failures.push(...localResult.errors);
  }

  if (args.checkDemoRegistry) {
    const remoteDriftAllowlist = buildRemoteDriftAllowlist(await readJsonFile(
      args.remoteDriftAllowlistFile
    ).catch((error) => {
      if (error?.code === 'ENOENT') {
        return {
          schemaVersion: 1,
          entries: [],
        };
      }
      throw error;
    }));
    const remoteRegistry = await fetchJson(args.registryUrl);
    const remoteResult = await validateRemoteRegistry(remoteRegistry, args.registryUrl, localCatalog, {
      remoteDriftAllowlist,
      probeArtifacts: args.probeArtifacts,
    });
    checks.push(
      `[hf-registry-check] remote demo-visible entries ${args.probeArtifacts ? 'probed' : 'checked'}: ` +
      `${remoteResult.summaries.length}`
    );
    if (remoteResult.allowedDrifts.length > 0) {
      checks.push(`[hf-registry-check] allowed remote registry drifts: ${remoteResult.allowedDrifts.length}`);
    }
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
