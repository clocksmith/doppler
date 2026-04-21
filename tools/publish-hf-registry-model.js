#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { mkdtemp, rm, readdir, symlink, copyFile, readFile, stat, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawn } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  buildHostedRegistryPayload,
  DEFAULT_HF_REGISTRY_PATH,
  DEFAULT_HF_REGISTRY_URL,
  DEFAULT_EXTERNAL_MODELS_ROOT,
  buildHfResolveUrl,
  buildManifestUrl,
  extractCommitShaFromUrl,
  fetchRepoHeadSha,
  fetchJson,
  findCatalogEntry,
  getEntryHfSpec,
  isHostedRegistryApprovedEntry,
  loadJsonFile,
  normalizeText,
  probeUrl,
  writeJsonFile,
} from '../src/tooling/hf-registry-utils.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_CATALOG_FILE = path.join(REPO_ROOT, 'models', 'catalog.json');

export function parseArgs(argv) {
  const out = {
    modelId: '',
    allApproved: false,
    catalogFile: DEFAULT_CATALOG_FILE,
    localDir: '',
    shardDir: '',
    localRoot: '',
    shardRoot: '',
    registryUrl: DEFAULT_HF_REGISTRY_URL,
    registryPath: DEFAULT_HF_REGISTRY_PATH,
    repoId: '',
    dryRun: false,
    manifestOnly: false,
    bootstrap: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--model-id') {
      out.modelId = normalizeText(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--all-approved') {
      out.allApproved = true;
      continue;
    }
    if (arg === '--catalog-file') {
      const value = normalizeText(argv[i + 1]);
      if (!value) throw new Error('Missing value for --catalog-file');
      out.catalogFile = path.resolve(REPO_ROOT, value);
      i += 1;
      continue;
    }
    if (arg === '--local-dir') {
      const value = normalizeText(argv[i + 1]);
      if (!value) throw new Error('Missing value for --local-dir');
      out.localDir = path.resolve(REPO_ROOT, value);
      i += 1;
      continue;
    }
    if (arg === '--shard-dir') {
      const value = normalizeText(argv[i + 1]);
      if (!value) throw new Error('Missing value for --shard-dir');
      out.shardDir = path.resolve(REPO_ROOT, value);
      i += 1;
      continue;
    }
    if (arg === '--local-root') {
      const value = normalizeText(argv[i + 1]);
      if (!value) throw new Error('Missing value for --local-root');
      out.localRoot = path.resolve(REPO_ROOT, value);
      i += 1;
      continue;
    }
    if (arg === '--shard-root') {
      const value = normalizeText(argv[i + 1]);
      if (!value) throw new Error('Missing value for --shard-root');
      out.shardRoot = path.resolve(REPO_ROOT, value);
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
    if (arg === '--registry-path') {
      const value = normalizeText(argv[i + 1]);
      if (!value) throw new Error('Missing value for --registry-path');
      out.registryPath = value;
      i += 1;
      continue;
    }
    if (arg === '--repo-id') {
      const value = normalizeText(argv[i + 1]);
      if (!value) throw new Error('Missing value for --repo-id');
      out.repoId = value;
      i += 1;
      continue;
    }
    if (arg === '--dry-run') {
      out.dryRun = true;
      continue;
    }
    if (arg === '--manifest-only') {
      out.manifestOnly = true;
      continue;
    }
    if (arg === '--bootstrap') {
      out.bootstrap = true;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  if (out.allApproved && out.modelId) {
    throw new Error('--all-approved cannot be combined with --model-id');
  }
  if (out.allApproved && out.manifestOnly) {
    throw new Error('--all-approved cannot be combined with --manifest-only');
  }
  if (out.allApproved && out.bootstrap) {
    throw new Error('--all-approved cannot be combined with --bootstrap');
  }
  if (out.allApproved && out.localDir) {
    throw new Error('--all-approved uses --local-root, not --local-dir');
  }
  if (out.allApproved && out.shardDir) {
    throw new Error('--all-approved uses --shard-root, not --shard-dir');
  }
  if (!out.modelId && !out.allApproved) {
    throw new Error('Missing required --model-id');
  }
  return out;
}

function spawnCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd || process.cwd(),
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => {
      const text = String(chunk);
      stdout += text;
      process.stdout.write(text);
    });
    child.stderr.on('data', (chunk) => {
      const text = String(chunk);
      stderr += text;
      process.stderr.write(text);
    });
    child.on('error', reject);
    child.on('exit', (code) => {
      if ((code ?? 1) !== 0) {
        reject(new Error(`${command} ${args.join(' ')} failed with code ${code ?? 1}`));
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

export function buildArtifactUploadPlan(entry, options = {}) {
  const modelId = normalizeText(entry?.modelId);
  if (!modelId) throw new Error('Catalog entry is missing modelId.');
  const hfSpec = getEntryHfSpec(entry);
  const repoId = normalizeText(options.repoId) || hfSpec.repoId;
  const targetPath = hfSpec.path;

  // Manifest source of truth: models/local/<modelId>/
  const manifestDir = options.localDir
    || (options.localRoot ? path.join(options.localRoot, modelId) : '')
    || path.join(REPO_ROOT, 'models', 'local', modelId);

  // Shard source: external drive (heavy files not in git)
  const shardDir = options.shardDir
    || (options.shardRoot ? path.join(options.shardRoot, modelId) : '')
    || path.join(DEFAULT_EXTERNAL_MODELS_ROOT, 'rdrr', modelId);

  if (!repoId) {
    throw new Error(`${modelId}: hf.repoId is required to publish`);
  }
  if (!targetPath) {
    throw new Error(`${modelId}: hf.path is required to publish`);
  }
  return {
    modelId,
    sourceCheckpointId: normalizeText(entry?.sourceCheckpointId),
    weightPackId: normalizeText(entry?.weightPackId),
    manifestVariantId: normalizeText(entry?.manifestVariantId),
    weightsRefAllowed: entry?.weightsRefAllowed === true,
    repoId,
    sourceRevision: hfSpec.revision,
    targetPath,
    manifestDir,
    shardDir,
  };
}

function buildDryRunPayload(uploadPlan, args, manifest) {
  return {
    modelId: uploadPlan.modelId,
    catalogFile: args.catalogFile,
    repoId: uploadPlan.repoId,
    manifestDir: uploadPlan.manifestDir,
    shardDir: args.manifestOnly ? null : uploadPlan.shardDir,
    targetPath: uploadPlan.targetPath,
    registryPath: args.registryPath,
    registryUrl: args.registryUrl,
    manifestOnly: args.manifestOnly,
    bootstrap: args.bootstrap,
    validated: true,
    artifactIdentity: manifest?.artifactIdentity ?? null,
    requiredArtifactFiles: collectRequiredManifestFiles(manifest).size,
  };
}

function normalizeArtifactPath(value, label) {
  const normalized = normalizeText(value);
  if (!normalized) return '';
  if (path.isAbsolute(normalized) || normalized.split(/[\\/]+/).includes('..')) {
    throw new Error(`${label} must be an artifact-relative path: ${normalized}`);
  }
  return normalized;
}

function collectRequiredManifestFiles(manifest) {
  const requiredFiles = new Map();
  const add = (filename, expectedSize = null, label = 'manifest file') => {
    const normalized = normalizeArtifactPath(filename, label);
    if (!normalized) return;
    requiredFiles.set(normalized, Number.isFinite(expectedSize) ? Number(expectedSize) : null);
  };

  if (typeof manifest?.tokenizer?.file === 'string') {
    add(manifest.tokenizer.file, null, 'tokenizer.file');
  }
  if (typeof manifest?.tokenizer?.sentencepieceModel === 'string') {
    add(manifest.tokenizer.sentencepieceModel, null, 'tokenizer.sentencepieceModel');
  }
  if (typeof manifest?.tensorsFile === 'string') {
    add(manifest.tensorsFile, null, 'tensorsFile');
  }
  if (Array.isArray(manifest?.shards)) {
    for (const shard of manifest.shards) {
      add(shard?.filename, Number(shard?.size), 'shards[].filename');
    }
  }
  return requiredFiles;
}

function buildArtifactFileUrl(baseUrl, filename) {
  const normalizedBaseUrl = normalizeText(baseUrl).replace(/\/+$/, '');
  const normalizedFilename = normalizeArtifactPath(filename, 'published artifact file');
  if (!normalizedBaseUrl || !normalizedFilename) {
    throw new Error('Published artifact file URL requires baseUrl and filename.');
  }
  return `${normalizedBaseUrl}/${normalizedFilename.split('/').map(encodeURIComponent).join('/')}`;
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
    throw new Error('weightsRef.artifactRoot is required.');
  }
  const base = normalizeText(baseUrl).replace(/\/+$/, '');
  if (!base) {
    throw new Error('weightsRef.artifactRoot requires a manifest base URL.');
  }
  return new URL(root, base.endsWith('/') ? base : `${base}/`).toString().replace(/\/+$/, '');
}

async function fetchManifestText(manifestUrl) {
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
    throw new Error(`${modelId}: weightsRef.weightPackId is required.`);
  }
  const variantWeightPackId = normalizeText(variantManifest?.artifactIdentity?.weightPackId);
  if (variantWeightPackId && variantWeightPackId !== expectedWeightPackId) {
    throw new Error(
      `${modelId}: weightsRef.weightPackId "${expectedWeightPackId}" does not match ` +
      `manifest artifactIdentity.weightPackId "${variantWeightPackId}".`
    );
  }
  const targetWeightPackId = normalizeText(weightsManifest?.artifactIdentity?.weightPackId);
  if (targetWeightPackId !== expectedWeightPackId) {
    throw new Error(
      `${modelId}: weightsRef target ${storageBaseUrl} has artifactIdentity.weightPackId ` +
      `"${targetWeightPackId}", expected "${expectedWeightPackId}".`
    );
  }
  const expectedShardSetHash = normalizeText(weightsRef?.shardSetHash);
  if (expectedShardSetHash) {
    const actualShardSetHash = normalizeText(weightsManifest?.artifactIdentity?.shardSetHash);
    if (actualShardSetHash !== expectedShardSetHash) {
      throw new Error(
        `${modelId}: weightsRef.shardSetHash "${expectedShardSetHash}" does not match ` +
        `target artifactIdentity.shardSetHash "${actualShardSetHash}".`
      );
    }
  }
}

async function resolveWeightsRefArtifactManifest(baseUrl, manifest) {
  const modelId = normalizeText(manifest?.modelId) || 'unknown-model';
  const weightsRef = manifest?.weightsRef;
  if (weightsRef == null) {
    return {
      baseUrl,
      manifest,
      weightsRef: false,
    };
  }
  const storageBaseUrl = resolveWeightsRefBaseUrl(baseUrl, weightsRef.artifactRoot);
  const storageManifestUrl = buildManifestUrl(storageBaseUrl);
  const expectedDigest = normalizeDigest(weightsRef.manifestDigest);
  if (!expectedDigest) {
    throw new Error(`${modelId}: weightsRef.manifestDigest is required.`);
  }
  const manifestProbe = await probeUrl(storageManifestUrl);
  if (!manifestProbe.ok) {
    throw new Error(`${modelId}: weightsRef manifest missing at ${storageManifestUrl}`);
  }
  const storageManifestPayload = await fetchManifestText(storageManifestUrl);
  const actualDigest = sha256Text(storageManifestPayload.text);
  if (actualDigest !== expectedDigest) {
    throw new Error(
      `${modelId}: weightsRef.manifestDigest mismatch for ${storageBaseUrl}. ` +
      `Expected ${expectedDigest}, got ${actualDigest}.`
    );
  }
  assertWeightsRefIdentity(modelId, manifest, storageManifestPayload.manifest, weightsRef, storageBaseUrl);
  return {
    baseUrl: storageBaseUrl,
    manifest: storageManifestPayload.manifest,
    weightsRef: true,
  };
}

async function assertCompleteUploadArtifact(uploadPlan, options = {}) {
  const { modelId, manifestDir, shardDir } = uploadPlan;
  const manifestOnly = options.manifestOnly === true;
  const manifestPath = path.join(manifestDir, 'manifest.json');
  if (!existsSync(manifestPath)) {
    throw new Error(
      `${modelId}: manifest.json not found in ${manifestDir}. `
      + 'The models/local/<modelId>/ directory is the source of truth for manifests.'
    );
  }

  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const identity = manifest?.artifactIdentity;
  if (!identity || typeof identity !== 'object' || Array.isArray(identity)) {
    throw new Error(`${modelId}: manifest artifactIdentity is required before publication.`);
  }
  for (const field of ['sourceCheckpointId', 'weightPackId', 'manifestVariantId']) {
    const expected = normalizeText(uploadPlan?.[field]);
    const actual = normalizeText(identity?.[field]);
    if (!expected || actual !== expected) {
      throw new Error(
        `${modelId}: manifest artifactIdentity.${field} "${actual}" does not match catalog "${expected}".`
      );
    }
  }
  if (manifest?.weightsRef != null) {
    if (!manifestOnly) {
      throw new Error(
        `${modelId}: manifest declares weightsRef; publish it with --manifest-only after the referenced weight pack is hosted.`
      );
    }
    if (uploadPlan.weightsRefAllowed !== true) {
      throw new Error(`${modelId}: catalog weightsRefAllowed must be true for --manifest-only weightsRef publication.`);
    }
    return manifest;
  }
  if (manifestOnly) {
    throw new Error(
      `${modelId}: --manifest-only requires manifest.weightsRef. Publish a complete artifact instead.`
    );
  }
  if (!existsSync(shardDir)) {
    throw new Error(
      `${modelId}: shard directory not found at ${shardDir}. `
      + 'The external drive must be mounted with model shards present.'
    );
  }
  const requiredFiles = collectRequiredManifestFiles(manifest);
  if (requiredFiles.size === 0) {
    throw new Error(`${modelId}: manifest.json does not declare tokenizer, tensor, or shard files to publish.`);
  }

  const missing = [];
  const mismatched = [];
  for (const [file, expectedSize] of requiredFiles.entries()) {
    const filePath = path.join(shardDir, file);
    if (!existsSync(filePath)) {
      missing.push(file);
      continue;
    }
    if (expectedSize != null) {
      const fileStat = await stat(filePath);
      if (fileStat.size !== expectedSize) {
        mismatched.push(`${file} expected ${expectedSize} bytes, found ${fileStat.size}`);
      }
    }
  }
  if (missing.length > 0) {
    throw new Error(
      `${modelId}: publish candidate is incomplete; missing required artifact files in ${shardDir}: `
      + missing.join(', ')
    );
  }
  if (mismatched.length > 0) {
    throw new Error(
      `${modelId}: publish candidate has shard size mismatches: ${mismatched.join('; ')}`
    );
  }
  return manifest;
}

async function verifyPublishedArtifactFiles(baseUrl, manifest) {
  const modelId = normalizeText(manifest?.modelId) || 'unknown-model';
  const artifact = await resolveWeightsRefArtifactManifest(baseUrl, manifest);
  const requiredFiles = collectRequiredManifestFiles(artifact.manifest);
  const shards = Array.isArray(artifact.manifest?.shards) ? artifact.manifest.shards : [];
  if (shards.length === 0) {
    throw new Error(`${modelId}: published artifact manifest does not declare shards.`);
  }
  const missing = [];
  for (const file of requiredFiles.keys()) {
    const url = buildArtifactFileUrl(artifact.baseUrl, file);
    const probe = await probeUrl(url);
    if (!probe.ok) {
      missing.push(`${file} (${url})`);
    }
  }
  if (missing.length > 0) {
    throw new Error(`${modelId}: published artifact is missing required files: ${missing.join(', ')}`);
  }
}

/**
 * Build a staging directory for HF upload by combining:
 * - manifest.json + origin.json from manifestDir (source of truth)
 * - shard_*.bin + tokenizer files from shardDir (external drive)
 */
export async function buildStagingDir(uploadPlan, options = {}) {
  const { modelId, manifestDir, shardDir } = uploadPlan;
  const manifestPath = path.join(manifestDir, 'manifest.json');
  const manifest = options.manifest ?? await assertCompleteUploadArtifact(uploadPlan, options);

  const stagingDir = await mkdtemp(path.join(os.tmpdir(), `doppler-publish-${modelId}-`));

  // Copy manifest + origin from local (source of truth)
  await copyFile(manifestPath, path.join(stagingDir, 'manifest.json'));
  const originPath = path.join(manifestDir, 'origin.json');
  if (existsSync(originPath)) {
    await copyFile(originPath, path.join(stagingDir, 'origin.json'));
  }
  if (options.manifestOnly === true || manifest?.weightsRef != null) {
    return stagingDir;
  }

  // Symlink shards + tokenizer from external drive (heavy files)
  const shardFiles = await readdir(shardDir);
  for (const file of shardFiles) {
    if (file === 'manifest.json' || file === 'origin.json') continue;
    const src = path.join(shardDir, file);
    const dst = path.join(stagingDir, file);
    await symlink(src, dst);
  }

  return stagingDir;
}

export function assertPromotionReady(entry, options = {}) {
  const { bootstrap = false, manifestOnly = false } = options;
  const modelId = normalizeText(entry?.modelId) || 'unknown-model';
  const lifecycle = entry?.lifecycle && typeof entry.lifecycle === 'object'
    ? entry.lifecycle
    : {};
  const status = lifecycle.status && typeof lifecycle.status === 'object'
    ? lifecycle.status
    : {};
  const tested = lifecycle.tested && typeof lifecycle.tested === 'object'
    ? lifecycle.tested
    : {};
  const contracts = tested.contracts && typeof tested.contracts === 'object'
    ? tested.contracts
    : {};
  const availabilityHf = lifecycle?.availability?.hf;
  const requiredIdentityFields = [
    'sourceCheckpointId',
    'weightPackId',
    'manifestVariantId',
  ];
  for (const field of requiredIdentityFields) {
    if (!normalizeText(entry?.[field])) {
      throw new Error(`${modelId}: ${field} is required before publication.`);
    }
  }
  if (entry?.artifactCompleteness !== 'complete') {
    throw new Error(`${modelId}: artifactCompleteness must be "complete" before publication.`);
  }
  if (entry?.runtimePromotionState !== 'manifest-owned') {
    throw new Error(`${modelId}: runtimePromotionState must be "manifest-owned" before publication.`);
  }
  if (manifestOnly) {
    if (entry?.weightsRefAllowed !== true) {
      throw new Error(`${modelId}: weightsRefAllowed must be true for --manifest-only weightsRef publication.`);
    }
  } else if (entry?.weightsRefAllowed !== false) {
    throw new Error(`${modelId}: weightsRefAllowed must be false for complete artifact publication.`);
  }
  if (bootstrap) {
    if (availabilityHf !== false) {
      throw new Error(`${modelId}: --bootstrap requires lifecycle.availability.hf=false (first publish only); got ${JSON.stringify(availabilityHf)}.`);
    }
  } else if (availabilityHf !== true) {
    throw new Error(`${modelId}: lifecycle.availability.hf must be true before publication (use --bootstrap for first publish).`);
  }
  if (status.runtime !== 'active') {
    throw new Error(`${modelId}: lifecycle.status.runtime must be "active" before publication.`);
  }
  if (status.tested !== 'verified') {
    throw new Error(`${modelId}: lifecycle.status.tested must be "verified" before publication.`);
  }
  if (contracts.executionContractOk !== true) {
    throw new Error(`${modelId}: execution contract gate must be explicitly true in lifecycle.tested.contracts.`);
  }
}

/**
 * Write the new hf.revision back into the local catalog after a successful publish.
 * On --bootstrap, also flips lifecycle.availability.hf to true.
 * Preserves the rest of the JSON structure (key order, indentation) by round-tripping.
 */
export async function writeBackLocalCatalog(catalogFile, modelId, revision, options = {}) {
  const { bootstrap = false } = options;
  const normalizedModelId = normalizeText(modelId);
  const normalizedRevision = normalizeText(revision);
  if (!normalizedModelId) throw new Error('writeBackLocalCatalog requires a modelId.');
  if (!normalizedRevision) throw new Error(`writeBackLocalCatalog requires a non-empty revision for ${normalizedModelId}.`);
  const raw = await readFile(catalogFile, 'utf8');
  const catalog = JSON.parse(raw);
  const entry = Array.isArray(catalog.models)
    ? catalog.models.find((m) => normalizeText(m?.modelId) === normalizedModelId)
    : null;
  if (!entry) {
    throw new Error(`Model "${normalizedModelId}" not found in ${catalogFile} during catalog writeback.`);
  }
  if (!entry.hf || !normalizeText(entry.hf.repoId) || !normalizeText(entry.hf.path)) {
    throw new Error(`Model "${normalizedModelId}" missing hf.repoId or hf.path in ${catalogFile}; cannot write back revision.`);
  }
  entry.hf = { ...entry.hf, revision: normalizedRevision };
  if (bootstrap) {
    const lifecycle = entry.lifecycle && typeof entry.lifecycle === 'object' ? entry.lifecycle : {};
    const availability = lifecycle.availability && typeof lifecycle.availability === 'object'
      ? lifecycle.availability
      : {};
    entry.lifecycle = {
      ...lifecycle,
      availability: { ...availability, hf: true },
    };
  }
  const serialized = JSON.stringify(catalog, null, 2) + (raw.endsWith('\n') ? '\n' : '');
  await writeFile(catalogFile, serialized, 'utf8');
}

function getApprovedModelIds(catalog) {
  const entries = Array.isArray(catalog?.models) ? catalog.models : [];
  return entries
    .filter((entry) => isHostedRegistryApprovedEntry(entry))
    .map((entry) => normalizeText(entry?.modelId))
    .filter(Boolean);
}

async function preflightUploadPlan(args, modelId) {
  const catalog = await loadJsonFile(args.catalogFile, args.catalogFile);
  const localEntry = findCatalogEntry(catalog, modelId);
  if (!localEntry) {
    throw new Error(`Model "${modelId}" not found in ${args.catalogFile}`);
  }
  assertPromotionReady(localEntry, { bootstrap: args.bootstrap, manifestOnly: args.manifestOnly });
  // In bootstrap mode the entry is not yet in the approved set
  // (availability.hf=false). We flip it in memory after a successful upload
  // before rebuilding the hosted registry payload.
  if (!args.bootstrap && !isHostedRegistryApprovedEntry(localEntry)) {
    throw new Error(
      `${modelId}: model is not eligible for the hosted registry; requires hf approval plus active verified runtime status.`
    );
  }

  const uploadPlan = buildArtifactUploadPlan(localEntry, {
    repoId: args.repoId,
    localDir: args.localDir,
    shardDir: args.shardDir,
    localRoot: args.localRoot,
    shardRoot: args.shardRoot,
  });
  const uploadManifest = await assertCompleteUploadArtifact(uploadPlan, { manifestOnly: args.manifestOnly });
  return { localEntry, uploadPlan, uploadManifest };
}

async function publishOne(args, modelId) {
  const catalog = await loadJsonFile(args.catalogFile, args.catalogFile);
  const { uploadPlan, uploadManifest } = await preflightUploadPlan(args, modelId);

  if (args.dryRun) {
    return buildDryRunPayload(uploadPlan, args, uploadManifest);
  }

  // Pre-validate remote registry before uploading any artifact.
  // This fails fast on registry access issues before making irreversible changes.
  await fetchJson(args.registryUrl);

  let uploadResult;
  if (args.manifestOnly) {
    const validationRevision = uploadPlan.sourceRevision || 'main';
    const validationBaseUrl = buildHfResolveUrl(uploadPlan.repoId, validationRevision, uploadPlan.targetPath);
    await verifyPublishedArtifactFiles(validationBaseUrl, uploadManifest);
  }
  const stagingDir = await buildStagingDir(uploadPlan, {
    manifest: uploadManifest,
    manifestOnly: args.manifestOnly,
  });
  try {
    uploadResult = await spawnCommand('hf', [
      'upload',
      uploadPlan.repoId,
      stagingDir,
      uploadPlan.targetPath,
      '--commit-message',
      `Publish ${uploadPlan.modelId} RDRR artifact`,
    ]);
  } finally {
    await rm(stagingDir, { recursive: true, force: true });
  }
  const artifactRevision = extractCommitShaFromUrl(uploadResult.stdout)
    || extractCommitShaFromUrl(uploadResult.stderr)
    || await fetchRepoHeadSha(uploadPlan.repoId);
  if (!artifactRevision) {
    throw new Error(`Could not extract artifact commit SHA from hf upload output for ${uploadPlan.modelId}`);
  }

  // For bootstrap, the in-memory entry still has availability.hf=false and
  // would be filtered out of buildHostedRegistryPayload. Flip it on a clone
  // so the hosted registry includes this model alongside the new revision.
  const catalogForHostedPayload = args.bootstrap
    ? structuredClone(catalog)
    : catalog;
  if (args.bootstrap) {
    const bootstrapEntry = findCatalogEntry(catalogForHostedPayload, uploadPlan.modelId);
    if (!bootstrapEntry) {
      throw new Error(`${uploadPlan.modelId}: entry disappeared from catalog during bootstrap.`);
    }
    const lifecycle = bootstrapEntry.lifecycle && typeof bootstrapEntry.lifecycle === 'object'
      ? bootstrapEntry.lifecycle : {};
    const availability = lifecycle.availability && typeof lifecycle.availability === 'object'
      ? lifecycle.availability : {};
    bootstrapEntry.lifecycle = {
      ...lifecycle,
      availability: { ...availability, hf: true },
    };
  }
  const nextRegistry = buildHostedRegistryPayload(
    catalogForHostedPayload,
    new Map([[uploadPlan.modelId, artifactRevision]])
  );

  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'doppler-hf-registry-'));
  let registryCommit;
  let manifestUrl;
  try {
    const registryFile = path.join(tempDir, 'catalog.json');
    await writeJsonFile(registryFile, nextRegistry);
    const registryResult = await spawnCommand('hf', [
      'upload',
      uploadPlan.repoId,
      registryFile,
      args.registryPath,
      '--commit-message',
      `Publish ${uploadPlan.modelId} registry metadata`,
    ]);
    registryCommit = extractCommitShaFromUrl(registryResult.stdout)
      || extractCommitShaFromUrl(registryResult.stderr)
      || await fetchRepoHeadSha(uploadPlan.repoId);

    const publishedBaseUrl = `https://huggingface.co/${uploadPlan.repoId}/resolve/${artifactRevision}/${uploadPlan.targetPath}`;
    manifestUrl = buildManifestUrl(publishedBaseUrl);
    const manifestProbe = await probeUrl(manifestUrl);
    if (!manifestProbe.ok) {
      throw new Error(`Published manifest did not resolve: ${manifestUrl}`);
    }
    await verifyPublishedArtifactFiles(publishedBaseUrl, uploadManifest);
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }

  // Only write back the local catalog after the remote upload + probe
  // succeeded. A failure earlier leaves the local catalog untouched so a
  // retry starts from the same state.
  await writeBackLocalCatalog(args.catalogFile, uploadPlan.modelId, artifactRevision, {
    bootstrap: args.bootstrap,
  });

  const result = {
    ok: true,
    modelId: uploadPlan.modelId,
    artifactRevision,
    registryCommit,
    manifestUrl,
    localCatalogUpdated: true,
    bootstrap: args.bootstrap,
  };
  console.log(JSON.stringify(result, null, 2));
  return result;
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args.allApproved) {
    const catalog = await loadJsonFile(args.catalogFile, args.catalogFile);
    const modelIds = getApprovedModelIds(catalog);
    const preflighted = [];
    for (const modelId of modelIds) {
      const { uploadPlan, uploadManifest } = await preflightUploadPlan(args, modelId);
      preflighted.push(buildDryRunPayload(uploadPlan, args, uploadManifest));
    }
    if (args.dryRun) {
      console.log(JSON.stringify({
        dryRun: true,
        allApproved: true,
        catalogFile: args.catalogFile,
        modelCount: preflighted.length,
        models: preflighted,
      }, null, 2));
      return;
    }
    const results = [];
    for (const modelId of modelIds) {
      results.push(await publishOne(args, modelId));
    }
    console.log(JSON.stringify({
      ok: true,
      allApproved: true,
      modelCount: results.length,
      results,
    }, null, 2));
    return;
  }

  const result = await publishOne(args, args.modelId);
  if (args.dryRun) {
    console.log(JSON.stringify(result, null, 2));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main()
    .then(() => {
      process.exit(0);
    })
    .catch((error) => {
      console.error(`[publish-hf-registry-model] ${error.message}`);
      process.exit(1);
    });
}
