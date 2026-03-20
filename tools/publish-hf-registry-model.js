#!/usr/bin/env node

import { mkdtemp, rm } from 'node:fs/promises';
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
} from './hf-registry-utils.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_CATALOG_FILE = path.join(REPO_ROOT, 'models', 'catalog.json');

export function parseArgs(argv) {
  const out = {
    modelId: '',
    catalogFile: DEFAULT_CATALOG_FILE,
    localDir: '',
    registryUrl: DEFAULT_HF_REGISTRY_URL,
    registryPath: DEFAULT_HF_REGISTRY_PATH,
    repoId: '',
    dryRun: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--model-id') {
      out.modelId = normalizeText(argv[i + 1]);
      i += 1;
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
    throw new Error(`Unknown argument: ${arg}`);
  }

  if (!out.modelId) {
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
  const localDir = options.localDir
    || path.join(DEFAULT_EXTERNAL_MODELS_ROOT, 'rdrr', modelId);
  if (!repoId) {
    throw new Error(`${modelId}: hf.repoId is required to publish`);
  }
  if (!targetPath) {
    throw new Error(`${modelId}: hf.path is required to publish`);
  }
  return {
    modelId,
    repoId,
    targetPath,
    localDir,
  };
}

export function assertPromotionReady(entry) {
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
  if (lifecycle?.availability?.hf !== true) {
    throw new Error(`${modelId}: lifecycle.availability.hf must be true before publication.`);
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

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const catalog = await loadJsonFile(args.catalogFile, args.catalogFile);
  const localEntry = findCatalogEntry(catalog, args.modelId);
  if (!localEntry) {
    throw new Error(`Model "${args.modelId}" not found in ${args.catalogFile}`);
  }
  assertPromotionReady(localEntry);
  if (!isHostedRegistryApprovedEntry(localEntry)) {
    throw new Error(
      `${args.modelId}: model is not eligible for the hosted registry; requires hf approval plus active verified runtime status.`
    );
  }

  const uploadPlan = buildArtifactUploadPlan(localEntry, {
    repoId: args.repoId,
    localDir: args.localDir,
  });

  if (args.dryRun) {
    console.log(JSON.stringify({
      modelId: uploadPlan.modelId,
      catalogFile: args.catalogFile,
      repoId: uploadPlan.repoId,
      localDir: uploadPlan.localDir,
      targetPath: uploadPlan.targetPath,
      registryPath: args.registryPath,
      registryUrl: args.registryUrl,
    }, null, 2));
    return;
  }

  // Pre-validate remote registry before uploading any artifact.
  // This fails fast on registry access issues before making irreversible changes.
  await fetchJson(args.registryUrl);

  const uploadResult = await spawnCommand('hf', [
    'upload',
    uploadPlan.repoId,
    uploadPlan.localDir,
    uploadPlan.targetPath,
    '--commit-message',
    `Publish ${uploadPlan.modelId} RDRR artifact`,
  ]);
  const artifactRevision = extractCommitShaFromUrl(uploadResult.stdout)
    || extractCommitShaFromUrl(uploadResult.stderr)
    || await fetchRepoHeadSha(uploadPlan.repoId);
  if (!artifactRevision) {
    throw new Error(`Could not extract artifact commit SHA from hf upload output for ${uploadPlan.modelId}`);
  }

  const nextRegistry = buildHostedRegistryPayload(
    catalog,
    new Map([[uploadPlan.modelId, artifactRevision]])
  );

  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'doppler-hf-registry-'));
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
    const registryCommit = extractCommitShaFromUrl(registryResult.stdout)
      || extractCommitShaFromUrl(registryResult.stderr)
      || await fetchRepoHeadSha(uploadPlan.repoId);

    const manifestUrl = buildManifestUrl(`https://huggingface.co/${uploadPlan.repoId}/resolve/${artifactRevision}/${uploadPlan.targetPath}`);
    const manifestProbe = await probeUrl(manifestUrl);
    if (!manifestProbe.ok) {
      throw new Error(`Published manifest did not resolve: ${manifestUrl}`);
    }

    console.log(JSON.stringify({
      ok: true,
      modelId: uploadPlan.modelId,
      artifactRevision,
      registryCommit,
      manifestUrl,
    }, null, 2));
  } finally {
    await rm(tempDir, { recursive: true, force: true });
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
