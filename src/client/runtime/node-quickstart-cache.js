import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { getManifestUrl } from '../../formats/rdrr/index.js';
import { createStreamingHasher } from '../../storage/shard-manager.js';
import { isNodeRuntime } from '../../utils/runtime-env.js';

const CACHE_DISABLE_VALUES = new Set(['0', 'false', 'off', 'no']);
const DEFAULT_CACHE_ROOT = path.join(os.homedir(), '.cache', 'doppler-gpu', 'models');
const MANIFEST_FILENAME = 'manifest.json';

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isQuickstartRegistrySource(resolved) {
  return Array.isArray(resolved?.trace)
    && resolved.trace.some((entry) => (
      entry?.source === 'quickstart-registry'
      && entry?.outcome === 'resolved'
    ));
}

function isCacheDisabled() {
  const value = normalizeText(process.env.DOPPLER_QUICKSTART_CACHE);
  return CACHE_DISABLE_VALUES.has(value.toLowerCase());
}

function sanitizeModelId(modelId) {
  const normalized = normalizeText(modelId);
  if (!normalized) {
    throw new Error('Node quickstart cache requires modelId.');
  }
  return normalized.replace(/[^a-zA-Z0-9._-]/g, '_');
}

function resolveCacheRoot() {
  const override = normalizeText(process.env.DOPPLER_QUICKSTART_CACHE_DIR);
  return override || DEFAULT_CACHE_ROOT;
}

function normalizeArtifactPath(value) {
  const normalized = normalizeText(value).replace(/\\/g, '/').replace(/^\/+/, '');
  if (!normalized || normalized.split('/').some((part) => part === '.' || part === '..')) {
    throw new Error(`Invalid quickstart cache artifact path "${value}".`);
  }
  return normalized;
}

function normalizeHashAlgorithm(value, fallback = 'sha256') {
  const normalized = normalizeText(value).toLowerCase();
  return normalized || fallback;
}

function normalizeExpectedHash(value) {
  const normalized = normalizeText(value).toLowerCase();
  return normalized || null;
}

function bytesToHex(bytes) {
  return Array.from(bytes)
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

function getShardEntries(manifest) {
  const hashAlgorithm = normalizeHashAlgorithm(manifest?.hashAlgorithm);
  return (Array.isArray(manifest?.shards) ? manifest.shards : []).map((shard, index) => {
    const relativePath = normalizeArtifactPath(shard?.filename);
    const size = Number(shard?.size);
    if (!Number.isFinite(size) || size < 0) {
      throw new Error(`Quickstart cache shard "${relativePath}" has invalid size.`);
    }
    return {
      kind: 'shard',
      index,
      relativePath,
      size: Math.floor(size),
      hash: normalizeExpectedHash(shard?.hash),
      hashAlgorithm: normalizeHashAlgorithm(shard?.hashAlgorithm, hashAlgorithm),
    };
  });
}

function getTokenizerEntries(manifest) {
  const tokenizer = manifest?.tokenizer;
  if (!tokenizer || typeof tokenizer !== 'object') {
    return [];
  }
  const entries = [];
  if (
    (tokenizer.type === 'bundled' || tokenizer.type === 'huggingface')
    && normalizeText(tokenizer.file)
  ) {
    entries.push({
      kind: 'tokenizer',
      relativePath: normalizeArtifactPath(tokenizer.file),
      size: null,
      hash: null,
      hashAlgorithm: null,
    });
  }
  if (normalizeText(tokenizer.sentencepieceModel)) {
    entries.push({
      kind: 'tokenizer',
      relativePath: normalizeArtifactPath(tokenizer.sentencepieceModel),
      size: null,
      hash: null,
      hashAlgorithm: null,
    });
  }
  return entries;
}

function getArtifactEntries(manifest) {
  return [
    ...getShardEntries(manifest),
    ...getTokenizerEntries(manifest),
  ];
}

function joinArtifactUrl(baseUrl, relativePath) {
  return new URL(relativePath, baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`).href;
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function fileMatchesEntry(cacheDir, entry) {
  const filePath = path.join(cacheDir, entry.relativePath);
  try {
    const stat = await fs.stat(filePath);
    if (!stat.isFile()) {
      return false;
    }
    if (entry.size != null) {
      return stat.size === entry.size;
    }
    return stat.size > 0;
  } catch {
    return false;
  }
}

async function cacheIsComplete(cacheDir, manifest) {
  const manifestPath = path.join(cacheDir, MANIFEST_FILENAME);
  if (!await pathExists(manifestPath)) {
    return false;
  }
  let cachedManifest;
  try {
    cachedManifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  } catch {
    return false;
  }
  if (!cachedManifestMatchesRemote(cachedManifest, manifest)) {
    return false;
  }
  const entries = getArtifactEntries(manifest);
  for (const entry of entries) {
    if (!await fileMatchesEntry(cacheDir, entry)) {
      return false;
    }
  }
  return true;
}

function cachedManifestMatchesRemote(cachedManifest, remoteManifest) {
  const cachedEntries = getArtifactEntries(cachedManifest);
  const remoteEntries = getArtifactEntries(remoteManifest);
  if (cachedEntries.length !== remoteEntries.length) {
    return false;
  }
  for (let index = 0; index < remoteEntries.length; index += 1) {
    const cached = cachedEntries[index];
    const remote = remoteEntries[index];
    if (
      cached.kind !== remote.kind
      || cached.relativePath !== remote.relativePath
      || cached.size !== remote.size
      || cached.hash !== remote.hash
      || cached.hashAlgorithm !== remote.hashAlgorithm
    ) {
      return false;
    }
  }
  return true;
}

async function writeManifest(cacheDir, manifestText) {
  await fs.mkdir(cacheDir, { recursive: true });
  await fs.writeFile(path.join(cacheDir, MANIFEST_FILENAME), manifestText);
}

async function downloadArtifactFile(baseUrl, cacheDir, entry, onProgress) {
  const url = joinArtifactUrl(baseUrl, entry.relativePath);
  const targetPath = path.join(cacheDir, entry.relativePath);
  const tempPath = `${targetPath}.tmp-${process.pid}-${Date.now()}`;
  await fs.mkdir(path.dirname(targetPath), { recursive: true });

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download ${entry.relativePath}: ${response.status}`);
  }
  if (!response.body) {
    throw new Error(`Failed to download ${entry.relativePath}: response body is unavailable.`);
  }

  const handle = await fs.open(tempPath, 'w');
  const hasher = entry.hash ? await createStreamingHasher(entry.hashAlgorithm) : null;
  const reader = response.body.getReader();
  let bytesWritten = 0;
  let nextProgressBytes = 16 * 1024 * 1024;
  let complete = false;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      const chunk = value instanceof Uint8Array ? value : new Uint8Array(value);
      if (chunk.byteLength === 0) {
        continue;
      }
      await handle.write(chunk);
      hasher?.update(chunk);
      bytesWritten += chunk.byteLength;
      if (bytesWritten >= nextProgressBytes) {
        onProgress?.({
          phase: 'cache',
          percent: null,
          message: `Caching ${entry.relativePath} (${Math.round(bytesWritten / (1024 * 1024))} MB)`,
        });
        nextProgressBytes += 16 * 1024 * 1024;
      }
    }
    if (entry.size != null && bytesWritten !== entry.size) {
      throw new Error(
        `Downloaded ${entry.relativePath} size mismatch: expected=${entry.size}, got=${bytesWritten}.`
      );
    }
    if (entry.hash && hasher) {
      const computed = bytesToHex(await hasher.finalize());
      if (computed !== entry.hash) {
        throw new Error(
          `Downloaded ${entry.relativePath} hash mismatch: expected=${entry.hash}, got=${computed}.`
        );
      }
    }
    complete = true;
  } finally {
    await handle.close();
    if (!complete) {
      await fs.rm(tempPath, { force: true });
    }
  }

  await fs.rename(tempPath, targetPath);
}

async function populateCache(cacheDir, baseUrl, manifest, manifestText, onProgress) {
  await writeManifest(cacheDir, manifestText);
  const entries = getArtifactEntries(manifest);
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    if (await fileMatchesEntry(cacheDir, entry)) {
      continue;
    }
    onProgress?.({
      phase: 'cache',
      percent: Math.round((index / Math.max(entries.length, 1)) * 100),
      message: `Caching quickstart artifact ${index + 1}/${entries.length}: ${entry.relativePath}`,
    });
    await downloadArtifactFile(baseUrl, cacheDir, entry, onProgress);
  }
  await writeManifest(cacheDir, manifestText);
}

export async function resolveNodeQuickstartCachedSource(resolved, manifestPayload, options = {}) {
  if (!isNodeRuntime() || isCacheDisabled() || !isQuickstartRegistrySource(resolved)) {
    return null;
  }
  const baseUrl = normalizeText(resolved?.baseUrl);
  if (!baseUrl) {
    return null;
  }
  const manifest = manifestPayload?.manifest;
  const manifestText = normalizeText(manifestPayload?.text)
    || JSON.stringify(manifest);
  if (!manifest || typeof manifest !== 'object') {
    return null;
  }

  const modelId = normalizeText(resolved?.modelId) || normalizeText(manifest.modelId);
  const cacheDir = path.join(resolveCacheRoot(), sanitizeModelId(modelId));
  const complete = await cacheIsComplete(cacheDir, manifest);
  if (complete) {
    await writeManifest(cacheDir, manifestText);
    options.onProgress?.({
      phase: 'cache',
      percent: 100,
      message: `Quickstart cache hit: ${modelId}`,
    });
    return {
      ...resolved,
      baseUrl: pathToFileURL(cacheDir).href,
      manifest,
      cache: {
        state: 'hit',
        dir: cacheDir,
      },
    };
  }

  options.onProgress?.({
    phase: 'cache',
    percent: 0,
    message: `Quickstart cache miss: ${modelId}`,
  });
  await populateCache(cacheDir, baseUrl, manifest, manifestText, options.onProgress);
  options.onProgress?.({
    phase: 'cache',
    percent: 100,
    message: `Quickstart cache ready: ${modelId}`,
  });
  return {
    ...resolved,
    baseUrl: pathToFileURL(cacheDir).href,
    manifest,
    cache: {
      state: 'imported',
      dir: cacheDir,
    },
  };
}
