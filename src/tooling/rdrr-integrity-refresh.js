import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';

import { buildIntegrityExtensions } from '../formats/rdrr/integrity.js';
import { parseManifest, parseTensorMap } from '../formats/rdrr/parsing.js';

function hashBytesSha256Node(bytes) {
  const view = bytes instanceof Uint8Array
    ? bytes
    : new Uint8Array(bytes);
  return `sha256:${createHash('sha256').update(view).digest('hex')}`;
}

function asObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value;
}

function asNonEmptyString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return value.trim();
}

function cloneMetadataWithIntegrityRefresh(metadata, blockSize) {
  const next = metadata && typeof metadata === 'object' && !Array.isArray(metadata)
    ? { ...metadata }
    : {};
  next.integrityRefresh = {
    ...(next.integrityRefresh && typeof next.integrityRefresh === 'object' ? next.integrityRefresh : {}),
    at: new Date().toISOString(),
    blockSize,
  };
  return next;
}

async function readJsonObject(filePath, label) {
  const raw = await fs.readFile(filePath, 'utf8');
  const parsed = JSON.parse(raw);
  return asObject(parsed, label);
}

async function loadTensorMap(manifest, modelDir) {
  if (manifest.tensors && typeof manifest.tensors === 'object' && !Array.isArray(manifest.tensors)) {
    return manifest.tensors;
  }
  if (typeof manifest.tensorsFile === 'string' && manifest.tensorsFile.trim()) {
    const tensorMapPath = path.join(modelDir, manifest.tensorsFile);
    const raw = await fs.readFile(tensorMapPath, 'utf8');
    return parseTensorMap(raw);
  }
  throw new Error('RDRR integrity refresh requires manifest.tensors or manifest.tensorsFile.');
}

async function verifyShardFiles(modelDir, manifest) {
  const shards = Array.isArray(manifest?.shards) ? manifest.shards : [];
  for (const shard of shards) {
    const filename = asNonEmptyString(shard?.filename, 'manifest.shards[].filename');
    await fs.access(path.join(modelDir, filename));
  }
}

function buildShardIndexMap(manifest) {
  const map = new Map();
  for (const shard of Array.isArray(manifest?.shards) ? manifest.shards : []) {
    const index = Number(shard?.index);
    if (Number.isInteger(index) && index >= 0) {
      map.set(index, shard);
    }
  }
  return map;
}

export async function buildManifestIntegrityFromModelDir(manifest, options = {}) {
  const normalizedManifest = asObject(manifest, 'manifest');
  const modelDir = path.resolve(asNonEmptyString(options?.modelDir, 'modelDir'));
  const tensorMap = options?.tensorMap ?? await loadTensorMap(normalizedManifest, modelDir);
  const readRange = typeof options?.readRange === 'function'
    ? options.readRange
    : async (filePath, offset, length) => {
      const handle = await fs.open(filePath, 'r');
      try {
        const buffer = Buffer.alloc(length);
        const { bytesRead } = await handle.read(buffer, 0, length, offset);
        return buffer.subarray(0, bytesRead);
      } finally {
        await handle.close();
      }
    };
  const shardIndexMap = buildShardIndexMap(normalizedManifest);

  return buildIntegrityExtensions(normalizedManifest, {
    tensorMap,
    blockSize: options?.blockSize,
    onProgress: options?.onProgress,
    hashBlockBytesSha256: options?.hashBlockBytesSha256 ?? hashBytesSha256Node,
    async readShardRange(shardIndex, offset, length) {
      const shard = shardIndexMap.get(shardIndex);
      if (!shard || typeof shard !== 'object') {
        throw new Error(`Missing shard descriptor for shard ${shardIndex}.`);
      }
      const shardPath = path.join(modelDir, asNonEmptyString(shard.filename, `manifest.shards[${shardIndex}].filename`));
      return readRange(shardPath, offset, length);
    },
  });
}

export async function refreshManifestIntegrity(options) {
  const modelDir = path.resolve(asNonEmptyString(options?.modelDir, 'modelDir'));
  const manifestPath = path.resolve(options?.manifestPath ?? path.join(modelDir, 'manifest.json'));
  const dryRun = options?.dryRun === true;
  const skipShardCheck = options?.skipShardCheck === true;
  const manifestJson = await readJsonObject(manifestPath, 'manifest');
  const manifest = parseManifest(JSON.stringify(manifestJson));

  if (!skipShardCheck) {
    await verifyShardFiles(modelDir, manifest);
  }

  const { integrityExtensions, integrityExtensionsHash } = await buildManifestIntegrityFromModelDir(
    manifest,
    {
      modelDir,
      blockSize: options?.blockSize,
      onProgress: options?.onProgress,
    }
  );

  const refreshed = parseManifest(JSON.stringify({
    ...manifest,
    integrityExtensions,
    metadata: cloneMetadataWithIntegrityRefresh(manifest.metadata, integrityExtensions.blockMerkle.blockSize),
  }));

  if (!dryRun) {
    await fs.writeFile(manifestPath, JSON.stringify(refreshed, null, 2), 'utf8');
  }

  return {
    manifestPath,
    manifest: refreshed,
    integrityExtensions,
    integrityExtensionsHash,
    wrote: !dryRun,
  };
}
