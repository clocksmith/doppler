

import { validateManifest } from './validation.js';

let currentManifest = null;

export function parseManifest(jsonString) {
  let manifest;

  try {
    manifest = JSON.parse(jsonString);
  } catch (e) {
    throw new Error(`Failed to parse manifest JSON: ${e.message}`);
  }

  // Normalize shards
  if (Array.isArray(manifest.shards)) {
    let offset = 0;
    manifest.shards = manifest.shards.map((shard, i) => {
      const normalized = {
        index: shard.index ?? i,
        filename: shard.filename || shard.fileName || '',
        size: shard.size,
        hash: shard.hash || shard.blake3 || '',
        blake3: shard.blake3 || shard.hash,
        offset: shard.offset ?? offset,
        hashAlgorithm: shard.hashAlgorithm,
      };
      offset += shard.size;
      return normalized;
    });
  }

  // Validate
  const validation = validateManifest(manifest);
  if (!validation.valid) {
    throw new Error(`Invalid manifest:\n  - ${validation.errors.join('\n  - ')}`);
  }

  currentManifest = manifest;
  return manifest;
}

export function parseTensorMap(jsonString) {
  try {
    const tensorMap = JSON.parse(jsonString);

    for (const [name, loc] of Object.entries(tensorMap)) {
      if (typeof loc.shard !== 'number') {
        throw new Error(`Tensor '${name}' missing shard index`);
      }
      if (typeof loc.offset !== 'number') {
        throw new Error(`Tensor '${name}' missing offset`);
      }
      if (typeof loc.size !== 'number') {
        throw new Error(`Tensor '${name}' missing size`);
      }
      if (!Array.isArray(loc.shape)) {
        throw new Error(`Tensor '${name}' missing shape`);
      }
      if (typeof loc.role !== 'string') {
        throw new Error(`Tensor '${name}' missing role`);
      }
    }

    return tensorMap;
  } catch (e) {
    if (e instanceof Error && e.message.includes('Tensor')) {
      throw e;
    }
    throw new Error(`Failed to parse tensors.json: ${e.message}`);
  }
}

export function getManifest() {
  return currentManifest;
}

export function setManifest(manifest) {
  currentManifest = manifest;
}

export function clearManifest() {
  currentManifest = null;
}

export function getShardInfo(index) {
  if (!currentManifest || index < 0 || index >= currentManifest.shards.length) {
    return null;
  }
  return currentManifest.shards[index];
}

export function getShardCount() {
  return currentManifest?.shards?.length ?? 0;
}

export function isMoE() {
  return currentManifest?.moeConfig != null ||
    Object.keys(currentManifest?.groups || {}).some(g => g.includes('.expert.'));
}
