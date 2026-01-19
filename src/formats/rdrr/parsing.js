

import { validateManifest } from './validation.js';

let currentManifest = null;

function isLegacyManifest(manifest) {
  if (!manifest) return false;
  const version = typeof manifest.version === 'string'
    ? parseFloat(manifest.version)
    : manifest.version;
  const hasLegacyConfig = Boolean(
    manifest.config && (
      manifest.config.model_type ||
      manifest.config.architectures ||
      manifest.config.transformers_version
    )
  );
  return version === 1 && !manifest.groups && hasLegacyConfig;
}

export function parseManifest(jsonString) {
  let manifest;

  try {
    manifest = JSON.parse(jsonString);
  } catch (e) {
    throw new Error(`Failed to parse manifest JSON: ${e.message}`);
  }

  // Normalize eos_token_id from legacy config if not at top level
  if (manifest.eos_token_id === undefined &&
      manifest.config?.eos_token_id !== undefined &&
      isLegacyManifest(manifest)) {
    manifest.eos_token_id = manifest.config.eos_token_id;
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

  // Normalize optional fields
  manifest.metadata = manifest.metadata || {};

  // Normalize architecture
  if (manifest.architecture && typeof manifest.architecture === 'object') {
    const arch = manifest.architecture;
    arch.numKeyValueHeads = arch.numKeyValueHeads || arch.numAttentionHeads;
    arch.headDim = arch.headDim || Math.floor(arch.hiddenSize / arch.numAttentionHeads);
  }

  currentManifest = manifest;
  return manifest;
}

export function parseTensorMap(jsonString) {
  try {
    const tensorMap = JSON.parse(jsonString);
    const allowLegacyRoleInference = isLegacyManifest(currentManifest);

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
      // Normalize group to role (backward compatibility)
      // Legacy tensors may include group but no role
      if (allowLegacyRoleInference && loc.role === undefined && loc.group !== undefined) {
        if (loc.group === 'embed') {
          loc.role = 'embedding';
        } else if (loc.group === 'head') {
          if (name === 'model.norm.weight' || name === 'final_norm.weight') {
            loc.role = 'norm';
          } else if (name.includes('lm_head') || name === 'output.weight') {
            loc.role = 'lm_head';
          }
        } else {
          loc.role = loc.group;
        }
      }
      // Normalize legacy role names
      if (loc.role === 'embed') {
        loc.role = 'embedding';
      }
      // Normalize legacy groups for known roles
      if (loc.group === undefined) {
        if (loc.role === 'embedding') {
          loc.group = 'embed';
        } else if (loc.role === 'norm' || loc.role === 'lm_head') {
          loc.group = 'head';
        }
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
