/**
 * RDRR Parsing Functions
 *
 * @module formats/rdrr/parsing
 */

import type { RDRRManifest, ShardInfo, TensorMap, LayerConfig } from './types.js';
import { validateManifest } from './validation.js';

let currentManifest: RDRRManifest | null = null;

export function parseManifest(jsonString: string): RDRRManifest {
  let manifest: RDRRManifest;

  try {
    manifest = JSON.parse(jsonString);
  } catch (e) {
    throw new Error(`Failed to parse manifest JSON: ${(e as Error).message}`);
  }

  // Normalize shards
  if (Array.isArray(manifest.shards)) {
    let offset = 0;
    manifest.shards = manifest.shards.map((shard: ShardInfo & { fileName?: string }, i: number) => {
      const normalized: ShardInfo = {
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
    const arch = manifest.architecture as LayerConfig;
    arch.numKeyValueHeads = arch.numKeyValueHeads || arch.numAttentionHeads;
    arch.headDim = arch.headDim || Math.floor(arch.hiddenSize / arch.numAttentionHeads);
  }

  currentManifest = manifest;
  return manifest;
}

export function parseTensorMap(jsonString: string): TensorMap {
  try {
    const tensorMap = JSON.parse(jsonString) as TensorMap;

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
    }

    return tensorMap;
  } catch (e) {
    if (e instanceof Error && e.message.includes('Tensor')) {
      throw e;
    }
    throw new Error(`Failed to parse tensors.json: ${(e as Error).message}`);
  }
}

export function getManifest(): RDRRManifest | null {
  return currentManifest;
}

export function setManifest(manifest: RDRRManifest): void {
  currentManifest = manifest;
}

export function clearManifest(): void {
  currentManifest = null;
}

export function getShardInfo(index: number): ShardInfo | null {
  if (!currentManifest || index < 0 || index >= currentManifest.shards.length) {
    return null;
  }
  return currentManifest.shards[index];
}

export function getShardCount(): number {
  return currentManifest?.shards?.length ?? 0;
}

export function isMoE(): boolean {
  return currentManifest?.moeConfig != null ||
    Object.keys(currentManifest?.groups || {}).some(g => g.includes('.expert.'));
}
