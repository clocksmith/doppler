/**
 * RDRR Manifest Creation and Serialization
 *
 * @module formats/rdrr/manifest
 */

import {
  RDRR_VERSION,
  SHARD_SIZE,
  TENSORS_FILENAME,
  type RDRRManifest,
  type ShardInfo,
  type TensorMap,
  type CreateManifestOptions,
} from './types.js';
import { validateManifest } from './validation.js';
import { getShardInfo } from './parsing.js';

export function generateShardFilename(index: number): string {
  return `shard_${String(index).padStart(5, '0')}.bin`;
}

export function calculateShardCount(totalSize: number, shardSize = SHARD_SIZE): number {
  return Math.ceil(totalSize / shardSize);
}

export function createShardLayout(
  totalSize: number,
  hashes: string[],
  shardSize = SHARD_SIZE
): ShardInfo[] {
  const numShards = calculateShardCount(totalSize, shardSize);

  if (hashes.length !== numShards) {
    throw new Error(`Hash count mismatch: expected ${numShards}, got ${hashes.length}`);
  }

  const shards: ShardInfo[] = [];
  let offset = 0;

  for (let i = 0; i < numShards; i++) {
    const isLast = i === numShards - 1;
    const size = isLast ? totalSize - offset : shardSize;

    shards.push({
      index: i,
      filename: generateShardFilename(i),
      size,
      hash: hashes[i],
      blake3: hashes[i],
      offset,
    });

    offset += size;
  }

  return shards;
}

export function createManifest(options: CreateManifestOptions): RDRRManifest {
  const manifest: RDRRManifest = {
    version: RDRR_VERSION,
    modelId: options.modelId,
    modelType: options.modelType,
    quantization: options.quantization,
    quantizationInfo: options.quantizationInfo,
    hashAlgorithm: options.hashAlgorithm,
    architecture: options.architecture,
    groups: options.groups,
    shards: options.shards,
    totalSize: options.totalSize,
    tensorsFile: options.tensorsFile || TENSORS_FILENAME,
    tensorCount: options.tensorCount,
    tokenizer: options.tokenizer,
    moeConfig: options.moeConfig,
    config: options.config,
    conversion: options.conversion,
    blake3Full: options.blake3Full,
    metadata: options.metadata,
  };

  const validation = validateManifest(manifest);
  if (!validation.valid) {
    throw new Error(`Created invalid manifest:\n  - ${validation.errors.join('\n  - ')}`);
  }

  return manifest;
}

export function serializeTensorMap(tensorMap: TensorMap): string {
  return JSON.stringify(tensorMap, null, 2);
}

export function serializeManifest(manifest: RDRRManifest): string {
  return JSON.stringify(manifest, null, 2);
}

export function getShardUrl(baseUrl: string, shardIndex: number): string {
  const shard = getShardInfo(shardIndex);
  if (!shard) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }
  const base = baseUrl.replace(/\/$/, '');
  return `${base}/${shard.filename}`;
}

export function getManifestUrl(baseUrl: string): string {
  const base = baseUrl.replace(/\/$/, '');
  return `${base}/manifest.json`;
}
