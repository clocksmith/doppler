/**
 * RDRR Manifest Creation and Serialization
 *
 * @module formats/rdrr/manifest
 */

import {
  RDRR_VERSION,
  SHARD_SIZE,
  TENSORS_FILENAME,
} from './types.js';
import { validateManifest } from './validation.js';
import { getShardInfo } from './parsing.js';

export function generateShardFilename(index) {
  return `shard_${String(index).padStart(5, '0')}.bin`;
}

export function calculateShardCount(totalSize, shardSize = SHARD_SIZE) {
  return Math.ceil(totalSize / shardSize);
}

export function createShardLayout(
  totalSize,
  hashes,
  shardSize = SHARD_SIZE
) {
  const numShards = calculateShardCount(totalSize, shardSize);

  if (hashes.length !== numShards) {
    throw new Error(`Hash count mismatch: expected ${numShards}, got ${hashes.length}`);
  }

  const shards = [];
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

export function createManifest(options) {
  const manifest = {
    version: RDRR_VERSION,
    modelId: options.modelId,
    modelType: options.modelType,
    quantization: options.quantization,
    quantizationInfo: options.quantizationInfo,
    hashAlgorithm: options.hashAlgorithm ?? 'sha256',
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
    inference: options.inference,
  };

  const validation = validateManifest(manifest);
  if (!validation.valid) {
    throw new Error(`Created invalid manifest:\n  - ${validation.errors.join('\n  - ')}`);
  }

  return manifest;
}

export function serializeTensorMap(tensorMap) {
  return JSON.stringify(tensorMap, null, 2);
}

export function serializeManifest(manifest) {
  return JSON.stringify(manifest, null, 2);
}

export function getShardUrl(baseUrl, shardIndex) {
  const shard = getShardInfo(shardIndex);
  if (!shard) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }
  const base = baseUrl.replace(/\/$/, '');
  return `${base}/${shard.filename}`;
}

export function getManifestUrl(baseUrl) {
  const base = baseUrl.replace(/\/$/, '');
  return `${base}/manifest.json`;
}
