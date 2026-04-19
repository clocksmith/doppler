

import {
  RDRR_VERSION,
  SHARD_SIZE,
} from './types.js';
import { validateManifest } from './validation.js';
import { createDopplerError, ERROR_CODES } from '../../errors/doppler-error.js';

export function generateShardFilename(index) {
  return `shard_${String(index).padStart(5, '0')}.bin`;
}

export function calculateShardCount(totalSize, shardSize = SHARD_SIZE) {
  return Math.ceil(totalSize / shardSize);
}

export function createManifest(options) {
  const manifest = {
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
    tensorsFile: options.tensorsFile,
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
    throw createDopplerError(
      ERROR_CODES.LOADER_MANIFEST_INVALID,
      `Created invalid manifest:\n  - ${validation.errors.join('\n  - ')}`
    );
  }

  return manifest;
}

export function serializeManifest(manifest) {
  return JSON.stringify(manifest, null, 2);
}

export function getManifestUrl(baseUrl) {
  const base = baseUrl.replace(/\/$/, '');
  return `${base}/manifest.json`;
}
