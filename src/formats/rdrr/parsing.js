

import { validateManifest } from './validation.js';
import { RDRR_VERSION } from './types.js';

const KNOWN_MANIFEST_VERSIONS = new Set([RDRR_VERSION]);

let currentManifest = null;

export function getExpectedShardHash(shard, manifestHashAlgorithm = null) {
  if (!shard || typeof shard !== 'object' || Array.isArray(shard)) {
    return '';
  }
  const algorithm = typeof manifestHashAlgorithm === 'string'
    ? manifestHashAlgorithm.trim().toLowerCase()
    : '';
  if (algorithm === 'blake3') {
    return shard.blake3 || shard.hash || '';
  }
  return shard.hash || shard.blake3 || '';
}

export function parseManifest(jsonString) {
  let manifest;

  try {
    manifest = JSON.parse(jsonString);
  } catch (e) {
    throw new Error(`Failed to parse manifest JSON: ${e.message}`);
  }

  // Warn on unknown manifest version so callers know they may be reading a
  // newer format than this parser understands.
  const parsedVersion = typeof manifest.version === 'string'
    ? parseFloat(manifest.version)
    : manifest.version;
  if (typeof parsedVersion === 'number' && !KNOWN_MANIFEST_VERSIONS.has(parsedVersion)) {
    console.warn(
      `[RDRR] Unknown manifest version ${manifest.version} ` +
      `(known: ${[...KNOWN_MANIFEST_VERSIONS].join(', ')}). ` +
      'Parsing will continue but results may be unreliable.'
    );
  }

  // Normalize shards (handle fileName vs filename, compute offset if missing)
  if (Array.isArray(manifest.shards)) {
    let offset = 0;
    manifest.shards = manifest.shards.map((shard, i) => {
      const normalized = {
        index: shard.index ?? i,
        filename: shard.filename || shard.fileName || '',
        size: shard.size,
        hash: getExpectedShardHash(shard, manifest.hashAlgorithm),
        blake3: shard.blake3 || shard.hash,
        offset: shard.offset ?? offset,
        hashAlgorithm: shard.hashAlgorithm,
      };
      offset += shard.size;
      return normalized;
    });
  }

  // Migration compat: promote sessionDefaults → session during transition.
  // Also fill in kvcache/decodeLoop fields that older manifests omit but the
  // contract now requires. Remove this block once all HF artifacts are regenerated.
  if (manifest.inference) {
    if (manifest.inference.sessionDefaults && !manifest.inference.session) {
      manifest.inference.session = manifest.inference.sessionDefaults;
    }
    const s = manifest.inference.session;
    if (s && typeof s === 'object') {
      if (s.kvcache && typeof s.kvcache === 'object') {
        if (!s.kvcache.layout) s.kvcache.layout = 'contiguous';
        if (!s.kvcache.pageSize) s.kvcache.pageSize = 256;
        if (!s.kvcache.tiering) s.kvcache.tiering = { mode: 'off' };
        else if (!s.kvcache.tiering.mode) s.kvcache.tiering.mode = 'off';
      }
      if (!s.decodeLoop || typeof s.decodeLoop !== 'object') {
        s.decodeLoop = { batchSize: 4, stopCheckMode: 'batch', readbackInterval: 1, ringTokens: 1, ringStop: 1, ringStaging: 1, disableCommandBatching: false };
      } else {
        if (!s.decodeLoop.stopCheckMode) s.decodeLoop.stopCheckMode = 'batch';
        if (!s.decodeLoop.batchSize) s.decodeLoop.batchSize = 4;
        if (s.decodeLoop.disableCommandBatching == null) s.decodeLoop.disableCommandBatching = false;
        if (!s.decodeLoop.readbackInterval) s.decodeLoop.readbackInterval = 1;
        if (!s.decodeLoop.ringTokens) s.decodeLoop.ringTokens = 1;
        if (!s.decodeLoop.ringStop) s.decodeLoop.ringStop = 1;
        if (!s.decodeLoop.ringStaging) s.decodeLoop.ringStaging = 1;
      }
    }
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
    const normalizedTensorMap = {};

    for (const [name, loc] of Object.entries(tensorMap)) {
      const shardIndex = typeof loc.shardIndex === 'number'
        ? loc.shardIndex
        : loc.shard;
      if (typeof shardIndex !== 'number') {
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

      let spans = undefined;
      if (loc.spans !== undefined) {
        if (!Array.isArray(loc.spans)) {
          throw new Error(`Tensor '${name}' has invalid spans array`);
        }
        spans = loc.spans.map((span, spanIndex) => {
          const spanShardIndex = typeof span?.shardIndex === 'number'
            ? span.shardIndex
            : span?.shard;
          if (typeof spanShardIndex !== 'number') {
            throw new Error(`Tensor '${name}' span[${spanIndex}] missing shard index`);
          }
          if (typeof span?.offset !== 'number') {
            throw new Error(`Tensor '${name}' span[${spanIndex}] missing offset`);
          }
          if (typeof span?.size !== 'number') {
            throw new Error(`Tensor '${name}' span[${spanIndex}] missing size`);
          }
          return {
            shardIndex: spanShardIndex,
            offset: span.offset,
            size: span.size,
          };
        });
      }

      normalizedTensorMap[name] = {
        ...loc,
        shard: shardIndex,
        shardIndex,
        spans,
      };
    }

    return normalizedTensorMap;
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
