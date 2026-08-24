import {
  createStreamingHasher,
  loadFileFromStore,
  writeShard,
} from '../shard-manager.js';
import { resolveSourceArtifact } from '../source-artifact-store.js';

export function buildManifestVersionSet(manifest) {
  const sourceArtifact = resolveSourceArtifact(manifest);
  if (!manifest || typeof manifest !== 'object') return 'manifest:invalid';
  const shards = Array.isArray(manifest.shards)
    ? manifest.shards.map((shard, index) => ({
      index,
      filename: shard?.filename ?? null,
      size: shard?.size ?? null,
      hash: shard?.hash ?? null,
    }))
    : [];
  const payload = {
    modelId: manifest.modelId ?? null,
    version: manifest.version ?? null,
    hashAlgorithm: manifest.hashAlgorithm ?? null,
    tensorCount: manifest.tensorCount ?? null,
    totalSize: manifest.totalSize ?? null,
    shards,
    sourceRuntime: sourceArtifact?.sourceRuntime ?? null,
  };
  return JSON.stringify(payload);
}

export function createDefaultSourceStats() {
  return {
    cache: 0,
    p2p: 0,
    http: 0,
    unknown: 0,
  };
}

export function normalizeSourceStats(value) {
  const defaults = createDefaultSourceStats();
  if (!value || typeof value !== 'object') {
    return defaults;
  }
  return {
    cache: Number.isFinite(value.cache) ? Math.max(0, Number(value.cache)) : defaults.cache,
    p2p: Number.isFinite(value.p2p) ? Math.max(0, Number(value.p2p)) : defaults.p2p,
    http: Number.isFinite(value.http) ? Math.max(0, Number(value.http)) : defaults.http,
    unknown: Number.isFinite(value.unknown) ? Math.max(0, Number(value.unknown)) : defaults.unknown,
  };
}

export function isTokenizerJsonRequired(tokenizer) {
  return Boolean(
    tokenizer
    && (tokenizer.type === 'bundled' || tokenizer.type === 'huggingface')
    && typeof tokenizer.file === 'string'
    && tokenizer.file.length > 0
  );
}

export function getTokenizerModelPath(tokenizer) {
  if (!tokenizer || typeof tokenizer !== 'object') {
    return null;
  }
  const explicit = typeof tokenizer.sentencepieceModel === 'string'
    ? tokenizer.sentencepieceModel
    : null;
  if (explicit && explicit.length > 0) {
    return explicit;
  }
  if (tokenizer.type === 'sentencepiece') {
    return 'tokenizer.model';
  }
  return null;
}

// ============================================================================
// IndexedDB Operations
// ============================================================================

export async function fileExistsInStore(path) {
  try {
    await loadFileFromStore(path);
    return true;
  } catch (error) {
    const message = String(error?.message || '');
    return error?.name === 'NotFoundError' || message.toLowerCase().includes('not found')
      ? false
      : Promise.reject(error);
  }
}

export async function computeAssetHash(payload, algorithm = 'sha256') {
  const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  const hasher = await createStreamingHasher(String(algorithm || 'sha256').trim().toLowerCase());
  hasher.update(bytes);
  const digest = await hasher.finalize();
  return Array.from(digest)
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}
export async function persistDownloadedShardIfNeeded(
  result,
  shardIndex,
  options = {}
) {
  const writeShardFn = typeof options.writeShardFn === 'function'
    ? options.writeShardFn
    : writeShard;

  if (!result || typeof result !== 'object') {
    throw new Error(`Shard ${shardIndex}: download result is missing`);
  }
  if (result.wrote === true) {
    return false;
  }
  if (result.source === 'cache') {
    return false;
  }
  if (!(result.buffer instanceof ArrayBuffer)) {
    throw new Error(`Shard ${shardIndex}: source "${result.source}" returned non-persisted data without buffer`);
  }
  await writeShardFn(shardIndex, result.buffer, { verify: false });
  return true;
}

// ============================================================================
// Public API
// ============================================================================
