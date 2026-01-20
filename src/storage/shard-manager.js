import {
  getManifest,
  getShardInfo,
  getShardCount,
} from './rdrr-format.js';
import {
  isOPFSAvailable,
  isIndexedDBAvailable,
  QuotaExceededError,
  checkSpaceAvailable,
} from './quota.js';
import { log } from '../debug/index.js';
import { createHasher as createBlake3Hasher, hash as blake3Hash } from './blake3.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { createOpfsStore } from './backends/opfs-store.js';
import { createIdbStore } from './backends/idb-store.js';
import { createMemoryStore } from './backends/memory-store.js';

export { getManifest } from './rdrr-format.js';

let opfsPathConfigOverride = null;
let blake3Module = null;
let hashAlgorithm = null;

let backend = null;
let backendType = null;
let currentModelId = null;

export function setOpfsPathConfig(config) {
  opfsPathConfigOverride = config;
}

export function getOpfsPathConfig() {
  return opfsPathConfigOverride ?? getRuntimeConfig().loading.opfsPath;
}

function normalizeModelId(modelId) {
  return modelId.replace(/[^a-zA-Z0-9_-]/g, '_');
}

function getBackendConfig() {
  return getRuntimeConfig().loading.storage.backend;
}

function buildBackend(type, config) {
  if (type === 'opfs') {
    return createOpfsStore({
      opfsRootDir: getOpfsPathConfig().opfsRootDir,
      useSyncAccessHandle: config.opfs.useSyncAccessHandle,
      maxConcurrentHandles: config.opfs.maxConcurrentHandles,
    });
  }
  if (type === 'indexeddb') {
    return createIdbStore(config.indexeddb);
  }
  return createMemoryStore(config.memory);
}

function resolveBackendType(config) {
  if (config.backend === 'opfs') {
    if (!isOPFSAvailable()) {
      throw new Error('OPFS requested but not available');
    }
    return 'opfs';
  }
  if (config.backend === 'indexeddb') {
    if (!isIndexedDBAvailable()) {
      throw new Error('IndexedDB requested but not available');
    }
    return 'indexeddb';
  }
  if (config.backend === 'memory') {
    return 'memory';
  }
  if (isOPFSAvailable()) return 'opfs';
  if (isIndexedDBAvailable()) return 'indexeddb';
  return 'memory';
}

async function initBlake3(requiredAlgorithm = null) {
  if (blake3Module && hashAlgorithm) return;

  try {
    blake3Module = {
      hash: blake3Hash,
      createHasher: createBlake3Hasher,
    };
    hashAlgorithm = 'blake3';
    return;
  } catch (e) {
    log.warn('ShardManager', `BLAKE3 module not available: ${e.message}`);
  }

  if (requiredAlgorithm === 'blake3') {
    throw new Error(
      'BLAKE3 required by manifest but not available. ' +
      'Install the JS blake3 module or re-convert model with SHA-256.'
    );
  }

  hashAlgorithm = 'sha256';
  blake3Module = {
    hash: async (data) => {
      const hashBuffer = await crypto.subtle.digest('SHA-256', data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
      return new Uint8Array(hashBuffer);
    },
    createHasher: () => {
      const chunks = [];
      return {
        update: (data) => {
          chunks.push(new Uint8Array(data));
        },
        finalize: async () => {
          const totalLength = chunks.reduce((acc, c) => acc + c.length, 0);
          const combined = new Uint8Array(totalLength);
          let offset = 0;
          for (const chunk of chunks) {
            combined.set(chunk, offset);
            offset += chunk.length;
          }
          const hashBuffer = await crypto.subtle.digest('SHA-256', combined);
          return new Uint8Array(hashBuffer);
        }
      };
    }
  };
}

export function getHashAlgorithm() {
  return hashAlgorithm;
}

function bytesToHex(bytes) {
  return Array.from(bytes)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

export function hexToBytes(hex) {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(hex.substr(i * 2, 2), 16);
  }
  return bytes;
}

export async function computeBlake3(data) {
  await initBlake3('blake3');

  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  const hash = await blake3Module.hash(bytes);
  return bytesToHex(hash);
}

export async function computeSHA256(data) {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  return bytesToHex(new Uint8Array(hashBuffer));
}

export async function computeHash(data, algorithm) {
  if (!algorithm) {
    throw new Error('computeHash requires an explicit hash algorithm.');
  }
  if (algorithm === 'sha256') {
    return computeSHA256(data);
  }
  return computeBlake3(data);
}

export async function createStreamingHasher(algorithm) {
  if (!algorithm) {
    throw new Error('createStreamingHasher requires an explicit hash algorithm.');
  }
  if (algorithm === 'sha256') {
    const chunks = [];
    return {
      update: (data) => {
        chunks.push(new Uint8Array(data));
      },
      finalize: async () => {
        const totalLength = chunks.reduce((acc, c) => acc + c.length, 0);
        const combined = new Uint8Array(totalLength);
        let offset = 0;
        for (const chunk of chunks) {
          combined.set(chunk, offset);
          offset += chunk.length;
        }
        const hashBuffer = await crypto.subtle.digest('SHA-256', combined);
        return new Uint8Array(hashBuffer);
      }
    };
  }
  await initBlake3('blake3');
  return blake3Module.createHasher();
}

function requireManifestHashAlgorithm(manifest, context) {
  const algorithm = manifest?.hashAlgorithm;
  if (!algorithm) {
    throw new Error(
      `Manifest missing hashAlgorithm for ${context}. ` +
      'Re-convert the model to include a manifest hash algorithm.'
    );
  }
  return algorithm;
}

export function getStorageCapabilities() {
  const hasReadableStream = typeof ReadableStream !== 'undefined';
  const supportsByob = hasReadableStream && typeof ReadableStreamBYOBReader !== 'undefined';
  const supportsSyncAccessHandle = typeof FileSystemSyncAccessHandle !== 'undefined';
  return {
    opfs: isOPFSAvailable(),
    indexeddb: isIndexedDBAvailable(),
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    byob: supportsByob,
    syncAccessHandle: supportsSyncAccessHandle,
  };
}

export function getStorageBackendType() {
  return backendType;
}

export async function initStorage() {
  if (backend) return;
  const backendConfig = getBackendConfig();
  backendType = resolveBackendType(backendConfig);
  backend = buildBackend(backendType, backendConfig);
  await backend.init();
}

export async function openModelStore(modelId) {
  if (!backend) {
    await initStorage();
  }
  const safeName = normalizeModelId(modelId);
  currentModelId = safeName;
  return backend.openModel(safeName, { create: true });
}

export function getCurrentModelId() {
  return currentModelId;
}

function requireModel() {
  if (!currentModelId) {
    throw new Error('No model open. Call openModelStore first.');
  }
}

async function ensureBackend() {
  if (!backend) {
    await initStorage();
  }
}

export async function writeShard(shardIndex, data, options = { verify: true }) {
  await ensureBackend();
  requireModel();

  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }

  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  const spaceCheck = await checkSpaceAvailable(bytes.byteLength);
  if (!spaceCheck.hasSpace) {
    throw new QuotaExceededError(bytes.byteLength, spaceCheck.info.available);
  }

  try {
    await backend.writeFile(shardInfo.filename, bytes);

    if (options.verify) {
      const manifest = getManifest();
      const algorithm = requireManifestHashAlgorithm(manifest, 'shard write');
      const hash = await computeHash(bytes, algorithm);
      const expectedHash = shardInfo.hash;
      if (!expectedHash) {
        await backend.deleteFile(shardInfo.filename);
        throw new Error(`Shard ${shardIndex} is missing hash in manifest`);
      }
      if (hash !== expectedHash) {
        await backend.deleteFile(shardInfo.filename);
        throw new Error(`Hash mismatch for shard ${shardIndex}: expected ${expectedHash}, got ${hash}`);
      }
      return { success: true, hash };
    }

    return { success: true, hash: null };
  } catch (error) {
    if (error instanceof QuotaExceededError) throw error;
    throw new Error(`Failed to write shard ${shardIndex}: ${error.message}`);
  }
}

export async function createShardWriter(shardIndex) {
  await ensureBackend();
  requireModel();
  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }
  if (!backend.createWriteStream) {
    throw new Error('Storage backend does not support streaming writes');
  }
  return backend.createWriteStream(shardInfo.filename);
}

export async function loadShard(shardIndex, options = { verify: false }) {
  await ensureBackend();
  requireModel();

  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }

  try {
    const buffer = await backend.readFile(shardInfo.filename);
    if (options.verify) {
      const manifest = getManifest();
      const algorithm = requireManifestHashAlgorithm(manifest, 'shard load');
      const hash = await computeHash(buffer, algorithm);
      const expectedHash = shardInfo.hash;
      if (!expectedHash) {
        throw new Error(`Shard ${shardIndex} is missing hash in manifest`);
      }
      if (hash !== expectedHash) {
        throw new Error(`Hash mismatch for shard ${shardIndex}: expected ${expectedHash}, got ${hash}`);
      }
    }
    return buffer;
  } catch (error) {
    if (error.name === 'NotFoundError') {
      throw new Error(`Shard ${shardIndex} not found`);
    }
    throw new Error(`Failed to load shard ${shardIndex}: ${error.message}`);
  }
}

export async function loadShardSync(shardIndex, offset = 0, length) {
  const buffer = await loadShard(shardIndex, { verify: false });
  const view = new Uint8Array(buffer);
  const start = Math.max(0, offset);
  const end = length == null ? view.length : Math.min(view.length, start + length);
  return view.slice(start, end);
}

export async function shardExists(shardIndex) {
  await ensureBackend();
  requireModel();
  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) return false;
  try {
    await backend.readFile(shardInfo.filename);
    return true;
  } catch {
    return false;
  }
}

export async function verifyIntegrity() {
  const manifest = getManifest();
  if (!manifest) {
    throw new Error('No manifest loaded');
  }

  const algorithm = requireManifestHashAlgorithm(manifest, 'integrity check');

  const missingShards = [];
  const corruptShards = [];
  const shardCount = getShardCount();

  for (let i = 0; i < shardCount; i++) {
    const exists = await shardExists(i);
    if (!exists) {
      missingShards.push(i);
      continue;
    }

    try {
      const buffer = await loadShard(i, { verify: false });
      const hash = await computeHash(buffer, algorithm);
      const shardInfo = getShardInfo(i);
      const expectedHash = shardInfo?.hash;
      if (!expectedHash) {
        corruptShards.push(i);
        continue;
      }
      if (hash !== expectedHash) {
        corruptShards.push(i);
      }
    } catch (_error) {
      corruptShards.push(i);
    }
  }

  return {
    valid: missingShards.length === 0 && corruptShards.length === 0,
    missingShards,
    corruptShards
  };
}

export async function deleteShard(shardIndex) {
  await ensureBackend();
  requireModel();
  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) return false;
  try {
    await backend.deleteFile(shardInfo.filename);
    return true;
  } catch {
    return false;
  }
}

export async function deleteModel(modelId) {
  await ensureBackend();
  const safeName = normalizeModelId(modelId);
  return backend.deleteModel(safeName);
}

export async function listModels() {
  await ensureBackend();
  return backend.listModels();
}

export async function getModelInfo(modelId) {
  await ensureBackend();
  const safeName = normalizeModelId(modelId);
  let exists = false;
  let hasManifest = false;
  let shardCount = 0;
  let totalSize = 0;

  const models = await backend.listModels();
  exists = models.includes(safeName);
  if (!exists) {
    return { exists: false, shardCount: 0, totalSize: 0, hasManifest: false };
  }

  try {
    await backend.openModel(safeName, { create: false });
    const manifestJson = await loadManifestFromStore();
    hasManifest = !!manifestJson;
    if (manifestJson) {
      try {
        const manifest = JSON.parse(manifestJson);
        shardCount = manifest.shards?.length ?? 0;
        totalSize = manifest.totalSize ?? 0;
      } catch {
        shardCount = 0;
        totalSize = 0;
      }
    }
  } catch {
    return { exists: false, shardCount: 0, totalSize: 0, hasManifest: false };
  }

  return { exists: true, shardCount, totalSize, hasManifest };
}

export async function modelExists(modelId) {
  const info = await getModelInfo(modelId);
  return info.exists && info.hasManifest;
}

export async function saveManifest(manifestJson) {
  await ensureBackend();
  requireModel();
  if (backend.writeManifest) {
    await backend.writeManifest(manifestJson);
    return;
  }
  const encoder = new TextEncoder();
  await backend.writeFile('manifest.json', encoder.encode(manifestJson));
}

export async function loadManifestFromStore() {
  await ensureBackend();
  requireModel();
  if (backend.readManifest) {
    return backend.readManifest();
  }
  if (backend.readText) {
    return backend.readText('manifest.json');
  }
  const buffer = await backend.readFile('manifest.json');
  return new TextDecoder().decode(buffer);
}

export async function loadTensorsFromStore() {
  await ensureBackend();
  requireModel();
  if (backend.readText) {
    return backend.readText('tensors.json');
  }
  try {
    const buffer = await backend.readFile('tensors.json');
    return new TextDecoder().decode(buffer);
  } catch (_error) {
    return null;
  }
}

export async function saveTokenizer(tokenizerJson) {
  await ensureBackend();
  requireModel();
  if (backend.writeTokenizer) {
    await backend.writeTokenizer(tokenizerJson);
    return;
  }
  const encoder = new TextEncoder();
  await backend.writeFile('tokenizer.json', encoder.encode(tokenizerJson));
}

export async function loadTokenizerFromStore() {
  await ensureBackend();
  requireModel();
  if (backend.readTokenizer) {
    return backend.readTokenizer();
  }
  if (backend.readText) {
    return backend.readText('tokenizer.json');
  }
  try {
    const buffer = await backend.readFile('tokenizer.json');
    return new TextDecoder().decode(buffer);
  } catch (_error) {
    return null;
  }
}

export async function cleanup() {
  if (backend?.cleanup) {
    await backend.cleanup();
  }
  backend = null;
  backendType = null;
  currentModelId = null;
}
