import {
  getManifest,
  getExpectedShardHash,
  getShardInfo,
  getShardCount,
  generateShardFilename,
} from '../formats/rdrr/index.js';
import {
  isOPFSAvailable,
  isIndexedDBAvailable,
  QuotaExceededError,
  checkSpaceAvailable,
} from './quota.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { normalizeModelId } from './normalize-model-id.js';
import { selectStorageBackend } from './backend-selection.js';
import {
  checkFileExistsInBackend,
  getFileSizeInBackend,
} from './shards/index.js';
import { createStorageWriteStream } from './shards/lifecycle.js';
import {
  computeHash,
  createTensorIntegrityController,
  requireManifestHashAlgorithm,
} from './shards/integrity.js';

export { getManifest } from '../formats/rdrr/index.js';
export { checkFileExistsInBackend, getFileSizeInBackend };
export {
  computeBlake3,
  computeHash,
  computeSHA256,
  createStreamingHasher,
  getHashAlgorithm,
  hexToBytes,
} from './shards/integrity.js';

let opfsPathConfigOverride = null;
let backend = null;
let backendType = null;
let currentModelId = null;

const tensorIntegrity = createTensorIntegrityController({
  readBackendFileRange,
  loadTensorsFromStore,
});

function resetTensorIntegrityCache() {
  tensorIntegrity.reset();
}

export function setOpfsPathConfig(config) {
  opfsPathConfigOverride = config;
}

export function getOpfsPathConfig() {
  return opfsPathConfigOverride ?? getRuntimeConfig().loading.opfsPath;
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
  const selected = selectStorageBackend(
    getRuntimeConfig().loading.storage.backend,
    getOpfsPathConfig()
  );
  backendType = selected.type;
  backend = selected.backend;
  await backend.init();
}

export async function openModelStore(modelId) {
  if (!backend) {
    await initStorage();
  }
  const safeName = normalizeModelId(modelId);
  currentModelId = safeName;
  resetTensorIntegrityCache();
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

async function readBackendFileRange(filename, offset = 0, length = null) {
  const start = Math.max(0, offset);
  const want = length == null ? null : Math.max(0, length);
  if (backend && typeof backend.readFileRange === 'function') {
    return backend.readFileRange(filename, start, want);
  }
  const buffer = await backend.readFile(filename);
  const view = new Uint8Array(buffer);
  const end = want == null ? view.length : Math.min(view.length, start + want);
  return view.slice(start, end).buffer;
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
    resetTensorIntegrityCache();
    await backend.writeFile(shardInfo.filename, bytes);

    if (options.verify) {
      const manifest = getManifest();
      const algorithm = requireManifestHashAlgorithm(manifest, 'shard write');
      const hash = await computeHash(bytes, algorithm);
      const expectedHash = getExpectedShardHash(shardInfo, algorithm);
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

export async function createShardWriter(shardIndex, options = {}) {
  await ensureBackend();
  requireModel();
  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }
  return createStorageWriteStream(backend, shardInfo.filename, options, resetTensorIntegrityCache);
}

export async function createConversionShardWriter(shardIndex) {
  await ensureBackend();
  requireModel();
  if (!Number.isInteger(shardIndex) || shardIndex < 0) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }
  return createStorageWriteStream(
    backend,
    generateShardFilename(shardIndex),
    {},
    resetTensorIntegrityCache
  );
}

export async function createFileWriter(filename, options = {}) {
  await ensureBackend();
  requireModel();
  if (!filename || typeof filename !== 'string') {
    throw new Error('createFileWriter requires a filename');
  }
  return createStorageWriteStream(backend, filename, options, resetTensorIntegrityCache);
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
      const expectedHash = getExpectedShardHash(shardInfo, algorithm);
      if (!expectedHash) {
        throw new Error(`Shard ${shardIndex} is missing hash in manifest`);
      }
      if (hash !== expectedHash) {
        try {
          await backend.deleteFile(shardInfo.filename);
        } catch {}
        throw new Error(
          `Hash mismatch for shard ${shardIndex}: expected ${expectedHash}, got ${hash}. Corrupt shard removed; re-import or re-download the model.`
        );
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

export async function loadShardRange(shardIndex, offset = 0, length = null, options = { verify: false }) {
  await ensureBackend();
  requireModel();

  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }

  const manifest = getManifest();
  if (options?.verify && options?.tensorId) {
    if (!manifest) {
      throw new Error('No manifest loaded');
    }
    await tensorIntegrity.verifyTensorRange(manifest, shardIndex, offset, length, options.tensorId);
  } else if (options?.verify) {
    // Generic range reads cannot be verified without hashing the full shard.
    const full = await loadShard(shardIndex, { verify: true });
    const view = new Uint8Array(full);
    const start = Math.max(0, offset);
    const end = length == null ? view.length : Math.min(view.length, start + Math.max(0, length));
    return view.slice(start, end).buffer;
  }

  const start = Math.max(0, offset);
  const want = length == null ? null : Math.max(0, length);

  try {
    return await readBackendFileRange(shardInfo.filename, start, want);
  } catch (error) {
    if (error.name === 'NotFoundError') {
      throw new Error(`Shard ${shardIndex} not found`);
    }
    throw new Error(`Failed to load shard ${shardIndex} range: ${error.message}`);
  }
}

export async function* streamShardRange(shardIndex, offset = 0, length = null, options = {}) {
  await ensureBackend();
  requireModel();

  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }

  const startRaw = Number(offset);
  const start = Number.isFinite(startRaw) ? Math.max(0, Math.floor(startRaw)) : 0;
  const wantRaw = length == null ? null : Number(length);
  const want = wantRaw == null ? null : (Number.isFinite(wantRaw) ? Math.max(0, Math.floor(wantRaw)) : 0);
  const manifest = getManifest();
  if (options?.verify && options?.tensorId) {
    if (!manifest) {
      throw new Error('No manifest loaded');
    }
    await tensorIntegrity.verifyTensorRange(manifest, shardIndex, start, want, options.tensorId);
  } else if (options?.verify) {
    const full = await loadShard(shardIndex, { verify: true });
    const view = new Uint8Array(full);
    const end = want == null ? view.length : Math.min(view.length, start + want);
    for (let at = start; at < end; at += (Number.isFinite(options.chunkBytes) && options.chunkBytes > 0 ? Math.floor(options.chunkBytes) : (4 * 1024 * 1024))) {
      yield view.slice(at, Math.min(end, at + (Number.isFinite(options.chunkBytes) && options.chunkBytes > 0 ? Math.floor(options.chunkBytes) : (4 * 1024 * 1024))));
    }
    return;
  }

  const runtime = getRuntimeConfig();
  const runtimeDefault = runtime?.loading?.storage?.backend?.streaming?.readChunkBytes ?? (4 * 1024 * 1024);
  const rawChunk = options.chunkBytes ?? runtimeDefault;
  const chunkBytes = Number.isFinite(rawChunk) && rawChunk > 0 ? Math.floor(rawChunk) : (4 * 1024 * 1024);

  // Prefer backend streaming when available.
  if (backend && typeof backend.readFileRangeStream === 'function') {
    yield* backend.readFileRangeStream(shardInfo.filename, start, want, { chunkBytes });
    return;
  }

  const end = want == null
    ? shardInfo.size
    : Math.min(shardInfo.size, start + want);
  for (let at = start; at < end; at += chunkBytes) {
    const ab = await loadShardRange(shardIndex, at, Math.min(chunkBytes, end - at), { verify: false });
    yield new Uint8Array(ab);
  }
}

export async function loadShardSync(shardIndex, offset = 0, length) {
  const ab = await loadShardRange(shardIndex, offset, length ?? null, { verify: false });
  return new Uint8Array(ab);
}

export async function shardExists(shardIndex) {
  await ensureBackend();
  requireModel();
  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) return false;
  return checkFileExistsInBackend(backend, shardInfo.filename);
}

export async function getShardStoredSize(shardIndex) {
  await ensureBackend();
  requireModel();
  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }

  return (await getFileSizeInBackend(backend, shardInfo.filename)) ?? 0;
}

export async function verifyIntegrity(options = {}) {
  const manifest = getManifest();
  if (!manifest) {
    throw new Error('No manifest loaded');
  }

  const checkHashes = options.checkHashes !== false;
  const algorithm = checkHashes
    ? requireManifestHashAlgorithm(manifest, 'integrity check')
    : null;

  const missingShards = [];
  const corruptShards = [];
  const corruptTensors = [];
  const shardCount = getShardCount();

  for (let i = 0; i < shardCount; i++) {
    const exists = await shardExists(i);
    if (!exists) {
      missingShards.push(i);
      continue;
    }

    if (checkHashes) {
      try {
        const buffer = await loadShard(i, { verify: false });
        const hash = await computeHash(buffer, algorithm);
        const shardInfo = getShardInfo(i);
        const expectedHash = getExpectedShardHash(shardInfo, algorithm);
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
  }

  if (options.checkTensorRoots === true) {
    const roots = manifest?.integrityExtensions?.blockMerkle?.roots;
    if (!roots || typeof roots !== 'object' || Array.isArray(roots)) {
      throw new Error('verifyIntegrity(checkTensorRoots=true) requires manifest.integrityExtensions.blockMerkle.roots.');
    }
    for (const tensorId of Object.keys(roots).sort()) {
      try {
        await tensorIntegrity.verifyTensorRoot(manifest, tensorId);
      } catch (_error) {
        corruptTensors.push(tensorId);
      }
    }
  }

  return {
    valid: (
      missingShards.length === 0
      && (checkHashes ? corruptShards.length === 0 : true)
      && (options.checkTensorRoots === true ? corruptTensors.length === 0 : true)
    ),
    missingShards,
    corruptShards,
    corruptTensors,
  };
}

export async function deleteShard(shardIndex) {
  await ensureBackend();
  requireModel();
  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) return false;
  resetTensorIntegrityCache();
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

export async function listFilesInStore() {
  await ensureBackend();
  requireModel();
  if (!backend.listFiles) {
    throw new Error('Storage backend does not support listing files');
  }
  return backend.listFiles();
}

export async function getFileStoredSize(filename) {
  await ensureBackend();
  requireModel();
  return getFileSizeInBackend(backend, filename);
}

export async function loadFileFromStore(filename) {
  await ensureBackend();
  requireModel();
  if (!filename || typeof filename !== 'string') {
    throw new Error('loadFileFromStore requires a filename');
  }
  return backend.readFile(filename);
}

export async function loadFileRangeFromStore(filename, offset = 0, length = null) {
  await ensureBackend();
  requireModel();
  if (!filename || typeof filename !== 'string') {
    throw new Error('loadFileRangeFromStore requires a filename');
  }
  const start = Number.isFinite(Number(offset)) ? Math.max(0, Math.floor(Number(offset))) : 0;
  const want = length == null
    ? null
    : (Number.isFinite(Number(length)) ? Math.max(0, Math.floor(Number(length))) : 0);
  if (backend && typeof backend.readFileRange === 'function') {
    return backend.readFileRange(filename, start, want);
  }
  const buffer = await backend.readFile(filename);
  const bytes = new Uint8Array(buffer);
  const end = want == null ? bytes.byteLength : Math.min(bytes.byteLength, start + want);
  return bytes.slice(start, end).buffer;
}

export function streamFileFromStore(filename, options = {}) {
  if (!backend || !filename || typeof filename !== 'string') {
    return null;
  }
  const runtime = getRuntimeConfig();
  const runtimeDefault = runtime?.loading?.storage?.backend?.streaming?.readChunkBytes ?? (4 * 1024 * 1024);
  const raw = options.chunkBytes ?? runtimeDefault;
  const chunkBytes = Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : (4 * 1024 * 1024);
  const offset = Number.isFinite(Number(options.offset)) ? Math.max(0, Math.floor(Number(options.offset))) : 0;
  const length = options.length == null
    ? null
    : (Number.isFinite(Number(options.length)) ? Math.max(0, Math.floor(Number(options.length))) : 0);
  if (typeof backend.readFileRangeStream === 'function') {
    return backend.readFileRangeStream(filename, offset, length, { chunkBytes });
  }
  return null;
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

  const previousModelId = currentModelId;
  try {
    await backend.openModel(safeName, { create: false });
    currentModelId = safeName;
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
  } finally {
    currentModelId = previousModelId;
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
  resetTensorIntegrityCache();
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

export async function saveTensorsToStore(tensorsJson) {
  await ensureBackend();
  requireModel();
  resetTensorIntegrityCache();
  const encoder = new TextEncoder();
  const payload = encoder.encode(tensorsJson);
  if (backend.writeText) {
    await backend.writeText('tensors.json', tensorsJson);
    return;
  }
  await backend.writeFile('tensors.json', payload);
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

export async function saveTokenizerModel(tokenizerModel) {
  await ensureBackend();
  requireModel();
  const data = tokenizerModel instanceof Uint8Array
    ? tokenizerModel
    : new Uint8Array(tokenizerModel);
  await backend.writeFile('tokenizer.model', data);
}

export async function loadTokenizerModelFromStore() {
  await ensureBackend();
  requireModel();
  try {
    return await backend.readFile('tokenizer.model');
  } catch (error) {
    if (error?.name === 'NotFoundError' || error?.message?.includes('not found')) {
      return null;
    }
    throw error;
  }
}

export async function saveAuxFile(filename, data) {
  await ensureBackend();
  requireModel();
  if (!filename || typeof filename !== 'string') {
    throw new Error('saveAuxFile requires a filename');
  }
  const bytes = typeof data === 'string'
    ? new TextEncoder().encode(data)
    : data instanceof ArrayBuffer
      ? new Uint8Array(data)
      : data instanceof Uint8Array
        ? data
        : null;
  if (!bytes) {
    throw new Error('saveAuxFile requires string, ArrayBuffer, or Uint8Array data');
  }
  resetTensorIntegrityCache();
  await backend.writeFile(filename, bytes);
}

export async function loadAuxFile(filename) {
  await ensureBackend();
  requireModel();
  if (!filename || typeof filename !== 'string') {
    throw new Error('loadAuxFile requires a filename');
  }
  try {
    return await backend.readFile(filename);
  } catch (error) {
    if (error?.name === 'NotFoundError') {
      return null;
    }
    return null;
  }
}

export async function deleteFileFromStore(filename) {
  await ensureBackend();
  requireModel();
  if (!filename || typeof filename !== 'string') {
    throw new Error('deleteFileFromStore requires a filename');
  }
  resetTensorIntegrityCache();
  return backend.deleteFile(filename);
}

export async function loadAuxText(filename) {
  await ensureBackend();
  requireModel();
  if (!filename || typeof filename !== 'string') {
    throw new Error('loadAuxText requires a filename');
  }
  if (backend.readText) {
    return backend.readText(filename);
  }
  try {
    const buffer = await backend.readFile(filename);
    return new TextDecoder().decode(buffer);
  } catch (error) {
    if (error?.name === 'NotFoundError') {
      return null;
    }
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
