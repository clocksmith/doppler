import {
  getManifest,
  getShardInfo,
  getShardCount,
} from './rdrr-format.js';
import { isOPFSAvailable, QuotaExceededError, checkSpaceAvailable } from './quota.js';
import { log } from '../debug/index.js';
import { getRuntimeConfig } from '../config/runtime.js';

export { getManifest } from './rdrr-format.js';

function getAlignmentBytes() {
  return getRuntimeConfig().loading.storage.alignment.bufferAlignmentBytes;
}

let opfsPathConfigOverride = null;

let rootDir = null;
let modelsDir = null;
let currentModelDir = null;
let blake3Module = null;
let hashAlgorithm = null;

export function setOpfsPathConfig(config) {
  opfsPathConfigOverride = config;
}

export function getOpfsPathConfig() {
  return opfsPathConfigOverride ?? getRuntimeConfig().loading.opfsPath;
}

async function initBlake3(requiredAlgorithm = null) {
  if (blake3Module && hashAlgorithm) return;

  try {
    const globalBlake3 = globalThis.blake3;
    if (globalBlake3 !== undefined) {
      blake3Module = globalBlake3;
      hashAlgorithm = 'blake3';
      return;
    }
  } catch (e) {
    log.warn('ShardManager', `BLAKE3 WASM module not available: ${e.message}`);
  }

  if (requiredAlgorithm === 'blake3') {
    throw new Error(
      'BLAKE3 required by manifest but not available. ' +
      'Install blake3 WASM module or re-convert model with SHA-256.'
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

export async function computeHash(data, algorithm = 'blake3') {
  if (algorithm === 'sha256') {
    return computeSHA256(data);
  }
  return computeBlake3(data);
}

export async function createStreamingHasher() {
  await initBlake3();
  return blake3Module.createHasher();
}

export async function initOPFS() {
  if (!isOPFSAvailable()) {
    throw new Error('OPFS not available in this browser');
  }

  try {
    rootDir = await navigator.storage.getDirectory();
    const { opfsRootDir } = getOpfsPathConfig();
    modelsDir = await rootDir.getDirectoryHandle(opfsRootDir, { create: true });
  } catch (error) {
    throw new Error(`Failed to initialize OPFS: ${error.message}`);
  }
}

export async function openModelDirectory(modelId) {
  if (!modelsDir) {
    await initOPFS();
  }

  const safeName = modelId.replace(/[^a-zA-Z0-9_-]/g, '_');
  currentModelDir = await modelsDir.getDirectoryHandle(safeName, { create: true });
  return currentModelDir;
}

export function getCurrentModelDirectory() {
  return currentModelDir;
}

export async function writeShard(shardIndex, data, options = { verify: true }) {
  if (!currentModelDir) {
    throw new Error('No model directory open. Call openModelDirectory first.');
  }

  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }

  const spaceCheck = await checkSpaceAvailable(data.byteLength);
  if (!spaceCheck.hasSpace) {
    throw new QuotaExceededError(data.byteLength, spaceCheck.info.available);
  }

  try {
    const fileHandle = await currentModelDir.getFileHandle(shardInfo.filename, { create: true });
    const writable = await fileHandle.createWritable();

    const alignment = getAlignmentBytes();
    const alignedSize = Math.ceil(data.byteLength / alignment) * alignment;
    if (alignedSize !== data.byteLength) {
      await writable.write(data);
    } else {
      await writable.write(data);
    }

    await writable.close();

    if (options.verify) {
      const manifest = getManifest();
      const algorithm = manifest?.hashAlgorithm || 'blake3';
      const hash = await computeHash(data, algorithm);
      const expectedHash = shardInfo.hash || shardInfo.blake3;

      if (hash !== expectedHash) {
        await currentModelDir.removeEntry(shardInfo.filename);
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

export async function loadShard(shardIndex, options = { verify: false }) {
  if (!currentModelDir) {
    throw new Error('No model directory open. Call openModelDirectory first.');
  }

  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }

  try {
    const fileHandle = await currentModelDir.getFileHandle(shardInfo.filename);
    const file = await fileHandle.getFile();
    const buffer = await file.arrayBuffer();

    if (options.verify) {
      const manifest = getManifest();
      const algorithm = manifest?.hashAlgorithm || 'blake3';
      const hash = await computeHash(buffer, algorithm);
      const expectedHash = shardInfo.hash || shardInfo.blake3;

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
  if (!currentModelDir) {
    throw new Error('No model directory open. Call openModelDirectory first.');
  }

  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }

  const alignment = getAlignmentBytes();
  const alignedOffset = Math.floor(offset / alignment) * alignment;
  const offsetDelta = offset - alignedOffset;

  const readLength = length ?? (shardInfo.size - offset);
  const alignedLength = Math.ceil((readLength + offsetDelta) / alignment) * alignment;

  try {
    const fileHandle = await currentModelDir.getFileHandle(shardInfo.filename);
    const syncHandle = await fileHandle.createSyncAccessHandle();

    try {
      const buffer = new Uint8Array(alignedLength);
      const bytesRead = syncHandle.read(buffer, { at: alignedOffset });

      if (offsetDelta > 0 || readLength !== alignedLength) {
        return buffer.slice(offsetDelta, offsetDelta + readLength);
      }
      return buffer.slice(0, bytesRead);
    } finally {
      syncHandle.close();
    }
  } catch (error) {
    if (error.name === 'NotFoundError') {
      throw new Error(`Shard ${shardIndex} not found`);
    }
    if (error.name === 'NotSupportedError') {
      log.warn('ShardManager', 'Sync access not supported, falling back to async read');
      const buffer = await loadShard(shardIndex);
      return new Uint8Array(buffer, offset, length);
    }
    throw new Error(`Failed to sync-load shard ${shardIndex}: ${error.message}`);
  }
}

export async function shardExists(shardIndex) {
  if (!currentModelDir) return false;

  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) return false;

  try {
    await currentModelDir.getFileHandle(shardInfo.filename);
    return true;
  } catch (_error) {
    return false;
  }
}

export async function verifyIntegrity() {
  const manifest = getManifest();
  if (!manifest) {
    throw new Error('No manifest loaded');
  }

  if (!currentModelDir) {
    throw new Error('No model directory open');
  }

  const algorithm = manifest.hashAlgorithm || 'blake3';

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

      const expectedHash = shardInfo?.hash || shardInfo?.blake3;

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
  if (!currentModelDir) return false;

  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) return false;

  try {
    await currentModelDir.removeEntry(shardInfo.filename);
    return true;
  } catch (_error) {
    return false;
  }
}

export async function deleteModel(modelId) {
  if (!modelsDir) {
    await initOPFS();
  }

  const safeName = modelId.replace(/[^a-zA-Z0-9_-]/g, '_');

  try {
    await modelsDir.removeEntry(safeName, { recursive: true });

    if (currentModelDir) {
      try {
        await currentModelDir.getFileHandle('.test', { create: true })
          .then((_h) => currentModelDir.removeEntry('.test'))
          .catch(() => {});
      } catch {
        currentModelDir = null;
      }
    }

    return true;
  } catch (error) {
    if (error.name === 'NotFoundError') {
      return true;
    }
    return false;
  }
}

export async function listModels() {
  if (!modelsDir) {
    try {
      await initOPFS();
    } catch {
      return [];
    }
  }

  const models = [];
  const entries = modelsDir.entries();
  for await (const [name, handle] of entries) {
    if (handle.kind === 'directory') {
      models.push(name);
    }
  }

  return models;
}

export async function getModelInfo(modelId) {
  if (!modelsDir) {
    await initOPFS();
  }

  const safeName = modelId.replace(/[^a-zA-Z0-9_-]/g, '_');

  try {
    const modelDir = await modelsDir.getDirectoryHandle(safeName);
    let shardCount = 0;
    let totalSize = 0;
    let hasManifest = false;

    const modelEntries = modelDir.entries();
    for await (const [name, handle] of modelEntries) {
      if (handle.kind === 'file') {
        if (name === 'manifest.json') {
          hasManifest = true;
        } else if (name.startsWith('shard_') && name.endsWith('.bin')) {
          shardCount++;
          const file = await handle.getFile();
          totalSize += file.size;
        }
      }
    }

    return { exists: true, shardCount, totalSize, hasManifest };
  } catch (_error) {
    return { exists: false, shardCount: 0, totalSize: 0, hasManifest: false };
  }
}

export async function modelExists(modelId) {
  const info = await getModelInfo(modelId);
  return info.exists && info.hasManifest;
}

export async function saveManifest(manifestJson) {
  if (!currentModelDir) {
    throw new Error('No model directory open');
  }

  const fileHandle = await currentModelDir.getFileHandle('manifest.json', { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(manifestJson);
  await writable.close();
}

export async function loadManifestFromOPFS() {
  if (!currentModelDir) {
    throw new Error('No model directory open');
  }

  try {
    const fileHandle = await currentModelDir.getFileHandle('manifest.json');
    const file = await fileHandle.getFile();
    return await file.text();
  } catch (error) {
    if (error.name === 'NotFoundError') {
      throw new Error('Manifest not found');
    }
    throw error;
  }
}

export async function loadTensorsFromOPFS() {
  if (!currentModelDir) {
    throw new Error('No model directory open');
  }

  try {
    const fileHandle = await currentModelDir.getFileHandle('tensors.json');
    const file = await fileHandle.getFile();
    return await file.text();
  } catch (error) {
    if (error.name === 'NotFoundError') {
      return null;
    }
    throw error;
  }
}

export async function saveTokenizer(tokenizerJson) {
  if (!currentModelDir) {
    throw new Error('No model directory open');
  }

  const fileHandle = await currentModelDir.getFileHandle('tokenizer.json', { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(tokenizerJson);
  await writable.close();
}

export async function loadTokenizerFromOPFS() {
  if (!currentModelDir) {
    throw new Error('No model directory open');
  }

  try {
    const fileHandle = await currentModelDir.getFileHandle('tokenizer.json');
    const file = await fileHandle.getFile();
    return await file.text();
  } catch (error) {
    if (error.name === 'NotFoundError') {
      return null;
    }
    throw error;
  }
}

export function cleanup() {
  rootDir = null;
  modelsDir = null;
  currentModelDir = null;
}

export class OpfsShardStore {
  #modelId;
  #initialized = false;

  constructor(modelId) {
    this.#modelId = modelId;
  }

  async #ensureInitialized() {
    if (this.#initialized) return;
    await openModelDirectory(this.#modelId);
    this.#initialized = true;
  }

  async read(shardIndex, offset, length) {
    await this.#ensureInitialized();
    return loadShardSync(shardIndex, offset, length);
  }

  async write(shardIndex, data) {
    await this.#ensureInitialized();
    await writeShard(shardIndex, data.buffer, { verify: true });
  }

  async exists(shardIndex) {
    await this.#ensureInitialized();
    return shardExists(shardIndex);
  }

  async delete(shardIndex) {
    await this.#ensureInitialized();
    await deleteShard(shardIndex);
  }

  async list() {
    await this.#ensureInitialized();
    const shardCount = getShardCount();
    const existing = [];
    for (let i = 0; i < shardCount; i++) {
      if (await this.exists(i)) {
        existing.push(i);
      }
    }
    return existing;
  }
}
