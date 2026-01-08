/**
 * shard-manager.ts - OPFS Shard Management with BLAKE3 Verification
 *
 * Handles:
 * - OPFS directory structure for model shards
 * - Shard read/write with 4KB alignment for optimal performance
 * - BLAKE3 hash verification for integrity checking
 * - FileSystemSyncAccessHandle for synchronous reads (in workers)
 *
 * @module storage/shard-manager
 */

import {
  getManifest,
  getShardInfo,
  getShardCount,
} from './rdrr-format.js';
import { isOPFSAvailable, QuotaExceededError, checkSpaceAvailable } from './quota.js';
import { log } from '../debug/index.js';
import { getRuntimeConfig } from '../config/runtime.js';

// Re-export for consumers that import from shard-manager
export { getManifest } from './rdrr-format.js';

// ============================================================================
// Constants
// ============================================================================

/**
 * @returns {number}
 */
function getAlignmentBytes() {
  return getRuntimeConfig().storage.alignment.bufferAlignmentBytes;
}

// Storage config - can be overridden via setOpfsPathConfig()
/** @type {import('../config/schema/loading.schema.js').OpfsPathConfigSchema | null} */
let opfsPathConfigOverride = null;

// ============================================================================
// Module State
// ============================================================================

/** @type {FileSystemDirectoryHandle | null} */
let rootDir = null;
/** @type {FileSystemDirectoryHandle | null} */
let modelsDir = null;
/** @type {FileSystemDirectoryHandle | null} */
let currentModelDir = null;
/** @type {{ hash(data: Uint8Array): Promise<Uint8Array>; createHasher(): { update(data: Uint8Array): void; finalize(): Promise<Uint8Array> } } | null} */
let blake3Module = null;
/** @type {import('./rdrr-format.js').HashAlgorithm | null} */
let hashAlgorithm = null;

/**
 * Set OPFS path configuration
 * @param {import('../config/schema/loading.schema.js').OpfsPathConfigSchema} config
 * @returns {void}
 */
export function setOpfsPathConfig(config) {
  opfsPathConfigOverride = config;
}

/**
 * Get current OPFS path configuration
 * @returns {import('../config/schema/loading.schema.js').OpfsPathConfigSchema}
 */
export function getOpfsPathConfig() {
  return opfsPathConfigOverride ?? getRuntimeConfig().loading.opfsPath;
}

// ============================================================================
// BLAKE3/SHA256 Hashing
// ============================================================================

/**
 * Initializes the BLAKE3 hashing module
 * Uses the BLAKE3 WASM implementation, falls back to SHA-256
 * @param {import('./rdrr-format.js').HashAlgorithm | null} [requiredAlgorithm]
 * @returns {Promise<void>}
 */
async function initBlake3(requiredAlgorithm = null) {
  if (blake3Module && hashAlgorithm) return;

  // Try to load BLAKE3 WASM module
  try {
    // Dynamic import of blake3 module (should be bundled or loaded separately)
    const globalBlake3 = /** @type {{ blake3?: { hash(data: Uint8Array): Promise<Uint8Array>; createHasher(): { update(data: Uint8Array): void; finalize(): Promise<Uint8Array> } } }} */ (globalThis).blake3;
    if (globalBlake3 !== undefined) {
      blake3Module = globalBlake3;
      hashAlgorithm = 'blake3';
      return;
    }
  } catch (e) {
    log.warn('ShardManager', `BLAKE3 WASM module not available: ${/** @type {Error} */ (e).message}`);
  }

  // If BLAKE3 is explicitly required and not available, fail
  if (requiredAlgorithm === 'blake3') {
    throw new Error(
      'BLAKE3 required by manifest but not available. ' +
      'Install blake3 WASM module or re-convert model with SHA-256.'
    );
  }

  // Fallback to SHA-256
  hashAlgorithm = 'sha256';
  blake3Module = {
    hash: async (/** @type {Uint8Array} */ data) => {
      const hashBuffer = await crypto.subtle.digest('SHA-256', /** @type {ArrayBuffer} */ (data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)));
      return new Uint8Array(hashBuffer);
    },
    createHasher: () => {
      /** @type {Uint8Array[]} */
      const chunks = [];
      return {
        update: (/** @type {Uint8Array} */ data) => {
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

/**
 * Get the current hash algorithm in use
 * @returns {import('./rdrr-format.js').HashAlgorithm | null}
 */
export function getHashAlgorithm() {
  return hashAlgorithm;
}

/**
 * Converts Uint8Array to hex string
 * @param {Uint8Array} bytes
 * @returns {string}
 */
function bytesToHex(bytes) {
  return Array.from(bytes)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Converts hex string to Uint8Array
 * @param {string} hex
 * @returns {Uint8Array}
 */
export function hexToBytes(hex) {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(hex.substr(i * 2, 2), 16);
  }
  return bytes;
}

/**
 * Computes BLAKE3 hash of data
 * @param {Uint8Array | ArrayBuffer} data
 * @returns {Promise<string>}
 */
export async function computeBlake3(data) {
  await initBlake3('blake3');

  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  const hash = await /** @type {NonNullable<typeof blake3Module>} */ (blake3Module).hash(bytes);
  return bytesToHex(hash);
}

/**
 * Computes SHA-256 hash of data
 * @param {Uint8Array | ArrayBuffer} data
 * @returns {Promise<string>}
 */
export async function computeSHA256(data) {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  const buffer = /** @type {ArrayBuffer} */ (bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  return bytesToHex(new Uint8Array(hashBuffer));
}

/**
 * Computes hash using specified algorithm
 * @param {Uint8Array | ArrayBuffer} data
 * @param {import('./rdrr-format.js').HashAlgorithm} [algorithm]
 * @returns {Promise<string>}
 */
export async function computeHash(
  data,
  algorithm = 'blake3'
) {
  if (algorithm === 'sha256') {
    return computeSHA256(data);
  }
  return computeBlake3(data);
}

/**
 * Creates a streaming BLAKE3 hasher for large data
 * @returns {Promise<{ update(data: Uint8Array): void; finalize(): Promise<Uint8Array> }>}
 */
export async function createStreamingHasher() {
  await initBlake3();
  return /** @type {NonNullable<typeof blake3Module>} */ (blake3Module).createHasher();
}

// ============================================================================
// OPFS Operations
// ============================================================================

/**
 * Initializes the OPFS directory structure
 * @returns {Promise<void>}
 */
export async function initOPFS() {
  if (!isOPFSAvailable()) {
    throw new Error('OPFS not available in this browser');
  }

  try {
    rootDir = await navigator.storage.getDirectory();
    const { opfsRootDir } = getOpfsPathConfig();
    modelsDir = await rootDir.getDirectoryHandle(opfsRootDir, { create: true });
  } catch (error) {
    throw new Error(`Failed to initialize OPFS: ${/** @type {Error} */ (error).message}`);
  }
}

/**
 * Opens a model directory, creating it if necessary
 * @param {string} modelId
 * @returns {Promise<FileSystemDirectoryHandle>}
 */
export async function openModelDirectory(modelId) {
  if (!modelsDir) {
    await initOPFS();
  }

  // Sanitize modelId for filesystem
  const safeName = modelId.replace(/[^a-zA-Z0-9_-]/g, '_');
  currentModelDir = await /** @type {NonNullable<typeof modelsDir>} */ (modelsDir).getDirectoryHandle(safeName, { create: true });
  return currentModelDir;
}

/**
 * Gets the current model directory handle
 * @returns {FileSystemDirectoryHandle | null}
 */
export function getCurrentModelDirectory() {
  return currentModelDir;
}

/**
 * Writes a shard to OPFS
 * @param {number} shardIndex
 * @param {ArrayBuffer} data
 * @param {import('./shard-manager.js').ShardWriteOptions} [options]
 * @returns {Promise<import('./shard-manager.js').ShardWriteResult>}
 */
export async function writeShard(
  shardIndex,
  data,
  options = { verify: true }
) {
  if (!currentModelDir) {
    throw new Error('No model directory open. Call openModelDirectory first.');
  }

  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }

  // Check available space before writing
  const spaceCheck = await checkSpaceAvailable(data.byteLength);
  if (!spaceCheck.hasSpace) {
    throw new QuotaExceededError(data.byteLength, spaceCheck.info.available);
  }

  try {
    // Get or create the shard file
    const fileHandle = await currentModelDir.getFileHandle(shardInfo.filename, { create: true });

    // Use writable stream for efficient writes
    const writable = await fileHandle.createWritable();

    // Write data with proper alignment consideration
    const alignment = getAlignmentBytes();
    const alignedSize = Math.ceil(data.byteLength / alignment) * alignment;
    if (alignedSize !== data.byteLength) {
      // Pad to alignment boundary (optional, depends on requirements)
      await writable.write(data);
    } else {
      await writable.write(data);
    }

    await writable.close();

    // Verify hash if requested
    if (options.verify) {
      const manifest = getManifest();
      const algorithm = manifest?.hashAlgorithm || 'blake3';
      const hash = await computeHash(data, algorithm);
      const expectedHash = shardInfo.hash || shardInfo.blake3;

      if (hash !== expectedHash) {
        // Delete the corrupted shard
        await currentModelDir.removeEntry(shardInfo.filename);
        throw new Error(`Hash mismatch for shard ${shardIndex}: expected ${expectedHash}, got ${hash}`);
      }
      return { success: true, hash };
    }

    return { success: true, hash: null };
  } catch (error) {
    if (error instanceof QuotaExceededError) throw error;
    throw new Error(`Failed to write shard ${shardIndex}: ${/** @type {Error} */ (error).message}`);
  }
}

/**
 * Reads a shard from OPFS
 * @param {number} shardIndex
 * @param {import('./shard-manager.js').ShardReadOptions} [options]
 * @returns {Promise<ArrayBuffer>}
 */
export async function loadShard(
  shardIndex,
  options = { verify: false }
) {
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

    // Verify hash if requested
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
    if (/** @type {Error} */ (error).name === 'NotFoundError') {
      throw new Error(`Shard ${shardIndex} not found`);
    }
    throw new Error(`Failed to load shard ${shardIndex}: ${/** @type {Error} */ (error).message}`);
  }
}

/**
 * Reads a shard using synchronous access (for Worker threads)
 * Provides better performance for repeated reads
 * @param {number} shardIndex
 * @param {number} [offset]
 * @param {number} [length]
 * @returns {Promise<Uint8Array>}
 */
export async function loadShardSync(
  shardIndex,
  offset = 0,
  length
) {
  if (!currentModelDir) {
    throw new Error('No model directory open. Call openModelDirectory first.');
  }

  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }

  // Align offset to 4KB boundary for optimal reads
  const alignment = getAlignmentBytes();
  const alignedOffset = Math.floor(offset / alignment) * alignment;
  const offsetDelta = offset - alignedOffset;

  const readLength = length ?? (shardInfo.size - offset);
  const alignedLength = Math.ceil((readLength + offsetDelta) / alignment) * alignment;

  try {
    const fileHandle = await currentModelDir.getFileHandle(shardInfo.filename);
    const syncHandle = await /** @type {FileSystemFileHandle & { createSyncAccessHandle(): Promise<FileSystemSyncAccessHandle> }} */ (fileHandle).createSyncAccessHandle();

    try {
      const buffer = new Uint8Array(alignedLength);
      const bytesRead = syncHandle.read(buffer, { at: alignedOffset });

      // Return only the requested portion
      if (offsetDelta > 0 || readLength !== alignedLength) {
        return buffer.slice(offsetDelta, offsetDelta + readLength);
      }
      return buffer.slice(0, bytesRead);
    } finally {
      syncHandle.close();
    }
  } catch (error) {
    if (/** @type {Error} */ (error).name === 'NotFoundError') {
      throw new Error(`Shard ${shardIndex} not found`);
    }
    // If sync access not supported, fall back to async
    if (/** @type {Error} */ (error).name === 'NotSupportedError') {
      log.warn('ShardManager', 'Sync access not supported, falling back to async read');
      const buffer = await loadShard(shardIndex);
      return new Uint8Array(buffer, offset, length);
    }
    throw new Error(`Failed to sync-load shard ${shardIndex}: ${/** @type {Error} */ (error).message}`);
  }
}

/**
 * Checks if a shard exists in OPFS
 * @param {number} shardIndex
 * @returns {Promise<boolean>}
 */
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

/**
 * Verifies the integrity of all shards
 * @returns {Promise<import('./shard-manager.js').IntegrityResult>}
 */
export async function verifyIntegrity() {
  const manifest = getManifest();
  if (!manifest) {
    throw new Error('No manifest loaded');
  }

  if (!currentModelDir) {
    throw new Error('No model directory open');
  }

  // Get hash algorithm from manifest (default to blake3 for backwards compatibility)
  const algorithm = manifest.hashAlgorithm || 'blake3';

  /** @type {number[]} */
  const missingShards = [];
  /** @type {number[]} */
  const corruptShards = [];
  const shardCount = getShardCount();

  for (let i = 0; i < shardCount; i++) {
    const exists = await shardExists(i);
    if (!exists) {
      missingShards.push(i);
      continue;
    }

    // Verify hash
    try {
      const buffer = await loadShard(i, { verify: false });
      const hash = await computeHash(buffer, algorithm);
      const shardInfo = getShardInfo(i);

      // Support both 'blake3' and 'hash' field names
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

/**
 * Deletes a shard from OPFS
 * @param {number} shardIndex
 * @returns {Promise<boolean>}
 */
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

/**
 * Deletes an entire model from OPFS
 * @param {string} modelId
 * @returns {Promise<boolean>}
 */
export async function deleteModel(modelId) {
  if (!modelsDir) {
    await initOPFS();
  }

  const safeName = modelId.replace(/[^a-zA-Z0-9_-]/g, '_');

  try {
    await /** @type {NonNullable<typeof modelsDir>} */ (modelsDir).removeEntry(safeName, { recursive: true });

    // Clear current model dir if it was this model
    if (currentModelDir) {
      try {
        // Check if current dir is the deleted one
        await currentModelDir.getFileHandle('.test', { create: true })
          .then((/** @type {FileSystemFileHandle} */ _h) => /** @type {NonNullable<typeof currentModelDir>} */ (currentModelDir).removeEntry('.test'))
          .catch(() => {});
      } catch {
        currentModelDir = null;
      }
    }

    return true;
  } catch (error) {
    if (/** @type {Error} */ (error).name === 'NotFoundError') {
      return true; // Already deleted
    }
    return false;
  }
}

/**
 * Lists all models stored in OPFS
 * @returns {Promise<string[]>}
 */
export async function listModels() {
  if (!modelsDir) {
    try {
      await initOPFS();
    } catch {
      return [];
    }
  }

  /** @type {string[]} */
  const models = [];
  const entries = /** @type {{ entries(): AsyncIterable<[string, FileSystemHandle]> }} */ (/** @type {unknown} */ (modelsDir)).entries();
  for await (const [name, handle] of entries) {
    if (handle.kind === 'directory') {
      models.push(name);
    }
  }

  return models;
}

/**
 * Gets information about a stored model
 * @param {string} modelId
 * @returns {Promise<import('./shard-manager.js').ModelInfo>}
 */
export async function getModelInfo(modelId) {
  if (!modelsDir) {
    await initOPFS();
  }

  const safeName = modelId.replace(/[^a-zA-Z0-9_-]/g, '_');

  try {
    const modelDir = await /** @type {NonNullable<typeof modelsDir>} */ (modelsDir).getDirectoryHandle(safeName);
    let shardCount = 0;
    let totalSize = 0;
    let hasManifest = false;

    const modelEntries = /** @type {{ entries(): AsyncIterable<[string, FileSystemHandle]> }} */ (/** @type {unknown} */ (modelDir)).entries();
    for await (const [name, handle] of modelEntries) {
      if (handle.kind === 'file') {
        if (name === 'manifest.json') {
          hasManifest = true;
        } else if (name.startsWith('shard_') && name.endsWith('.bin')) {
          shardCount++;
          const file = await /** @type {FileSystemFileHandle} */ (handle).getFile();
          totalSize += file.size;
        }
      }
    }

    return { exists: true, shardCount, totalSize, hasManifest };
  } catch (_error) {
    return { exists: false, shardCount: 0, totalSize: 0, hasManifest: false };
  }
}

/**
 * Checks if a model exists in OPFS
 * @param {string} modelId
 * @returns {Promise<boolean>}
 */
export async function modelExists(modelId) {
  const info = await getModelInfo(modelId);
  return info.exists && info.hasManifest;
}

/**
 * Saves the manifest to OPFS
 * @param {string} manifestJson
 * @returns {Promise<void>}
 */
export async function saveManifest(manifestJson) {
  if (!currentModelDir) {
    throw new Error('No model directory open');
  }

  const fileHandle = await currentModelDir.getFileHandle('manifest.json', { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(manifestJson);
  await writable.close();
}

/**
 * Loads the manifest from OPFS
 * @returns {Promise<string>}
 */
export async function loadManifestFromOPFS() {
  if (!currentModelDir) {
    throw new Error('No model directory open');
  }

  try {
    const fileHandle = await currentModelDir.getFileHandle('manifest.json');
    const file = await fileHandle.getFile();
    return await file.text();
  } catch (error) {
    if (/** @type {Error} */ (error).name === 'NotFoundError') {
      throw new Error('Manifest not found');
    }
    throw error;
  }
}

/**
 * Loads the tensors.json from OPFS (v1 format)
 * @returns {Promise<string | null>} Tensors JSON string or null if not found
 */
export async function loadTensorsFromOPFS() {
  if (!currentModelDir) {
    throw new Error('No model directory open');
  }

  try {
    const fileHandle = await currentModelDir.getFileHandle('tensors.json');
    const file = await fileHandle.getFile();
    return await file.text();
  } catch (error) {
    if (/** @type {Error} */ (error).name === 'NotFoundError') {
      return null; // v1 tensors.json is optional (may be inline in manifest)
    }
    throw error;
  }
}

/**
 * Saves the tokenizer.json to OPFS
 * @param {string} tokenizerJson
 * @returns {Promise<void>}
 */
export async function saveTokenizer(tokenizerJson) {
  if (!currentModelDir) {
    throw new Error('No model directory open');
  }

  const fileHandle = await currentModelDir.getFileHandle('tokenizer.json', { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(tokenizerJson);
  await writable.close();
}

/**
 * Loads the tokenizer.json from OPFS
 * @returns {Promise<string | null>}
 */
export async function loadTokenizerFromOPFS() {
  if (!currentModelDir) {
    throw new Error('No model directory open');
  }

  try {
    const fileHandle = await currentModelDir.getFileHandle('tokenizer.json');
    const file = await fileHandle.getFile();
    return await file.text();
  } catch (error) {
    if (/** @type {Error} */ (error).name === 'NotFoundError') {
      return null; // Tokenizer not bundled, will fall back to HuggingFace
    }
    throw error;
  }
}

/**
 * Cleans up module state (useful for testing)
 * @returns {void}
 */
export function cleanup() {
  rootDir = null;
  modelsDir = null;
  currentModelDir = null;
}

// ============================================================================
// OpfsShardStore Class Implementation
// ============================================================================

/**
 * OPFS-backed shard store implementing the ShardStore interface
 */
export class OpfsShardStore {
  /** @type {string} */
  #modelId;
  /** @type {boolean} */
  #initialized = false;

  /**
   * @param {string} modelId
   */
  constructor(modelId) {
    this.#modelId = modelId;
  }

  /**
   * @returns {Promise<void>}
   */
  async #ensureInitialized() {
    if (this.#initialized) return;
    await openModelDirectory(this.#modelId);
    this.#initialized = true;
  }

  /**
   * @param {number} shardIndex
   * @param {number} offset
   * @param {number} length
   * @returns {Promise<Uint8Array>}
   */
  async read(shardIndex, offset, length) {
    await this.#ensureInitialized();
    return loadShardSync(shardIndex, offset, length);
  }

  /**
   * @param {number} shardIndex
   * @param {Uint8Array} data
   * @returns {Promise<void>}
   */
  async write(shardIndex, data) {
    await this.#ensureInitialized();
    await writeShard(shardIndex, /** @type {ArrayBuffer} */ (data.buffer), { verify: true });
  }

  /**
   * @param {number} shardIndex
   * @returns {Promise<boolean>}
   */
  async exists(shardIndex) {
    await this.#ensureInitialized();
    return shardExists(shardIndex);
  }

  /**
   * @param {number} shardIndex
   * @returns {Promise<void>}
   */
  async delete(shardIndex) {
    await this.#ensureInitialized();
    await deleteShard(shardIndex);
  }

  /**
   * @returns {Promise<number[]>}
   */
  async list() {
    await this.#ensureInitialized();
    const shardCount = getShardCount();
    /** @type {number[]} */
    const existing = [];
    for (let i = 0; i < shardCount; i++) {
      if (await this.exists(i)) {
        existing.push(i);
      }
    }
    return existing;
  }
}
