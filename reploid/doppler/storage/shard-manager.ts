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
  type ShardInfo,
  type RDRRManifest,
  type HashAlgorithm,
} from './rdrr-format.js';
import { isOPFSAvailable, QuotaExceededError, checkSpaceAvailable } from './quota.js';

// Re-export for consumers that import from shard-manager
export { getManifest } from './rdrr-format.js';
export type { ShardInfo, RDRRManifest, HashAlgorithm };

// ============================================================================
// Types and Interfaces
// ============================================================================

/**
 * ShardStore interface for abstracting shard storage backends
 */
export interface ShardStore {
  read(shardIndex: number, offset: number, length: number): Promise<Uint8Array>;
  write(shardIndex: number, data: Uint8Array): Promise<void>;
  exists(shardIndex: number): Promise<boolean>;
  delete(shardIndex: number): Promise<void>;
  list(): Promise<number[]>;
}

/**
 * Options for reading shards
 */
export interface ShardReadOptions {
  /** Verify hash after read */
  verify?: boolean;
  /** Abort signal for cancellation */
  signal?: AbortSignal;
  /** Progress callback */
  onProgress?: (loaded: number, total: number) => void;
}

/**
 * Options for writing shards
 */
export interface ShardWriteOptions {
  /** Verify hash after write */
  verify?: boolean;
  /** Progress callback */
  onProgress?: (written: number, total: number) => void;
}

/**
 * Result of a shard write operation
 */
export interface ShardWriteResult {
  success: boolean;
  hash: string | null;
}

/**
 * Result of integrity verification
 */
export interface IntegrityResult {
  valid: boolean;
  missingShards: number[];
  corruptShards: number[];
}

/**
 * Model information from OPFS
 */
export interface ModelInfo {
  exists: boolean;
  shardCount: number;
  totalSize: number;
  hasManifest: boolean;
}

/**
 * BLAKE3 hasher interface
 */
interface Blake3Hasher {
  update(data: Uint8Array): void;
  finalize(): Promise<Uint8Array>;
}

/**
 * BLAKE3 module interface
 */
interface Blake3Module {
  hash(data: Uint8Array): Promise<Uint8Array>;
  createHasher(): Blake3Hasher;
}

// ============================================================================
// Constants
// ============================================================================

const ALIGNMENT = 4096; // 4KB alignment for optimal disk I/O
const _READ_CHUNK_SIZE = 1024 * 1024; // 1MB chunks for streaming reads
const MODELS_DIR = 'doppler-models';

// ============================================================================
// Module State
// ============================================================================

let rootDir: FileSystemDirectoryHandle | null = null;
let modelsDir: FileSystemDirectoryHandle | null = null;
let currentModelDir: FileSystemDirectoryHandle | null = null;
let blake3Module: Blake3Module | null = null;
let hashAlgorithm: HashAlgorithm | null = null;

// ============================================================================
// BLAKE3/SHA256 Hashing
// ============================================================================

/**
 * Initializes the BLAKE3 hashing module
 * Uses the BLAKE3 WASM implementation, falls back to SHA-256
 */
async function initBlake3(requiredAlgorithm: HashAlgorithm | null = null): Promise<void> {
  if (blake3Module && hashAlgorithm) return;

  // Try to load BLAKE3 WASM module
  try {
    // Dynamic import of blake3 module (should be bundled or loaded separately)
    const globalBlake3 = (globalThis as { blake3?: Blake3Module }).blake3;
    if (globalBlake3 !== undefined) {
      blake3Module = globalBlake3;
      hashAlgorithm = 'blake3';
      return;
    }
  } catch (e) {
    console.warn('BLAKE3 WASM module not available:', (e as Error).message);
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
    hash: async (data: Uint8Array): Promise<Uint8Array> => {
      const hashBuffer = await crypto.subtle.digest('SHA-256', data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer);
      return new Uint8Array(hashBuffer);
    },
    createHasher: (): Blake3Hasher => {
      const chunks: Uint8Array[] = [];
      return {
        update: (data: Uint8Array): void => {
          chunks.push(new Uint8Array(data));
        },
        finalize: async (): Promise<Uint8Array> => {
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
 */
export function getHashAlgorithm(): HashAlgorithm | null {
  return hashAlgorithm;
}

/**
 * Converts Uint8Array to hex string
 */
function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Converts hex string to Uint8Array
 */
export function hexToBytes(hex: string): Uint8Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(hex.substr(i * 2, 2), 16);
  }
  return bytes;
}

/**
 * Computes BLAKE3 hash of data
 */
export async function computeBlake3(data: Uint8Array | ArrayBuffer): Promise<string> {
  await initBlake3();

  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  const hash = await blake3Module!.hash(bytes);
  return bytesToHex(hash);
}

/**
 * Computes SHA-256 hash of data
 */
export async function computeSHA256(data: Uint8Array | ArrayBuffer): Promise<string> {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  return bytesToHex(new Uint8Array(hashBuffer));
}

/**
 * Computes hash using specified algorithm
 */
export async function computeHash(
  data: Uint8Array | ArrayBuffer,
  algorithm: HashAlgorithm = 'blake3'
): Promise<string> {
  if (algorithm === 'sha256') {
    return computeSHA256(data);
  }
  return computeBlake3(data);
}

/**
 * Creates a streaming BLAKE3 hasher for large data
 */
export async function createStreamingHasher(): Promise<Blake3Hasher> {
  await initBlake3();
  return blake3Module!.createHasher();
}

// ============================================================================
// OPFS Operations
// ============================================================================

/**
 * Initializes the OPFS directory structure
 */
export async function initOPFS(): Promise<void> {
  if (!isOPFSAvailable()) {
    throw new Error('OPFS not available in this browser');
  }

  try {
    rootDir = await navigator.storage.getDirectory();
    modelsDir = await rootDir.getDirectoryHandle(MODELS_DIR, { create: true });
  } catch (error) {
    throw new Error(`Failed to initialize OPFS: ${(error as Error).message}`);
  }
}

/**
 * Opens a model directory, creating it if necessary
 */
export async function openModelDirectory(modelId: string): Promise<FileSystemDirectoryHandle> {
  if (!modelsDir) {
    await initOPFS();
  }

  // Sanitize modelId for filesystem
  const safeName = modelId.replace(/[^a-zA-Z0-9_-]/g, '_');
  currentModelDir = await modelsDir!.getDirectoryHandle(safeName, { create: true });
  return currentModelDir;
}

/**
 * Gets the current model directory handle
 */
export function getCurrentModelDirectory(): FileSystemDirectoryHandle | null {
  return currentModelDir;
}

/**
 * Writes a shard to OPFS
 */
export async function writeShard(
  shardIndex: number,
  data: ArrayBuffer,
  options: ShardWriteOptions = { verify: true }
): Promise<ShardWriteResult> {
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
    const alignedSize = Math.ceil(data.byteLength / ALIGNMENT) * ALIGNMENT;
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
    throw new Error(`Failed to write shard ${shardIndex}: ${(error as Error).message}`);
  }
}

/**
 * Reads a shard from OPFS
 */
export async function loadShard(
  shardIndex: number,
  options: ShardReadOptions = { verify: false }
): Promise<ArrayBuffer> {
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
    if ((error as Error).name === 'NotFoundError') {
      throw new Error(`Shard ${shardIndex} not found`);
    }
    throw new Error(`Failed to load shard ${shardIndex}: ${(error as Error).message}`);
  }
}

/**
 * Reads a shard using synchronous access (for Worker threads)
 * Provides better performance for repeated reads
 */
export async function loadShardSync(
  shardIndex: number,
  offset = 0,
  length?: number
): Promise<Uint8Array> {
  if (!currentModelDir) {
    throw new Error('No model directory open. Call openModelDirectory first.');
  }

  const shardInfo = getShardInfo(shardIndex);
  if (!shardInfo) {
    throw new Error(`Invalid shard index: ${shardIndex}`);
  }

  // Align offset to 4KB boundary for optimal reads
  const alignedOffset = Math.floor(offset / ALIGNMENT) * ALIGNMENT;
  const offsetDelta = offset - alignedOffset;

  const readLength = length ?? (shardInfo.size - offset);
  const alignedLength = Math.ceil((readLength + offsetDelta) / ALIGNMENT) * ALIGNMENT;

  try {
    const fileHandle = await currentModelDir.getFileHandle(shardInfo.filename);
    const syncHandle = await (fileHandle as FileSystemFileHandle & {
      createSyncAccessHandle(): Promise<FileSystemSyncAccessHandle>;
    }).createSyncAccessHandle();

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
    if ((error as Error).name === 'NotFoundError') {
      throw new Error(`Shard ${shardIndex} not found`);
    }
    // If sync access not supported, fall back to async
    if ((error as Error).name === 'NotSupportedError') {
      console.warn('Sync access not supported, falling back to async read');
      const buffer = await loadShard(shardIndex);
      return new Uint8Array(buffer, offset, length);
    }
    throw new Error(`Failed to sync-load shard ${shardIndex}: ${(error as Error).message}`);
  }
}

/**
 * Checks if a shard exists in OPFS
 */
export async function shardExists(shardIndex: number): Promise<boolean> {
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
 */
export async function verifyIntegrity(): Promise<IntegrityResult> {
  const manifest = getManifest();
  if (!manifest) {
    throw new Error('No manifest loaded');
  }

  if (!currentModelDir) {
    throw new Error('No model directory open');
  }

  // Get hash algorithm from manifest (default to blake3 for backwards compatibility)
  const algorithm = manifest.hashAlgorithm || 'blake3';

  const missingShards: number[] = [];
  const corruptShards: number[] = [];
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
 */
export async function deleteShard(shardIndex: number): Promise<boolean> {
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
 */
export async function deleteModel(modelId: string): Promise<boolean> {
  if (!modelsDir) {
    await initOPFS();
  }

  const safeName = modelId.replace(/[^a-zA-Z0-9_-]/g, '_');

  try {
    await modelsDir!.removeEntry(safeName, { recursive: true });

    // Clear current model dir if it was this model
    if (currentModelDir) {
      try {
        // Check if current dir is the deleted one
        await currentModelDir.getFileHandle('.test', { create: true })
          .then((_h: FileSystemFileHandle) => currentModelDir!.removeEntry('.test'))
          .catch(() => {});
      } catch {
        currentModelDir = null;
      }
    }

    return true;
  } catch (error) {
    if ((error as Error).name === 'NotFoundError') {
      return true; // Already deleted
    }
    return false;
  }
}

/**
 * Lists all models stored in OPFS
 */
export async function listModels(): Promise<string[]> {
  if (!modelsDir) {
    try {
      await initOPFS();
    } catch {
      return [];
    }
  }

  const models: string[] = [];
  const entries = (modelsDir as unknown as { entries(): AsyncIterable<[string, FileSystemHandle]> }).entries();
  for await (const [name, handle] of entries) {
    if (handle.kind === 'directory') {
      models.push(name);
    }
  }

  return models;
}

/**
 * Gets information about a stored model
 */
export async function getModelInfo(modelId: string): Promise<ModelInfo> {
  if (!modelsDir) {
    await initOPFS();
  }

  const safeName = modelId.replace(/[^a-zA-Z0-9_-]/g, '_');

  try {
    const modelDir = await modelsDir!.getDirectoryHandle(safeName);
    let shardCount = 0;
    let totalSize = 0;
    let hasManifest = false;

    const modelEntries = (modelDir as unknown as { entries(): AsyncIterable<[string, FileSystemHandle]> }).entries();
    for await (const [name, handle] of modelEntries) {
      if (handle.kind === 'file') {
        if (name === 'manifest.json') {
          hasManifest = true;
        } else if (name.startsWith('shard_') && name.endsWith('.bin')) {
          shardCount++;
          const file = await (handle as FileSystemFileHandle).getFile();
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
 */
export async function modelExists(modelId: string): Promise<boolean> {
  const info = await getModelInfo(modelId);
  return info.exists && info.hasManifest;
}

/**
 * Saves the manifest to OPFS
 */
export async function saveManifest(manifestJson: string): Promise<void> {
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
 */
export async function loadManifestFromOPFS(): Promise<string> {
  if (!currentModelDir) {
    throw new Error('No model directory open');
  }

  try {
    const fileHandle = await currentModelDir.getFileHandle('manifest.json');
    const file = await fileHandle.getFile();
    return await file.text();
  } catch (error) {
    if ((error as Error).name === 'NotFoundError') {
      throw new Error('Manifest not found');
    }
    throw error;
  }
}

/**
 * Saves the tokenizer.json to OPFS
 */
export async function saveTokenizer(tokenizerJson: string): Promise<void> {
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
 */
export async function loadTokenizerFromOPFS(): Promise<string | null> {
  if (!currentModelDir) {
    throw new Error('No model directory open');
  }

  try {
    const fileHandle = await currentModelDir.getFileHandle('tokenizer.json');
    const file = await fileHandle.getFile();
    return await file.text();
  } catch (error) {
    if ((error as Error).name === 'NotFoundError') {
      return null; // Tokenizer not bundled, will fall back to HuggingFace
    }
    throw error;
  }
}

/**
 * Cleans up module state (useful for testing)
 */
export function cleanup(): void {
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
export class OpfsShardStore implements ShardStore {
  private modelId: string;
  private initialized = false;

  constructor(modelId: string) {
    this.modelId = modelId;
  }

  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return;
    await openModelDirectory(this.modelId);
    this.initialized = true;
  }

  async read(shardIndex: number, offset: number, length: number): Promise<Uint8Array> {
    await this.ensureInitialized();
    return loadShardSync(shardIndex, offset, length);
  }

  async write(shardIndex: number, data: Uint8Array): Promise<void> {
    await this.ensureInitialized();
    await writeShard(shardIndex, data.buffer as ArrayBuffer, { verify: true });
  }

  async exists(shardIndex: number): Promise<boolean> {
    await this.ensureInitialized();
    return shardExists(shardIndex);
  }

  async delete(shardIndex: number): Promise<void> {
    await this.ensureInitialized();
    await deleteShard(shardIndex);
  }

  async list(): Promise<number[]> {
    await this.ensureInitialized();
    const shardCount = getShardCount();
    const existing: number[] = [];
    for (let i = 0; i < shardCount; i++) {
      if (await this.exists(i)) {
        existing.push(i);
      }
    }
    return existing;
  }
}
