/**
 * shard-io-browser.ts - Browser I/O Adapter for Shard Packer
 *
 * Implements ShardIO interface using OPFS (Origin Private File System).
 *
 * @module browser/shard-io-browser
 */

import { generateShardFilename } from '../storage/rdrr-format.js';

/**
 * Browser/OPFS implementation of ShardIO interface.
 */
export class BrowserShardIO {
  constructor(modelDir) {
    this.modelDir = modelDir;
  }

  /**
   * Create a BrowserShardIO from a model ID.
   * Opens or creates the model directory in OPFS.
   */
  static async create(modelId) {
    const opfsRoot = await navigator.storage.getDirectory();
    const modelsDir = await opfsRoot.getDirectoryHandle('models', { create: true });
    const modelDir = await modelsDir.getDirectoryHandle(modelId, { create: true });
    return new BrowserShardIO(modelDir);
  }

  /**
   * Write shard data to OPFS, returns hash.
   */
  async writeShard(index, data) {
    const filename = generateShardFilename(index);
    const fileHandle = await this.modelDir.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    // Use ArrayBuffer for FileSystemWritableFileStream compatibility
    const buffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    await writable.write(buffer);
    await writable.close();
    return this.computeHash(data);
  }

  /**
   * Compute SHA-256 hash using Web Crypto API.
   */
  async computeHash(data) {
    // Use ArrayBuffer slice for SubtleCrypto compatibility
    const buffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = new Uint8Array(hashBuffer);
    return Array.from(hashArray)
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');
  }

  /**
   * Write a JSON file to the model directory.
   */
  async writeJson(filename, data) {
    const fileHandle = await this.modelDir.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(JSON.stringify(data, null, 2));
    await writable.close();
  }

  /**
   * Write raw file to model directory.
   */
  async writeFile(filename, data) {
    const fileHandle = await this.modelDir.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    if (typeof data === 'string') {
      await writable.write(data);
    } else {
      const buffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
      await writable.write(buffer);
    }
    await writable.close();
  }

  /**
   * Get the model directory handle.
   */
  getModelDir() {
    return this.modelDir;
  }

  /**
   * Delete all files in the model directory.
   */
  async clear() {
    const entries = this.modelDir.values();
    for await (const entry of entries) {
      await this.modelDir.removeEntry(entry.name);
    }
  }
}

/**
 * Check if OPFS is supported in this browser.
 */
export function isOPFSSupported() {
  return (
    typeof navigator !== 'undefined' &&
    'storage' in navigator &&
    'getDirectory' in (navigator.storage)
  );
}
