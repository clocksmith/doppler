/**
 * shard-io-browser.ts - Browser I/O Adapter for Shard Packer
 *
 * Implements ShardIO interface using OPFS (Origin Private File System).
 *
 * @module browser/shard-io-browser
 */

import type { ShardIO } from '../converter/shard-packer.js';

/**
 * Browser/OPFS implementation of ShardIO interface.
 */
export declare class BrowserShardIO implements ShardIO {
  constructor(modelDir: FileSystemDirectoryHandle, options?: { hashAlgorithm?: string });

  /**
   * Create a BrowserShardIO from a model ID.
   * Opens or creates the model directory in OPFS.
   */
  static create(modelId: string, options?: { hashAlgorithm?: string }): Promise<BrowserShardIO>;

  /**
   * Write shard data to OPFS, returns hash.
   */
  writeShard(index: number, data: Uint8Array): Promise<string>;

  /**
   * Compute hash using the configured hash algorithm.
   */
  computeHash(data: Uint8Array): Promise<string>;

  /**
   * Write a JSON file to the model directory.
   */
  writeJson(filename: string, data: unknown): Promise<void>;

  /**
   * Write raw file to model directory.
   */
  writeFile(filename: string, data: string | Uint8Array): Promise<void>;

  /**
   * Get the model directory handle.
   */
  getModelDir(): FileSystemDirectoryHandle;

  /**
   * Delete all files in the model directory.
   */
  clear(): Promise<void>;
}

/**
 * Check if OPFS is supported in this browser.
 */
export declare function isOPFSSupported(): boolean;
