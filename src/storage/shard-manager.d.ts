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

import type { OpfsPathConfigSchema } from '../config/schema/loading.schema.js';

// Re-export for consumers that import from shard-manager
export { getManifest } from './rdrr-format.js';
export type { ShardInfo, RDRRManifest, HashAlgorithm } from './rdrr-format.js';

import type { HashAlgorithm } from './rdrr-format.js';

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
 * Set OPFS path configuration
 */
export declare function setOpfsPathConfig(config: OpfsPathConfigSchema): void;

/**
 * Get current OPFS path configuration
 */
export declare function getOpfsPathConfig(): OpfsPathConfigSchema;

/**
 * Get the current hash algorithm in use
 */
export declare function getHashAlgorithm(): HashAlgorithm | null;

/**
 * Converts hex string to Uint8Array
 */
export declare function hexToBytes(hex: string): Uint8Array;

/**
 * Computes BLAKE3 hash of data
 */
export declare function computeBlake3(data: Uint8Array | ArrayBuffer): Promise<string>;

/**
 * Computes SHA-256 hash of data
 */
export declare function computeSHA256(data: Uint8Array | ArrayBuffer): Promise<string>;

/**
 * Computes hash using specified algorithm
 */
export declare function computeHash(
  data: Uint8Array | ArrayBuffer,
  algorithm?: HashAlgorithm
): Promise<string>;

/**
 * BLAKE3 hasher interface
 */
interface Blake3Hasher {
  update(data: Uint8Array): void;
  finalize(): Promise<Uint8Array>;
}

/**
 * Creates a streaming BLAKE3 hasher for large data
 */
export declare function createStreamingHasher(): Promise<Blake3Hasher>;

/**
 * Initializes the OPFS directory structure
 */
export declare function initOPFS(): Promise<void>;

/**
 * Opens a model directory, creating it if necessary
 */
export declare function openModelDirectory(modelId: string): Promise<FileSystemDirectoryHandle>;

/**
 * Gets the current model directory handle
 */
export declare function getCurrentModelDirectory(): FileSystemDirectoryHandle | null;

/**
 * Writes a shard to OPFS
 */
export declare function writeShard(
  shardIndex: number,
  data: ArrayBuffer,
  options?: ShardWriteOptions
): Promise<ShardWriteResult>;

/**
 * Reads a shard from OPFS
 */
export declare function loadShard(
  shardIndex: number,
  options?: ShardReadOptions
): Promise<ArrayBuffer>;

/**
 * Reads a shard using synchronous access (for Worker threads)
 * Provides better performance for repeated reads
 */
export declare function loadShardSync(
  shardIndex: number,
  offset?: number,
  length?: number
): Promise<Uint8Array>;

/**
 * Checks if a shard exists in OPFS
 */
export declare function shardExists(shardIndex: number): Promise<boolean>;

/**
 * Verifies the integrity of all shards
 */
export declare function verifyIntegrity(): Promise<IntegrityResult>;

/**
 * Deletes a shard from OPFS
 */
export declare function deleteShard(shardIndex: number): Promise<boolean>;

/**
 * Deletes an entire model from OPFS
 */
export declare function deleteModel(modelId: string): Promise<boolean>;

/**
 * Lists all models stored in OPFS
 */
export declare function listModels(): Promise<string[]>;

/**
 * Gets information about a stored model
 */
export declare function getModelInfo(modelId: string): Promise<ModelInfo>;

/**
 * Checks if a model exists in OPFS
 */
export declare function modelExists(modelId: string): Promise<boolean>;

/**
 * Saves the manifest to OPFS
 */
export declare function saveManifest(manifestJson: string): Promise<void>;

/**
 * Loads the manifest from OPFS
 */
export declare function loadManifestFromOPFS(): Promise<string>;

/**
 * Loads the tensors.json from OPFS (v1 format)
 * @returns Tensors JSON string or null if not found
 */
export declare function loadTensorsFromOPFS(): Promise<string | null>;

/**
 * Saves the tokenizer.json to OPFS
 */
export declare function saveTokenizer(tokenizerJson: string): Promise<void>;

/**
 * Loads the tokenizer.json from OPFS
 */
export declare function loadTokenizerFromOPFS(): Promise<string | null>;

/**
 * Cleans up module state (useful for testing)
 */
export declare function cleanup(): void;

/**
 * OPFS-backed shard store implementing the ShardStore interface
 */
export declare class OpfsShardStore implements ShardStore {
  constructor(modelId: string);
  read(shardIndex: number, offset: number, length: number): Promise<Uint8Array>;
  write(shardIndex: number, data: Uint8Array): Promise<void>;
  exists(shardIndex: number): Promise<boolean>;
  delete(shardIndex: number): Promise<void>;
  list(): Promise<number[]>;
}
