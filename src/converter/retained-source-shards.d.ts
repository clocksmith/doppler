import type { RetainedSourceBinding } from './source-tensor-rules.js';

export interface RetainedSourceTensor {
  name: string;
  shape: number[];
  dtype: string;
  /** Absolute byte offset in the source file. */
  offset: number;
  size: number;
  sourcePath?: string;
  role?: string;
  group?: string | null;
  retainedSource?: RetainedSourceBinding;
  [key: string]: unknown;
}

export interface RetainedSourceShardIO {
  /**
   * Make the source file artifact shard `index` without copying its bytes and
   * return its size and SHA-256; fails when the bytes differ from the pin.
   */
  retainSourceFile?(sourcePath: string, index: number, sha256: string): Promise<{ size: number; hash: string }>;
}

export interface RetainedShardEntry {
  index: number;
  filename: string;
  size: number;
  hash: string;
  offset: number;
}

export interface RetainedSourceShardReceipt {
  shard: number;
  file: string;
  sha256: string;
  bytes: number;
  tensors: number;
}

export declare function partitionRetainedSourceTensors<T extends { retainedSource?: RetainedSourceBinding }>(
  tensors: T[]
): { packed: T[]; retained: T[] };

/**
 * Append one shard per retained source file after the packed shards, record
 * each retained tensor's byte span there, and set
 * manifestConfig.conversion.retainedSourceShards.
 */
export declare function appendRetainedSourceShards(
  retained: RetainedSourceTensor[],
  io: RetainedSourceShardIO,
  shards: RetainedShardEntry[],
  tensorLocations: Record<string, unknown>,
  options: {
    hashAlgorithm: string;
    manifestConfig: { conversion?: Record<string, unknown> | null; [key: string]: unknown };
    signal?: AbortSignal | null;
  }
): Promise<RetainedSourceShardReceipt[]>;
