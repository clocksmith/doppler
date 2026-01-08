/**
 * RDRR Shard Writer
 *
 * @module converter/writer/shard-writer
 */

import type { ShardRecord, HashAlgorithm } from './types.js';

/**
 * Manages writing tensors to binary shards with alignment and hashing.
 */
export declare class ShardWriter {
  constructor(outputDir: string, shardSize: number, hashAlgorithm: HashAlgorithm);

  get shardIndex(): number;
  get offset(): number;
  get finalizedShards(): ShardRecord[];

  startNewShard(): void;
  finalizeShard(): Promise<void>;
  writeData(data: Uint8Array): Promise<Array<{ shardIndex: number; offset: number; size: number }>>;
}
