/**
 * RDRR Shard Writer
 *
 * Handles creation and packing of binary shards.
 * Manages shard lifecycle: create, write chunks, finalize with hash.
 *
 * @module converter/writer/shard-writer
 */

import { writeFile } from 'fs/promises';
import { join } from 'path';
import type { ShardData, ShardRecord, HashAlgorithm } from './types.js';
import { computeHash, alignOffset, createPadding } from './utils.js';

/**
 * Manages writing tensors to binary shards with alignment and hashing.
 */
export class ShardWriter {
  private outputDir: string;
  private shardSize: number;
  private hashAlgorithm: HashAlgorithm;

  private shards: ShardRecord[] = [];
  private currentShard: ShardData | null = null;
  private currentShardIndex = 0;
  private currentShardOffset = 0;

  constructor(outputDir: string, shardSize: number, hashAlgorithm: HashAlgorithm) {
    this.outputDir = outputDir;
    this.shardSize = shardSize;
    this.hashAlgorithm = hashAlgorithm;
  }

  /** Get current shard index */
  get shardIndex(): number {
    return this.currentShardIndex;
  }

  /** Get current offset within shard */
  get offset(): number {
    return this.currentShardOffset;
  }

  /** Get all finalized shard records */
  get finalizedShards(): ShardRecord[] {
    return this.shards;
  }

  /** Start a new shard, finalizing current if needed */
  startNewShard(): void {
    if (this.currentShard && this.currentShardOffset > 0) {
      // Will be finalized in async context
    }

    this.currentShard = {
      index: this.currentShardIndex,
      data: [],
      size: 0,
    };
  }

  /** Finalize current shard: concatenate chunks, compute hash, write to disk */
  async finalizeShard(): Promise<void> {
    if (!this.currentShard || this.currentShard.size === 0) {
      return;
    }

    const totalSize = this.currentShard.data.reduce((sum, chunk) => sum + chunk.length, 0);
    const shardData = new Uint8Array(totalSize);
    let offset = 0;
    for (const chunk of this.currentShard.data) {
      shardData.set(chunk, offset);
      offset += chunk.length;
    }

    const hash = await computeHash(shardData, this.hashAlgorithm);
    const shardFileName = `shard_${String(this.currentShardIndex).padStart(5, '0')}.bin`;
    const shardPath = join(this.outputDir, shardFileName);
    await writeFile(shardPath, shardData);

    this.shards.push({
      index: this.currentShardIndex,
      fileName: shardFileName,
      size: totalSize,
      hash,
      hashAlgorithm: this.hashAlgorithm,
    });

    this.currentShardIndex++;
    this.currentShardOffset = 0;
    this.currentShard = null;
  }

  /**
   * Write data to shards with alignment.
   * Handles spanning across multiple shards if needed.
   *
   * @returns Array of shard spans where data was written
   */
  async writeData(data: Uint8Array): Promise<Array<{ shardIndex: number; offset: number; size: number }>> {
    if (!this.currentShard) {
      this.startNewShard();
    }

    // Add alignment padding if needed
    const alignedOffset = alignOffset(this.currentShardOffset);
    const spaceNeeded = (alignedOffset - this.currentShardOffset) + data.length;

    // Start new shard if current can't fit the data
    if (this.currentShardOffset > 0 && this.currentShardOffset + spaceNeeded > this.shardSize) {
      await this.finalizeShard();
      this.startNewShard();
    }

    // Add padding for alignment
    const paddingNeeded = alignOffset(this.currentShardOffset) - this.currentShardOffset;
    if (paddingNeeded > 0) {
      this.currentShard!.data.push(createPadding(paddingNeeded));
      this.currentShard!.size += paddingNeeded;
      this.currentShardOffset += paddingNeeded;
    }

    const spans: Array<{ shardIndex: number; offset: number; size: number }> = [];
    let remaining = data;

    while (remaining.length > 0) {
      const spaceInShard = this.shardSize - this.currentShardOffset;
      const writeSize = Math.min(remaining.length, spaceInShard);

      const chunk = remaining.slice(0, writeSize);
      this.currentShard!.data.push(chunk);
      this.currentShard!.size += writeSize;
      this.currentShardOffset += writeSize;

      spans.push({
        shardIndex: this.currentShardIndex,
        offset: this.currentShardOffset - writeSize,
        size: writeSize,
      });

      remaining = remaining.slice(writeSize);

      if (remaining.length > 0) {
        await this.finalizeShard();
        this.startNewShard();
      }
    }

    return spans;
  }
}
