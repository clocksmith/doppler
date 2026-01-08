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
import { computeHash, alignOffset, createPadding } from './utils.js';

/**
 * Manages writing tensors to binary shards with alignment and hashing.
 */
export class ShardWriter {
  #outputDir;
  #shardSize;
  #hashAlgorithm;

  #shards = [];
  #currentShard = null;
  #currentShardIndex = 0;
  #currentShardOffset = 0;

  constructor(outputDir, shardSize, hashAlgorithm) {
    this.#outputDir = outputDir;
    this.#shardSize = shardSize;
    this.#hashAlgorithm = hashAlgorithm;
  }

  get shardIndex() {
    return this.#currentShardIndex;
  }

  get offset() {
    return this.#currentShardOffset;
  }

  get finalizedShards() {
    return this.#shards;
  }

  startNewShard() {
    if (this.#currentShard && this.#currentShardOffset > 0) {
      // Will be finalized in async context
    }

    this.#currentShard = {
      index: this.#currentShardIndex,
      data: [],
      size: 0,
    };
  }

  async finalizeShard() {
    if (!this.#currentShard || this.#currentShard.size === 0) {
      return;
    }

    const totalSize = this.#currentShard.data.reduce((sum, chunk) => sum + chunk.length, 0);
    const shardData = new Uint8Array(totalSize);
    let offset = 0;
    for (const chunk of this.#currentShard.data) {
      shardData.set(chunk, offset);
      offset += chunk.length;
    }

    const hash = await computeHash(shardData, this.#hashAlgorithm);
    const shardFileName = `shard_${String(this.#currentShardIndex).padStart(5, '0')}.bin`;
    const shardPath = join(this.#outputDir, shardFileName);
    await writeFile(shardPath, shardData);

    this.#shards.push({
      index: this.#currentShardIndex,
      fileName: shardFileName,
      size: totalSize,
      hash,
      hashAlgorithm: this.#hashAlgorithm,
    });

    this.#currentShardIndex++;
    this.#currentShardOffset = 0;
    this.#currentShard = null;
  }

  /**
   * Write data to shards with alignment.
   * Handles spanning across multiple shards if needed.
   */
  async writeData(data) {
    if (!this.#currentShard) {
      this.startNewShard();
    }

    // Add alignment padding if needed
    const alignedOffset = alignOffset(this.#currentShardOffset);
    const spaceNeeded = (alignedOffset - this.#currentShardOffset) + data.length;

    // Start new shard if current can't fit the data
    if (this.#currentShardOffset > 0 && this.#currentShardOffset + spaceNeeded > this.#shardSize) {
      await this.finalizeShard();
      this.startNewShard();
    }

    // Add padding for alignment
    const paddingNeeded = alignOffset(this.#currentShardOffset) - this.#currentShardOffset;
    if (paddingNeeded > 0) {
      this.#currentShard.data.push(createPadding(paddingNeeded));
      this.#currentShard.size += paddingNeeded;
      this.#currentShardOffset += paddingNeeded;
    }

    const spans = [];
    let remaining = data;

    while (remaining.length > 0) {
      const spaceInShard = this.#shardSize - this.#currentShardOffset;
      const writeSize = Math.min(remaining.length, spaceInShard);

      const chunk = remaining.slice(0, writeSize);
      this.#currentShard.data.push(chunk);
      this.#currentShard.size += writeSize;
      this.#currentShardOffset += writeSize;

      spans.push({
        shardIndex: this.#currentShardIndex,
        offset: this.#currentShardOffset - writeSize,
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
