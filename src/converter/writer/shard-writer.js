

import { writeFile } from 'fs/promises';
import { join } from 'path';
import { computeHash, alignOffset, createPadding } from './utils.js';
import { formatBytes, MB } from '../../config/schema/index.js';

function formatDuration(ms) {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export class ShardWriter {
  #outputDir;
  #shardSize;
  #hashAlgorithm;

  #shards = [];
  #currentShard = null;
  #currentShardIndex = 0;
  #currentShardOffset = 0;
  #cumulativeOffset = 0;
  #shardStartTime = null;
  #totalTensorsInShard = 0;

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
    this.#shardStartTime = performance.now();
    this.#totalTensorsInShard = 0;
    console.log(`  Shard ${this.#currentShardIndex}: writing...`);
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
    const shardFilename = `shard_${String(this.#currentShardIndex).padStart(5, '0')}.bin`;
    const shardPath = join(this.#outputDir, shardFilename);
    await writeFile(shardPath, shardData);

    const elapsed = this.#shardStartTime ? performance.now() - this.#shardStartTime : 0;
    const throughput = elapsed > 0 ? (totalSize / MB) / (elapsed / 1000) : 0;
    console.log(
      `  Shard ${this.#currentShardIndex}: done ` +
      `(${formatBytes(totalSize)}, ${this.#totalTensorsInShard} tensors, ` +
      `${formatDuration(elapsed)}, ${throughput.toFixed(1)} MB/s)`
    );

    this.#shards.push({
      index: this.#currentShardIndex,
      filename: shardFilename,
      size: totalSize,
      hash,
      hashAlgorithm: this.#hashAlgorithm,
      offset: this.#cumulativeOffset,
    });

    this.#cumulativeOffset += totalSize;
    this.#currentShardIndex++;
    this.#currentShardOffset = 0;
    this.#currentShard = null;
  }


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

    this.#totalTensorsInShard++;

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
