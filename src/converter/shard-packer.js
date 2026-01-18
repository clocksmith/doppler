

import {
  SHARD_SIZE,
  generateShardFilename,
  classifyTensor,
  classifyTensorRole,
  getGroupType,
  sortGroupIds,
} from '../storage/rdrr-format.js';


export class ShardPacker {
  #io;
  #shardSize;
  #hashAlgorithm;
  #modelType;

  // Current shard state
  #currentShardIndex = 0;
  #currentShardData = [];
  #currentShardSize = 0;

  // Results
  #shards = [];
  #tensorLocations = new Map();
  #groupTensorMap = new Map();
  #totalSize = 0;

  constructor(io, options = {}) {
    this.#io = io;
    this.#shardSize = options.shardSize ?? SHARD_SIZE;
    this.#hashAlgorithm = options.hashAlgorithm ?? 'sha256';
    this.#modelType = options.modelType ?? 'transformer';
  }

  
  async pack(tensors, options = {}) {
    const { onProgress, signal } = options;
    const totalTensors = tensors.length;

    for (let i = 0; i < tensors.length; i++) {
      if (signal?.aborted) {
        throw new DOMException('Aborted', 'AbortError');
      }

      const tensor = tensors[i];
      onProgress?.(i + 1, totalTensors, tensor.name);

      // Get tensor data
      const data = await tensor.getData();

      // Classify tensor into component group
      const groupId = classifyTensor(tensor.name, this.#modelType);
      this.#addTensorToGroup(groupId, tensor.name);

      // Pack tensor data into shards
      const role = classifyTensorRole(tensor.name);
      await this.#packTensor(tensor, data, groupId, role);
    }

    // Flush final shard
    if (this.#currentShardData.length > 0) {
      await this.#flushShard();
    }

    // Build component groups
    const groups = this.#buildGroups();

    return {
      shards: this.#shards,
      tensors: Object.fromEntries(this.#tensorLocations),
      groups,
      totalSize: this.#totalSize,
      tensorCount: tensors.length,
    };
  }

  
  async #packTensor(tensor, data, groupId, role) {
    const tensorSpans = [];
    let remaining = data;
    let remainingOffset = 0;

    while (remaining.length > 0) {
      const availableInShard = this.#shardSize - this.#currentShardSize;
      const chunkSize = Math.min(remaining.length, availableInShard);

      // Add chunk to current shard
      this.#currentShardData.push(remaining.slice(0, chunkSize));

      // Track span
      tensorSpans.push({
        shardIndex: this.#currentShardIndex,
        offset: this.#currentShardSize,
        size: chunkSize,
      });

      this.#currentShardSize += chunkSize;
      this.#totalSize += chunkSize;

      remaining = remaining.slice(chunkSize);
      remainingOffset += chunkSize;

      // Flush shard if full
      if (this.#currentShardSize >= this.#shardSize) {
        await this.#flushShard();
      }
    }

    // Record tensor location
    if (tensorSpans.length === 1) {
      this.#tensorLocations.set(tensor.name, {
        shard: tensorSpans[0].shardIndex,
        offset: tensorSpans[0].offset,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
        role,
        group: groupId,
      });
    } else {
      this.#tensorLocations.set(tensor.name, {
        spans: tensorSpans,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
        role,
        group: groupId,
      });
    }
  }

  
  async #flushShard() {
    if (this.#currentShardData.length === 0) return;

    // Concatenate chunks
    const totalSize = this.#currentShardData.reduce((sum, chunk) => sum + chunk.length, 0);
    const shardData = new Uint8Array(totalSize);
    let offset = 0;
    for (const chunk of this.#currentShardData) {
      shardData.set(chunk, offset);
      offset += chunk.length;
    }

    // Write shard via I/O adapter
    const hash = await this.#io.writeShard(this.#currentShardIndex, shardData);

    // Record shard info
    this.#shards.push({
      index: this.#currentShardIndex,
      filename: generateShardFilename(this.#currentShardIndex),
      size: shardData.length,
      hash,
      offset: this.#currentShardIndex * this.#shardSize,
    });

    // Reset for next shard
    this.#currentShardIndex++;
    this.#currentShardData = [];
    this.#currentShardSize = 0;
  }

  
  #addTensorToGroup(groupId, tensorName) {
    const existing = this.#groupTensorMap.get(groupId) || [];
    existing.push(tensorName);
    this.#groupTensorMap.set(groupId, existing);
  }

  
  #buildGroups() {
    const groups = {};
    const sortedGroupIds = sortGroupIds(Array.from(this.#groupTensorMap.keys()));

    for (const groupId of sortedGroupIds) {
      const tensorNames = this.#groupTensorMap.get(groupId) || [];

      // Collect unique shards for this group
      const shardSet = new Set();
      for (const name of tensorNames) {
        const loc = this.#tensorLocations.get(name);
        if (!loc) continue;
        if ('shard' in loc) {
          shardSet.add(loc.shard);
        } else if ('spans' in loc) {
          for (const span of loc.spans) {
            shardSet.add(span.shardIndex);
          }
        }
      }

      // Parse layer/expert indices from group ID
      const layerMatch = groupId.match(/^layer\.(\d+)/);
      const expertMatch = groupId.match(/\.expert\.(\d+)$/);

      groups[groupId] = {
        type: getGroupType(groupId, this.#modelType),
        version: '1.0.0',
        shards: Array.from(shardSet).sort((a, b) => a - b),
        tensors: tensorNames,
        hash: '', // Will be computed when writing manifest
        layerIndex: layerMatch ? parseInt(layerMatch[1], 10) : undefined,
        expertIndex: expertMatch ? parseInt(expertMatch[1], 10) : undefined,
      };
    }

    return groups;
  }

  
  reset() {
    this.#currentShardIndex = 0;
    this.#currentShardData = [];
    this.#currentShardSize = 0;
    this.#shards = [];
    this.#tensorLocations.clear();
    this.#groupTensorMap.clear();
    this.#totalSize = 0;
  }
}


export function sortTensorsByGroup(tensors, modelType = 'transformer') {
  return [...tensors].sort((a, b) => {
    const groupA = classifyTensor(a.name, modelType);
    const groupB = classifyTensor(b.name, modelType);

    // Use sortGroupIds logic for consistent ordering
    const sorted = sortGroupIds([groupA, groupB]);
    if (sorted[0] === groupA && sorted[1] === groupB) return -1;
    if (sorted[0] === groupB && sorted[1] === groupA) return 1;
    return 0;
  });
}


export function estimateShardCount(tensors, shardSize = SHARD_SIZE) {
  const totalSize = tensors.reduce((sum, t) => sum + t.size, 0);
  return Math.ceil(totalSize / shardSize);
}
