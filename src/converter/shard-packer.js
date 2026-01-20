

import {
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
  #createHasher;
  #computeHash;
  #supportsStreaming;

  // Current shard state
  #currentShardIndex = 0;
  #currentShardData = [];
  #currentShardSize = 0;
  #currentShardWriter = null;
  #currentShardHasher = null;

  // Results
  #shards = [];
  #tensorLocations = new Map();
  #groupTensorMap = new Map();
  #groupHashers = new Map();
  #groupDataMap = new Map();
  #totalSize = 0;

  constructor(io, options = {}) {
    this.#io = io;
    this.#shardSize = options.shardSize;
    this.#hashAlgorithm = options.hashAlgorithm;
    this.#modelType = options.modelType;
    this.#createHasher = typeof io?.createHasher === 'function' ? io.createHasher.bind(io) : null;
    this.#computeHash = typeof io?.computeHash === 'function' ? io.computeHash.bind(io) : null;
    this.#supportsStreaming = typeof io?.createShardWriter === 'function' && typeof this.#createHasher === 'function';
    if (!this.#shardSize || this.#shardSize <= 0) {
      throw new Error('Missing shard size for shard packer');
    }
    if (!this.#hashAlgorithm) {
      throw new Error('Missing hashAlgorithm for shard packer');
    }
    if (!this.#modelType) {
      throw new Error('Missing modelType for shard packer');
    }
    if (typeof this.#computeHash !== 'function') {
      throw new Error('Missing computeHash for shard packer');
    }
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

      // Classify tensor into component group
      const groupId = classifyTensor(tensor.name, this.#modelType);
      this.#addTensorToGroup(groupId, tensor.name);

      // Pack tensor data into shards
      const role = classifyTensorRole(tensor.name);
      if (this.#supportsStreaming) {
        if (typeof tensor.getChunks === 'function') {
          await this.#packTensorStream(tensor, tensor.getChunks(), groupId, role);
        } else {
          const data = await tensor.getData();
          const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
          await this.#packTensorStream(tensor, [bytes], groupId, role);
        }
      } else {
        const data = await tensor.getData();
        const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
        await this.#packTensor(tensor, bytes, groupId, role);
      }
    }

    // Flush final shard
    await this.#flushShard();

    // Build component groups
    const groups = await this.#buildGroups();

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

    await this.#trackGroupData(groupId, data);

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

  
  async #packTensorStream(tensor, chunks, groupId, role) {
    const tensorSpans = [];
    let processed = 0;

    for await (const chunk of chunks) {
      const bytes = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
      if (bytes.length === 0) continue;

      let offset = 0;
      while (offset < bytes.length) {
        await this.#ensureShardWriter();
        const availableInShard = this.#shardSize - this.#currentShardSize;
        const chunkSize = Math.min(bytes.length - offset, availableInShard);
        const slice = bytes.subarray(offset, offset + chunkSize);

        await this.#currentShardWriter.write(slice);
        this.#currentShardHasher.update(slice);
        await this.#trackGroupData(groupId, slice);

        tensorSpans.push({
          shardIndex: this.#currentShardIndex,
          offset: this.#currentShardSize,
          size: chunkSize,
        });

        this.#currentShardSize += chunkSize;
        this.#totalSize += chunkSize;
        processed += chunkSize;
        offset += chunkSize;

        if (this.#currentShardSize >= this.#shardSize) {
          await this.#flushShard();
        }
      }
    }

    if (processed !== tensor.size) {
      throw new Error(`Tensor ${tensor.name} size mismatch: expected ${tensor.size}, got ${processed}`);
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


  async #ensureShardWriter() {
    if (this.#currentShardWriter) return;
    this.#currentShardWriter = await this.#io.createShardWriter(this.#currentShardIndex);
    this.#currentShardHasher = await this.#createHasher();
  }


  async #flushShard() {
    if (this.#currentShardWriter) {
      const hashBytes = await this.#currentShardHasher.finalize();
      await this.#currentShardWriter.close();

      const hash = bytesToHex(hashBytes);

      this.#shards.push({
        index: this.#currentShardIndex,
        filename: generateShardFilename(this.#currentShardIndex),
        size: this.#currentShardSize,
        hash,
        offset: this.#currentShardIndex * this.#shardSize,
      });

      this.#currentShardIndex++;
      this.#currentShardWriter = null;
      this.#currentShardHasher = null;
      this.#currentShardSize = 0;
      return;
    }

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

  
  async #trackGroupData(groupId, data) {
    if (this.#createHasher) {
      let hasher = this.#groupHashers.get(groupId);
      if (!hasher) {
        hasher = await this.#createHasher();
        this.#groupHashers.set(groupId, hasher);
      }
      hasher.update(data);
      return;
    }

    const existing = this.#groupDataMap.get(groupId) || [];
    existing.push(data);
    this.#groupDataMap.set(groupId, existing);
  }


  async #finalizeGroupHashes() {
    const hashes = new Map();
    for (const [groupId, hasher] of this.#groupHashers.entries()) {
      const hashBytes = await hasher.finalize();
      hashes.set(groupId, bytesToHex(hashBytes));
    }

    for (const [groupId, chunks] of this.#groupDataMap.entries()) {
      if (hashes.has(groupId)) continue;
      const totalSize = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const combined = new Uint8Array(totalSize);
      let offset = 0;
      for (const chunk of chunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }
      const hash = await this.#computeHash(combined);
      hashes.set(groupId, hash);
    }

    return hashes;
  }


  #addTensorToGroup(groupId, tensorName) {
    const existing = this.#groupTensorMap.get(groupId) || [];
    existing.push(tensorName);
    this.#groupTensorMap.set(groupId, existing);
  }

  
  async #buildGroups() {
    const groupHashes = await this.#finalizeGroupHashes();
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
      const hash = groupHashes.get(groupId);
      if (!hash) {
        throw new Error(`Missing hash for group ${groupId}`);
      }

      groups[groupId] = {
        type: getGroupType(groupId, this.#modelType),
        version: '1.0.0',
        shards: Array.from(shardSet).sort((a, b) => a - b),
        tensors: tensorNames,
        hash,
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
    this.#currentShardWriter = null;
    this.#currentShardHasher = null;
    this.#shards = [];
    this.#tensorLocations.clear();
    this.#groupTensorMap.clear();
    this.#groupHashers.clear();
    this.#groupDataMap.clear();
    this.#totalSize = 0;
  }
}


function bytesToHex(bytes) {
  return Array.from(bytes)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
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


export function estimateShardCount(tensors, shardSize) {
  if (!shardSize || shardSize <= 0) {
    throw new Error('Missing shard size for shard count estimate');
  }
  const totalSize = tensors.reduce((sum, t) => sum + t.size, 0);
  return Math.ceil(totalSize / shardSize);
}
