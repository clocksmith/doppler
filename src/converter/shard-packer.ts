/**
 * shard-packer.ts - Platform-agnostic Shard Packing
 *
 * Core shard packing logic shared between Node.js CLI and browser converters.
 * Uses an I/O adapter interface for platform-specific operations.
 *
 * @module converter/shard-packer
 */

import {
  SHARD_SIZE,
  generateShardFilename,
  classifyTensor,
  getGroupType,
  sortGroupIds,
} from '../storage/rdrr-format.js';

import {
  type ModelType,
  type HashAlgorithm,
  type ComponentGroupSchema,
  type TensorInfoSchema,
  type ShardSchema,
} from '../config/schema/index.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Platform-specific I/O adapter interface.
 * Implementations provided by Node.js or browser runtime.
 */
export interface ShardIO {
  /** Write shard data, returns hash */
  writeShard(index: number, data: Uint8Array): Promise<string>;
  /** Compute hash of data */
  computeHash(data: Uint8Array): Promise<string>;
}

/**
 * Tensor span for multi-shard tensors
 */
export interface TensorSpan {
  shardIndex: number;
  offset: number;
  size: number;
}

/**
 * Tensor location (single shard)
 */
export interface TensorLocationSingle {
  shard: number;
  offset: number;
  size: number;
  shape: number[];
  dtype: string;
  group?: string;
}

/**
 * Tensor location (multi shard)
 */
export interface TensorLocationMulti {
  spans: TensorSpan[];
  size: number;
  shape: number[];
  dtype: string;
  group?: string;
}

export type TensorLocation = TensorLocationSingle | TensorLocationMulti;

/**
 * Shard packer options
 */
export interface ShardPackerOptions {
  shardSize?: number;
  hashAlgorithm?: HashAlgorithm;
  modelType?: ModelType;
  /** Progress callback */
  onProgress?: (current: number, total: number, tensorName: string) => void;
  /** Abort signal */
  signal?: AbortSignal;
}

/**
 * Shard packer result
 */
export interface ShardPackerResult {
  shards: ShardSchema[];
  tensors: Record<string, TensorLocation>;
  groups: Record<string, ComponentGroupSchema>;
  totalSize: number;
  tensorCount: number;
}

/**
 * Input tensor with data getter
 */
export interface PackerTensorInput {
  name: string;
  shape: number[];
  dtype: string;
  size: number;
  /** Async function to get tensor data */
  getData: () => Promise<Uint8Array>;
}

// ============================================================================
// ShardPacker Class
// ============================================================================

/**
 * Platform-agnostic shard packer.
 * Handles shard boundary logic, tensor tracking, and component groups.
 */
export class ShardPacker {
  private io: ShardIO;
  private shardSize: number;
  private hashAlgorithm: HashAlgorithm;
  private modelType: ModelType;

  // Current shard state
  private currentShardIndex = 0;
  private currentShardData: Uint8Array[] = [];
  private currentShardSize = 0;

  // Results
  private shards: ShardSchema[] = [];
  private tensorLocations = new Map<string, TensorLocation>();
  private groupTensorMap = new Map<string, string[]>();
  private totalSize = 0;

  constructor(io: ShardIO, options: ShardPackerOptions = {}) {
    this.io = io;
    this.shardSize = options.shardSize ?? SHARD_SIZE;
    this.hashAlgorithm = options.hashAlgorithm ?? 'sha256';
    this.modelType = options.modelType ?? 'transformer';
  }

  /**
   * Pack tensors into shards.
   * Tensors are processed in order - caller should sort by group first.
   */
  async pack(
    tensors: PackerTensorInput[],
    options: { onProgress?: ShardPackerOptions['onProgress']; signal?: AbortSignal } = {}
  ): Promise<ShardPackerResult> {
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
      const groupId = classifyTensor(tensor.name, this.modelType);
      this.addTensorToGroup(groupId, tensor.name);

      // Pack tensor data into shards
      await this.packTensor(tensor, data, groupId);
    }

    // Flush final shard
    if (this.currentShardData.length > 0) {
      await this.flushShard();
    }

    // Build component groups
    const groups = this.buildGroups();

    return {
      shards: this.shards,
      tensors: Object.fromEntries(this.tensorLocations),
      groups,
      totalSize: this.totalSize,
      tensorCount: tensors.length,
    };
  }

  /**
   * Pack a single tensor, handling shard boundaries.
   */
  private async packTensor(
    tensor: PackerTensorInput,
    data: Uint8Array,
    groupId: string
  ): Promise<void> {
    const tensorSpans: TensorSpan[] = [];
    let remaining = data;
    let remainingOffset = 0;

    while (remaining.length > 0) {
      const availableInShard = this.shardSize - this.currentShardSize;
      const chunkSize = Math.min(remaining.length, availableInShard);

      // Add chunk to current shard
      this.currentShardData.push(remaining.slice(0, chunkSize));

      // Track span
      tensorSpans.push({
        shardIndex: this.currentShardIndex,
        offset: this.currentShardSize,
        size: chunkSize,
      });

      this.currentShardSize += chunkSize;
      this.totalSize += chunkSize;

      remaining = remaining.slice(chunkSize);
      remainingOffset += chunkSize;

      // Flush shard if full
      if (this.currentShardSize >= this.shardSize) {
        await this.flushShard();
      }
    }

    // Record tensor location
    if (tensorSpans.length === 1) {
      this.tensorLocations.set(tensor.name, {
        shard: tensorSpans[0].shardIndex,
        offset: tensorSpans[0].offset,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
        group: groupId,
      });
    } else {
      this.tensorLocations.set(tensor.name, {
        spans: tensorSpans,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
        group: groupId,
      });
    }
  }

  /**
   * Flush current shard to storage.
   */
  private async flushShard(): Promise<void> {
    if (this.currentShardData.length === 0) return;

    // Concatenate chunks
    const totalSize = this.currentShardData.reduce((sum, chunk) => sum + chunk.length, 0);
    const shardData = new Uint8Array(totalSize);
    let offset = 0;
    for (const chunk of this.currentShardData) {
      shardData.set(chunk, offset);
      offset += chunk.length;
    }

    // Write shard via I/O adapter
    const hash = await this.io.writeShard(this.currentShardIndex, shardData);

    // Record shard info
    this.shards.push({
      index: this.currentShardIndex,
      filename: generateShardFilename(this.currentShardIndex),
      size: shardData.length,
      hash,
      offset: this.currentShardIndex * this.shardSize,
    });

    // Reset for next shard
    this.currentShardIndex++;
    this.currentShardData = [];
    this.currentShardSize = 0;
  }

  /**
   * Add tensor to component group tracking.
   */
  private addTensorToGroup(groupId: string, tensorName: string): void {
    const existing = this.groupTensorMap.get(groupId) || [];
    existing.push(tensorName);
    this.groupTensorMap.set(groupId, existing);
  }

  /**
   * Build component groups with hashes.
   */
  private buildGroups(): Record<string, ComponentGroupSchema> {
    const groups: Record<string, ComponentGroupSchema> = {};
    const sortedGroupIds = sortGroupIds(Array.from(this.groupTensorMap.keys()));

    for (const groupId of sortedGroupIds) {
      const tensorNames = this.groupTensorMap.get(groupId) || [];

      // Collect unique shards for this group
      const shardSet = new Set<number>();
      for (const name of tensorNames) {
        const loc = this.tensorLocations.get(name);
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
        type: getGroupType(groupId, this.modelType),
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

  /**
   * Reset packer state for reuse.
   */
  reset(): void {
    this.currentShardIndex = 0;
    this.currentShardData = [];
    this.currentShardSize = 0;
    this.shards = [];
    this.tensorLocations.clear();
    this.groupTensorMap.clear();
    this.totalSize = 0;
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Sort tensors by component group for optimal shard packing.
 * Groups: embed → layers (in order) → head
 */
export function sortTensorsByGroup(
  tensors: TensorInfoSchema[],
  modelType: ModelType = 'transformer'
): TensorInfoSchema[] {
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

/**
 * Estimate shard count for a set of tensors.
 */
export function estimateShardCount(
  tensors: TensorInfoSchema[],
  shardSize: number = SHARD_SIZE
): number {
  const totalSize = tensors.reduce((sum, t) => sum + t.size, 0);
  return Math.ceil(totalSize / shardSize);
}
