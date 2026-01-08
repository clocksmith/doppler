/**
 * shard-packer.ts - Platform-agnostic Shard Packing
 *
 * Core shard packing logic shared between Node.js CLI and browser converters.
 * Uses an I/O adapter interface for platform-specific operations.
 *
 * @module converter/shard-packer
 */

import type {
  ModelType,
  HashAlgorithm,
  ComponentGroupSchema,
  TensorInfoSchema,
  ShardSchema,
} from '../config/schema/index.js';

/**
 * Platform-specific I/O adapter interface.
 */
export interface ShardIO {
  writeShard(index: number, data: Uint8Array): Promise<string>;
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
  onProgress?: (current: number, total: number, tensorName: string) => void;
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
  getData: () => Promise<Uint8Array>;
}

/**
 * Platform-agnostic shard packer.
 */
export declare class ShardPacker {
  constructor(io: ShardIO, options?: ShardPackerOptions);

  pack(
    tensors: PackerTensorInput[],
    options?: { onProgress?: ShardPackerOptions['onProgress']; signal?: AbortSignal }
  ): Promise<ShardPackerResult>;

  reset(): void;
}

/**
 * Sort tensors by component group for optimal shard packing.
 */
export declare function sortTensorsByGroup(
  tensors: TensorInfoSchema[],
  modelType?: ModelType
): TensorInfoSchema[];

/**
 * Estimate shard count for a set of tensors.
 */
export declare function estimateShardCount(
  tensors: TensorInfoSchema[],
  shardSize?: number
): number;
