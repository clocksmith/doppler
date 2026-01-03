/**
 * Partitioned buffer pools for multi-model execution.
 *
 * @module gpu/partitioned-buffer-pool
 */

import { BufferPool } from '../memory/buffer-pool.js';

export interface PartitionConfig {
  id: string;
}

export declare class PartitionedBufferPool {
  constructor(partitions: PartitionConfig[]);

  acquire(
    partitionId: string,
    size: number,
    usage: GPUBufferUsageFlags,
    label?: string
  ): GPUBuffer;

  release(partitionId: string, buffer: GPUBuffer): void;

  getSharedPool(): BufferPool;

  getExpertPool(partitionId: string): BufferPool | null;
}
