/**
 * Partitioned buffer pools for multi-model execution.
 *
 * @module gpu/partitioned-buffer-pool
 */

import { BufferPool } from './buffer-pool.js';

export interface PartitionConfig {
  id: string;
}

export class PartitionedBufferPool {
  private sharedPool: BufferPool;
  private expertPools: Map<string, BufferPool>;

  constructor(partitions: PartitionConfig[]) {
    this.sharedPool = new BufferPool(false);
    this.expertPools = new Map();
    for (const partition of partitions) {
      this.expertPools.set(partition.id, new BufferPool(false));
    }
  }

  acquire(
    partitionId: string,
    size: number,
    usage: GPUBufferUsageFlags,
    label?: string
  ): GPUBuffer {
    const pool = this.expertPools.get(partitionId) || this.sharedPool;
    return pool.acquire(size, usage, label);
  }

  release(partitionId: string, buffer: GPUBuffer): void {
    const pool = this.expertPools.get(partitionId) || this.sharedPool;
    pool.release(buffer);
  }

  getSharedPool(): BufferPool {
    return this.sharedPool;
  }

  getExpertPool(partitionId: string): BufferPool | null {
    return this.expertPools.get(partitionId) || null;
  }
}
