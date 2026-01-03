

import { BufferPool } from '../memory/buffer-pool.js';



export class PartitionedBufferPool {
  
  #sharedPool;
  
  #expertPools;

  
  constructor(partitions) {
    this.#sharedPool = new BufferPool(false);
    this.#expertPools = new Map();
    for (const partition of partitions) {
      this.#expertPools.set(partition.id, new BufferPool(false));
    }
  }

  
  acquire(
    partitionId,
    size,
    usage,
    label
  ) {
    const pool = this.#expertPools.get(partitionId) || this.#sharedPool;
    return pool.acquire(size, usage, label);
  }

  
  release(partitionId, buffer) {
    const pool = this.#expertPools.get(partitionId) || this.#sharedPool;
    pool.release(buffer);
  }

  
  getSharedPool() {
    return this.#sharedPool;
  }

  
  getExpertPool(partitionId) {
    return this.#expertPools.get(partitionId) || null;
  }
}
