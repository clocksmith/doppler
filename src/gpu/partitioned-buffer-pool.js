

import { BufferPool } from '../memory/buffer-pool.js';
import { getRuntimeConfig } from '../config/runtime.js';



export class PartitionedBufferPool {
  
  #sharedPool;
  
  #expertPools;

  
  constructor(partitions, schemaConfig = getRuntimeConfig().shared.bufferPool) {
    this.#sharedPool = new BufferPool(false, schemaConfig);
    this.#expertPools = new Map();
    for (const partition of partitions) {
      this.#expertPools.set(partition.id, new BufferPool(false, schemaConfig));
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
