

import { BufferPool } from '../memory/buffer-pool.js';
import { getRuntimeConfig } from '../config/runtime.js';



export class PartitionedBufferPool {
  
  #sharedPool;
  
  #expertPools;

  #bufferOwners;

  
  constructor(partitions, schemaConfig = getRuntimeConfig().shared.bufferPool) {
    this.#sharedPool = new BufferPool(false, schemaConfig);
    this.#expertPools = new Map();
    this.#bufferOwners = new WeakMap();
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
    const buffer = pool.acquire(size, usage, label);
    this.#bufferOwners.set(buffer, pool);
    return buffer;
  }

  
  release(partitionId, buffer) {
    const pool = this.#bufferOwners.get(buffer)
      || this.#expertPools.get(partitionId)
      || this.#sharedPool;
    this.#bufferOwners.delete(buffer);
    pool.release(buffer);
  }

  
  getSharedPool() {
    return this.#sharedPool;
  }

  
  getExpertPool(partitionId) {
    return this.#expertPools.get(partitionId) || null;
  }
}
