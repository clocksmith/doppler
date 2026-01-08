/**
 * Partitioned buffer pools for multi-model execution.
 *
 * @module gpu/partitioned-buffer-pool
 */

import { BufferPool } from './buffer-pool.js';

/**
 * @typedef {Object} PartitionConfig
 * @property {string} id
 */

export class PartitionedBufferPool {
  /** @type {BufferPool} */
  #sharedPool;
  /** @type {Map<string, BufferPool>} */
  #expertPools;

  /**
   * @param {PartitionConfig[]} partitions
   */
  constructor(partitions) {
    this.#sharedPool = new BufferPool(false);
    this.#expertPools = new Map();
    for (const partition of partitions) {
      this.#expertPools.set(partition.id, new BufferPool(false));
    }
  }

  /**
   * @param {string} partitionId
   * @param {number} size
   * @param {GPUBufferUsageFlags} usage
   * @param {string} [label]
   * @returns {GPUBuffer}
   */
  acquire(
    partitionId,
    size,
    usage,
    label
  ) {
    const pool = this.#expertPools.get(partitionId) || this.#sharedPool;
    return pool.acquire(size, usage, label);
  }

  /**
   * @param {string} partitionId
   * @param {GPUBuffer} buffer
   * @returns {void}
   */
  release(partitionId, buffer) {
    const pool = this.#expertPools.get(partitionId) || this.#sharedPool;
    pool.release(buffer);
  }

  /**
   * @returns {BufferPool}
   */
  getSharedPool() {
    return this.#sharedPool;
  }

  /**
   * @param {string} partitionId
   * @returns {BufferPool | null}
   */
  getExpertPool(partitionId) {
    return this.#expertPools.get(partitionId) || null;
  }
}
