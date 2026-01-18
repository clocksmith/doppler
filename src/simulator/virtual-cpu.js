/**
 * Virtual CPU representing Grace CPU memory space.
 * @module simulator/virtual-cpu
 */

import { log } from '../debug/index.js';
import { MODULE, generateBufferId } from './virtual-utils.js';

/**
 * Virtual CPU representing Grace CPU memory space
 */
export class VirtualCPU {
  /**
   * @param {number} index - CPU index
   * @param {import('../config/schema/emulation.schema.js').EmulatedCPUSpec} spec - CPU spec
   * @param {string} opfsRootPath - Root path for OPFS storage
   */
  constructor(index, spec, opfsRootPath) {
    this.index = index;
    this.spec = spec;
    this.opfsPath = `${opfsRootPath}/cpu${index}`;

    /** @type {Map<string, {sizeBytes: number, label?: string}>} */
    this._buffers = new Map();

    /** @type {Map<string, ArrayBuffer>} */
    this._ramBuffers = new Map();

    /** @type {number} */
    this._allocated = 0;

    /** @type {boolean} */
    this._initialized = false;
  }

  /**
   * Initialize storage
   */
  async initialize() {
    if (this._initialized) return;
    this._initialized = true;
    log.verbose(MODULE, `VirtualCPU ${this.index} initialized`);
  }

  /**
   * Allocate a buffer in CPU memory space
   * @param {number} sizeBytes - Size in bytes
   * @param {string} [label] - Optional label
   * @returns {Promise<import('./virtual-device.js').VirtualBufferRef>}
   */
  async allocate(sizeBytes, label) {
    await this.initialize();

    const id = generateBufferId();
    const buffer = new ArrayBuffer(sizeBytes);

    this._buffers.set(id, { sizeBytes, label });
    this._ramBuffers.set(id, buffer);
    this._allocated += sizeBytes;

    log.verbose(MODULE, `CPU ${this.index}: Allocated ${label || id} (${sizeBytes} bytes)`);

    const cpu = this;
    return {
      metadata: {
        id,
        sizeBytes,
        location: 'ram',
        gpuIndex: -1,
        label,
        lastAccessTime: Date.now(),
        accessCount: 0,
        pinned: false,
      },
      async getData() {
        return cpu.read(id);
      },
      async getGPUBuffer() {
        throw new Error('CPU buffers cannot be promoted to GPU');
      },
      release() {
        cpu.free(id);
      },
    };
  }

  /**
   * Write data to a buffer
   * @param {string} bufferId - Buffer ID
   * @param {ArrayBuffer} data - Data to write
   */
  async write(bufferId, data) {
    const buffer = this._ramBuffers.get(bufferId);
    if (!buffer) {
      throw new Error(`Buffer ${bufferId} not found on CPU ${this.index}`);
    }
    new Uint8Array(buffer).set(new Uint8Array(data));
  }

  /**
   * Read data from a buffer
   * @param {string} bufferId - Buffer ID
   * @returns {Promise<ArrayBuffer>}
   */
  async read(bufferId) {
    const buffer = this._ramBuffers.get(bufferId);
    if (!buffer) {
      throw new Error(`Buffer ${bufferId} not found on CPU ${this.index}`);
    }
    return buffer.slice(0);
  }

  /**
   * Free a buffer
   * @param {string} bufferId - Buffer ID
   */
  async free(bufferId) {
    const meta = this._buffers.get(bufferId);
    if (!meta) return;

    this._ramBuffers.delete(bufferId);
    this._buffers.delete(bufferId);
    this._allocated -= meta.sizeBytes;

    log.verbose(MODULE, `CPU ${this.index}: Freed ${meta.label || bufferId}`);
  }

  /**
   * Get memory statistics for this CPU
   * @returns {import('./virtual-device.js').VirtualCPUMemoryStats}
   */
  getMemoryStats() {
    return {
      cpuIndex: this.index,
      totalMemoryBytes: this.spec.memoryBytes,
      allocatedBytes: this._allocated,
      bufferCount: this._buffers.size,
    };
  }

  /**
   * Destroy and clean up
   */
  async destroy() {
    this._ramBuffers.clear();
    this._buffers.clear();
    this._allocated = 0;
    this._initialized = false;

    log.verbose(MODULE, `VirtualCPU ${this.index} destroyed`);
  }
}
