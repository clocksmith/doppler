/**
 * Virtual GPU representing one emulated GPU's memory space.
 * @module simulator/virtual-gpu
 */

import { log } from '../debug/index.js';
import { getBufferPool } from '../memory/buffer-pool.js';
import { MODULE, DEFAULT_VRAM_BUDGET_BYTES, generateBufferId } from './virtual-utils.js';

/**
 * Virtual GPU representing one emulated GPU's memory space
 */
export class VirtualGPU {
  /**
   * @param {number} index - GPU index in cluster
   * @param {import('../config/schema/emulation.schema.js').EmulatedGPUSpec} spec - GPU specification
   * @param {string} opfsRootPath - Root path for OPFS storage
   */
  constructor(index, spec, opfsRootPath) {
    /** @type {number} */
    this.index = index;
    /** @type {import('../config/schema/emulation.schema.js').EmulatedGPUSpec} */
    this.spec = spec;
    /** @type {string} */
    this.opfsPath = `${opfsRootPath}/gpu${index}`;

    /** @type {Map<string, import('./virtual-device.js').VirtualBufferMetadata>} */
    this._buffers = new Map();

    /** @type {Map<string, ArrayBuffer>} RAM-staged buffers */
    this._ramBuffers = new Map();

    /** @type {Map<string, GPUBuffer>} Actual VRAM buffers */
    this._vramBuffers = new Map();

    /** @type {FileSystemDirectoryHandle|null} */
    this._opfsDir = null;

    /** @type {number} Current VRAM budget */
    this._vramBudget = DEFAULT_VRAM_BUDGET_BYTES;

    /** @type {number} Current VRAM usage */
    this._vramUsed = 0;

    /** @type {number} Current RAM usage */
    this._ramUsed = 0;

    /** @type {number} Current OPFS usage */
    this._opfsUsed = 0;

    /** @type {boolean} */
    this._initialized = false;
  }

  /**
   * Initialize OPFS storage for this GPU
   */
  async initialize() {
    if (this._initialized) return;

    if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) {
      log.warn(MODULE, `OPFS not available for GPU ${this.index} (navigator.storage.getDirectory missing)`);
      this._initialized = true;
      return;
    }

    try {
      // Get OPFS root
      const opfsRoot = await navigator.storage.getDirectory();

      // Create nested directory structure
      const parts = this.opfsPath.split('/').filter(p => p);
      let current = opfsRoot;
      for (const part of parts) {
        current = await current.getDirectoryHandle(part, { create: true });
      }
      this._opfsDir = current;

      this._initialized = true;
      log.verbose(MODULE, `VirtualGPU ${this.index} initialized at ${this.opfsPath}`);
    } catch (err) {
      log.warn(MODULE, `Failed to initialize OPFS for GPU ${this.index}: ${err.message}`);
      // Continue without OPFS - will use RAM only
      this._initialized = true;
    }
  }

  /**
   * Allocate a virtual buffer on this GPU
   * @param {number} sizeBytes - Size in bytes
   * @param {string} [label] - Optional label for debugging
   * @returns {Promise<import('./virtual-device.js').VirtualBufferRef>}
   */
  async allocate(sizeBytes, label) {
    await this.initialize();

    const id = generateBufferId();
    const now = Date.now();

    // Determine initial location based on size and available VRAM
    let location = 'ram';
    if (sizeBytes <= this._vramBudget - this._vramUsed) {
      location = 'vram';
    }

    /** @type {import('./virtual-device.js').VirtualBufferMetadata} */
    const metadata = {
      id,
      sizeBytes,
      location,
      gpuIndex: this.index,
      label,
      lastAccessTime: now,
      accessCount: 0,
      pinned: false,
    };

    this._buffers.set(id, metadata);

    // Allocate in the appropriate tier
    if (location === 'vram') {
      const pool = getBufferPool();
      const gpuBuffer = pool.acquire(sizeBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label);
      this._vramBuffers.set(id, gpuBuffer);
      this._vramUsed += sizeBytes;
    } else {
      const arrayBuffer = new ArrayBuffer(sizeBytes);
      this._ramBuffers.set(id, arrayBuffer);
      this._ramUsed += sizeBytes;
    }

    log.verbose(MODULE, `GPU ${this.index}: Allocated ${label || id} (${sizeBytes} bytes) in ${location}`);

    return this._createBufferRef(metadata);
  }

  /**
   * Create a buffer reference object
   * @param {import('./virtual-device.js').VirtualBufferMetadata} metadata
   * @returns {import('./virtual-device.js').VirtualBufferRef}
   */
  _createBufferRef(metadata) {
    const gpu = this;
    return {
      metadata,
      async getData() {
        return gpu.read(metadata.id);
      },
      async getGPUBuffer() {
        return gpu._promoteToVram(metadata.id);
      },
      release() {
        gpu.free(metadata.id);
      },
    };
  }

  /**
   * Promote a buffer to VRAM
   * @param {string} bufferId - Buffer ID
   * @returns {Promise<GPUBuffer>}
   */
  async _promoteToVram(bufferId) {
    const metadata = this._buffers.get(bufferId);
    if (!metadata) {
      throw new Error(`Buffer ${bufferId} not found on GPU ${this.index}`);
    }

    // Already in VRAM
    if (metadata.location === 'vram') {
      metadata.lastAccessTime = Date.now();
      metadata.accessCount++;
      return this._vramBuffers.get(bufferId);
    }

    // Need to promote from RAM or OPFS
    let data;
    if (metadata.location === 'ram') {
      data = this._ramBuffers.get(bufferId);
    } else {
      data = await this._readFromOpfs(bufferId);
    }

    // Evict if needed
    const needed = metadata.sizeBytes;
    if (this._vramUsed + needed > this._vramBudget) {
      await this.evictLRU(needed);
    }

    // Allocate GPU buffer and upload
    const pool = getBufferPool();
    const gpuBuffer = pool.acquire(
      metadata.sizeBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      metadata.label
    );
    pool.uploadData(gpuBuffer, data);

    // Update tracking
    if (metadata.location === 'ram') {
      this._ramBuffers.delete(bufferId);
      this._ramUsed -= metadata.sizeBytes;
    } else {
      this._opfsUsed -= metadata.sizeBytes;
    }

    this._vramBuffers.set(bufferId, gpuBuffer);
    this._vramUsed += metadata.sizeBytes;
    metadata.location = 'vram';
    metadata.lastAccessTime = Date.now();
    metadata.accessCount++;

    return gpuBuffer;
  }

  /**
   * Write data to a buffer
   * @param {string} bufferId - Buffer ID
   * @param {ArrayBuffer} data - Data to write
   * @param {number} [offset=0] - Offset in bytes
   */
  async write(bufferId, data, offset = 0) {
    const metadata = this._buffers.get(bufferId);
    if (!metadata) {
      throw new Error(`Buffer ${bufferId} not found on GPU ${this.index}`);
    }

    metadata.lastAccessTime = Date.now();
    metadata.accessCount++;

    if (metadata.location === 'vram') {
      const pool = getBufferPool();
      const gpuBuffer = this._vramBuffers.get(bufferId);
      pool.uploadData(gpuBuffer, data, offset);
    } else if (metadata.location === 'ram') {
      const existing = this._ramBuffers.get(bufferId);
      new Uint8Array(existing).set(new Uint8Array(data), offset);
    } else {
      // OPFS - read, modify, write back
      const existing = await this._readFromOpfs(bufferId);
      new Uint8Array(existing).set(new Uint8Array(data), offset);
      await this._writeToOpfs(bufferId, existing);
    }
  }

  /**
   * Read data from a buffer
   * @param {string} bufferId - Buffer ID
   * @param {number} [offset=0] - Offset in bytes
   * @param {number} [length] - Length in bytes
   * @returns {Promise<ArrayBuffer>}
   */
  async read(bufferId, offset = 0, length) {
    const metadata = this._buffers.get(bufferId);
    if (!metadata) {
      throw new Error(`Buffer ${bufferId} not found on GPU ${this.index}`);
    }

    metadata.lastAccessTime = Date.now();
    metadata.accessCount++;

    const readLength = length ?? metadata.sizeBytes - offset;

    if (metadata.location === 'vram') {
      const pool = getBufferPool();
      const gpuBuffer = this._vramBuffers.get(bufferId);
      const fullData = await pool.readBuffer(gpuBuffer, metadata.sizeBytes);
      return fullData.slice(offset, offset + readLength);
    } else if (metadata.location === 'ram') {
      const data = this._ramBuffers.get(bufferId);
      return data.slice(offset, offset + readLength);
    } else {
      const data = await this._readFromOpfs(bufferId);
      return data.slice(offset, offset + readLength);
    }
  }

  /**
   * Free a virtual buffer
   * @param {string} bufferId - Buffer ID
   */
  async free(bufferId) {
    const metadata = this._buffers.get(bufferId);
    if (!metadata) return;

    if (metadata.location === 'vram') {
      const pool = getBufferPool();
      const gpuBuffer = this._vramBuffers.get(bufferId);
      if (gpuBuffer) {
        pool.release(gpuBuffer);
        this._vramBuffers.delete(bufferId);
        this._vramUsed -= metadata.sizeBytes;
      }
    } else if (metadata.location === 'ram') {
      this._ramBuffers.delete(bufferId);
      this._ramUsed -= metadata.sizeBytes;
    } else {
      await this._deleteFromOpfs(bufferId);
      this._opfsUsed -= metadata.sizeBytes;
    }

    this._buffers.delete(bufferId);
    log.verbose(MODULE, `GPU ${this.index}: Freed ${metadata.label || bufferId}`);
  }

  /**
   * Pin buffer in VRAM
   * @param {string} bufferId - Buffer ID
   */
  async pin(bufferId) {
    const metadata = this._buffers.get(bufferId);
    if (!metadata) {
      throw new Error(`Buffer ${bufferId} not found on GPU ${this.index}`);
    }

    // Promote to VRAM if not already there
    if (metadata.location !== 'vram') {
      await this._promoteToVram(bufferId);
    }

    metadata.pinned = true;
  }

  /**
   * Unpin buffer
   * @param {string} bufferId - Buffer ID
   */
  async unpin(bufferId) {
    const metadata = this._buffers.get(bufferId);
    if (metadata) {
      metadata.pinned = false;
    }
  }

  /**
   * Get buffer metadata
   * @param {string} bufferId - Buffer ID
   * @returns {import('./virtual-device.js').VirtualBufferMetadata|null}
   */
  getBufferInfo(bufferId) {
    return this._buffers.get(bufferId) || null;
  }

  /**
   * Get all buffer IDs
   * @returns {string[]}
   */
  listBuffers() {
    return Array.from(this._buffers.keys());
  }

  /**
   * Get memory statistics for this GPU
   * @returns {import('./virtual-device.js').VirtualGPUMemoryStats}
   */
  getMemoryStats() {
    let pinnedCount = 0;
    for (const meta of this._buffers.values()) {
      if (meta.pinned) pinnedCount++;
    }

    return {
      gpuIndex: this.index,
      totalVramBytes: this.spec.vramBytes,
      allocatedBytes: this._vramUsed + this._ramUsed + this._opfsUsed,
      vramUsedBytes: this._vramUsed,
      ramUsedBytes: this._ramUsed,
      opfsUsedBytes: this._opfsUsed,
      bufferCount: this._buffers.size,
      pinnedBufferCount: pinnedCount,
    };
  }

  /**
   * Evict least recently used buffers to free VRAM
   * @param {number} bytesNeeded - Bytes to free
   * @returns {Promise<number>} Bytes actually freed
   */
  async evictLRU(bytesNeeded) {
    // Collect evictable buffers (in VRAM, not pinned)
    const evictable = [];
    for (const [id, meta] of this._buffers) {
      if (meta.location === 'vram' && !meta.pinned) {
        evictable.push({ id, meta });
      }
    }

    // Sort by last access time (oldest first)
    evictable.sort((a, b) => a.meta.lastAccessTime - b.meta.lastAccessTime);

    let freed = 0;
    for (const { id, meta } of evictable) {
      if (freed >= bytesNeeded) break;

      // Demote to RAM
      const pool = getBufferPool();
      const gpuBuffer = this._vramBuffers.get(id);
      const data = await pool.readBuffer(gpuBuffer, meta.sizeBytes);

      // Store in RAM
      this._ramBuffers.set(id, data);
      this._ramUsed += meta.sizeBytes;

      // Release GPU buffer
      pool.release(gpuBuffer);
      this._vramBuffers.delete(id);
      this._vramUsed -= meta.sizeBytes;

      meta.location = 'ram';
      freed += meta.sizeBytes;

      log.verbose(MODULE, `GPU ${this.index}: Evicted ${meta.label || id} to RAM`);
    }

    return freed;
  }

  /**
   * Read buffer data from OPFS
   * @param {string} bufferId - Buffer ID
   * @returns {Promise<ArrayBuffer>}
   */
  async _readFromOpfs(bufferId) {
    if (!this._opfsDir) {
      throw new Error('OPFS not initialized');
    }

    const fileHandle = await this._opfsDir.getFileHandle(`${bufferId}.bin`);
    const file = await fileHandle.getFile();
    return file.arrayBuffer();
  }

  /**
   * Write buffer data to OPFS
   * @param {string} bufferId - Buffer ID
   * @param {ArrayBuffer} data - Data to write
   */
  async _writeToOpfs(bufferId, data) {
    if (!this._opfsDir) {
      throw new Error('OPFS not initialized');
    }

    const fileHandle = await this._opfsDir.getFileHandle(`${bufferId}.bin`, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(data);
    await writable.close();
  }

  /**
   * Delete buffer from OPFS
   * @param {string} bufferId - Buffer ID
   */
  async _deleteFromOpfs(bufferId) {
    if (!this._opfsDir) return;

    try {
      await this._opfsDir.removeEntry(`${bufferId}.bin`);
    } catch (err) {
      // Ignore if file doesn't exist
    }
  }

  /**
   * Destroy and clean up all resources
   */
  async destroy() {
    const pool = getBufferPool();

    // Release all VRAM buffers
    for (const gpuBuffer of this._vramBuffers.values()) {
      pool.release(gpuBuffer);
    }

    this._vramBuffers.clear();
    this._ramBuffers.clear();
    this._buffers.clear();
    this._vramUsed = 0;
    this._ramUsed = 0;
    this._opfsUsed = 0;
    this._initialized = false;

    log.verbose(MODULE, `VirtualGPU ${this.index} destroyed`);
  }
}
