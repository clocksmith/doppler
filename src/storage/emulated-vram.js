
import { log } from '../debug/index.js';
import { getBufferPool } from '../memory/buffer-pool.js';

// =============================================================================
// Constants
// =============================================================================

const MODULE = 'EmulatedVRAM';

const DEFAULT_VRAM_BUDGET = 2 * 1024 * 1024 * 1024;

const DEFAULT_RAM_BUDGET = 8 * 1024 * 1024 * 1024;

let chunkIdCounter = 0;

// =============================================================================
// Emulated VRAM Store
// =============================================================================

export class EmulatedVramStore {
    constructor(rootPath, vramBudgetBytes = DEFAULT_VRAM_BUDGET, ramBudgetBytes = DEFAULT_RAM_BUDGET) {
        this.rootPath = rootPath;

        this.vramBudgetBytes = vramBudgetBytes;

        this.ramBudgetBytes = ramBudgetBytes;

        this._chunks = new Map();

        this._ramStore = new Map();

        this._vramStore = new Map();

        this._partitions = new Map();

        this._opfsDirs = new Map();

        this._vramUsed = 0;

        this._ramUsed = 0;

        this._opfsUsed = 0;

        this._evictionCount = 0;

        this._totalBytesEvicted = 0;

        this._rootDir = null;

        this._initialized = false;
  }

    async initialize() {
    if (this._initialized) return;

    if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) {
      log.warn(MODULE, 'OPFS not available (navigator.storage.getDirectory missing)');
      this._initialized = true;
      return;
    }

    try {
      const opfsRoot = await navigator.storage.getDirectory();

      // Create root directory
      const parts = this.rootPath.split('/').filter(p => p);
      let current = opfsRoot;
      for (const part of parts) {
        current = await current.getDirectoryHandle(part, { create: true });
      }
      this._rootDir = current;

      this._initialized = true;
      log.verbose(MODULE, `Initialized at ${this.rootPath}`);
    } catch (err) {
      log.warn(MODULE, `OPFS init failed: ${err.message}`);
      this._initialized = true; // Continue without OPFS
    }
  }

    async createPartition(config) {
    await this.initialize();

    if (this._partitions.has(config.name)) {
      log.warn(MODULE, `Partition ${config.name} already exists`);
      return;
    }

    this._partitions.set(config.name, config);

    // Create OPFS directory for partition
    if (this._rootDir) {
      try {
        const partDir = await this._rootDir.getDirectoryHandle(config.name, { create: true });
        this._opfsDirs.set(config.name, partDir);
      } catch (err) {
        log.warn(MODULE, `Failed to create OPFS partition ${config.name}: ${err.message}`);
      }
    }

    log.verbose(MODULE, `Created partition ${config.name} (max: ${config.maxBytes} bytes)`);
  }

    async allocate(partition, sizeBytes, label) {
    await this.initialize();

    if (!this._partitions.has(partition)) {
      throw new Error(`Partition ${partition} does not exist`);
    }

    const id = `chunk_${Date.now()}_${chunkIdCounter++}`;
    const now = Date.now();

    // Determine initial tier based on available space
    let tier = 'opfs';
    if (this._vramUsed + sizeBytes <= this.vramBudgetBytes) {
      tier = 'vram';
    } else if (this._ramUsed + sizeBytes <= this.ramBudgetBytes) {
      tier = 'ram';
    }

        const chunk = {
      id,
      sizeBytes,
      tier,
      partition,
      locked: false,
      lastAccessMs: now,
      accessCount: 0,
    };

    this._chunks.set(id, chunk);

    // Allocate in appropriate tier
    if (tier === 'vram') {
      const pool = getBufferPool();
      const gpuBuffer = pool.acquire(
        sizeBytes,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        label
      );
      this._vramStore.set(id, gpuBuffer);
      this._vramUsed += sizeBytes;
    } else if (tier === 'ram') {
      this._ramStore.set(id, new ArrayBuffer(sizeBytes));
      this._ramUsed += sizeBytes;
    } else {
      // OPFS - create empty file
      await this._writeToOpfs(id, partition, new ArrayBuffer(sizeBytes));
      this._opfsUsed += sizeBytes;
    }

    log.verbose(MODULE, `Allocated ${label || id} (${sizeBytes} bytes) in ${tier}`);
    return id;
  }

    async write(chunkId, data, offset = 0) {
    const start = performance.now();
    const chunk = this._chunks.get(chunkId);
    if (!chunk) {
      throw new Error(`Chunk ${chunkId} not found`);
    }

    chunk.lastAccessMs = Date.now();
    chunk.accessCount++;

    if (chunk.tier === 'vram') {
      const pool = getBufferPool();
      const gpuBuffer = this._vramStore.get(chunkId);
      pool.uploadData(gpuBuffer, data, offset);
    } else if (chunk.tier === 'ram') {
      const buffer = this._ramStore.get(chunkId);
      new Uint8Array(buffer).set(new Uint8Array(data), offset);
    } else {
      // OPFS - read, modify, write
      const existing = await this._readFromOpfs(chunkId, chunk.partition);
      new Uint8Array(existing).set(new Uint8Array(data), offset);
      await this._writeToOpfs(chunkId, chunk.partition, existing);
    }

    return {
      toTier: chunk.tier,
      writeTimeMs: performance.now() - start,
    };
  }

    async read(chunkId, offset = 0, length) {
    const start = performance.now();
    const chunk = this._chunks.get(chunkId);
    if (!chunk) {
      throw new Error(`Chunk ${chunkId} not found`);
    }

    chunk.lastAccessMs = Date.now();
    chunk.accessCount++;

    const readLen = length ?? chunk.sizeBytes - offset;
    let data;

    if (chunk.tier === 'vram') {
      const pool = getBufferPool();
      const gpuBuffer = this._vramStore.get(chunkId);
      const fullData = await pool.readBuffer(gpuBuffer, chunk.sizeBytes);
      data = fullData.slice(offset, offset + readLen);
    } else if (chunk.tier === 'ram') {
      const buffer = this._ramStore.get(chunkId);
      data = buffer.slice(offset, offset + readLen);
    } else {
      const buffer = await this._readFromOpfs(chunkId, chunk.partition);
      data = buffer.slice(offset, offset + readLen);
    }

    return {
      data,
      fromTier: chunk.tier,
      readTimeMs: performance.now() - start,
    };
  }

    async free(chunkId) {
    const chunk = this._chunks.get(chunkId);
    if (!chunk) return;

    if (chunk.tier === 'vram') {
      const pool = getBufferPool();
      const gpuBuffer = this._vramStore.get(chunkId);
      if (gpuBuffer) {
        pool.release(gpuBuffer);
        this._vramStore.delete(chunkId);
        this._vramUsed -= chunk.sizeBytes;
      }
    } else if (chunk.tier === 'ram') {
      this._ramStore.delete(chunkId);
      this._ramUsed -= chunk.sizeBytes;
    } else {
      await this._deleteFromOpfs(chunkId, chunk.partition);
      this._opfsUsed -= chunk.sizeBytes;
    }

    this._chunks.delete(chunkId);
    log.verbose(MODULE, `Freed chunk ${chunkId}`);
  }

    async lock(chunkId) {
    const chunk = this._chunks.get(chunkId);
    if (chunk) {
      chunk.locked = true;
    }
  }

    async unlock(chunkId) {
    const chunk = this._chunks.get(chunkId);
    if (chunk) {
      chunk.locked = false;
    }
  }

    async promote(chunkId, targetTier) {
    const chunk = this._chunks.get(chunkId);
    if (!chunk) {
      throw new Error(`Chunk ${chunkId} not found`);
    }

    const tierOrder = { opfs: 0, ram: 1, vram: 2 };
    if (tierOrder[targetTier] <= tierOrder[chunk.tier]) {
      return; // Already at or above target tier
    }

    // Read current data
    const result = await this.read(chunkId);
    const data = result.data;

    // Free from current tier
    await this._freeFromTier(chunkId, chunk);

    // Evict if needed
    if (targetTier === 'vram' && this._vramUsed + chunk.sizeBytes > this.vramBudgetBytes) {
      await this.evict('vram', chunk.sizeBytes);
    } else if (targetTier === 'ram' && this._ramUsed + chunk.sizeBytes > this.ramBudgetBytes) {
      await this.evict('ram', chunk.sizeBytes);
    }

    // Write to target tier
    chunk.tier = targetTier;
    if (targetTier === 'vram') {
      const pool = getBufferPool();
      const gpuBuffer = pool.acquire(
        chunk.sizeBytes,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      );
      pool.uploadData(gpuBuffer, data);
      this._vramStore.set(chunkId, gpuBuffer);
      this._vramUsed += chunk.sizeBytes;
    } else if (targetTier === 'ram') {
      const buffer = new ArrayBuffer(chunk.sizeBytes);
      new Uint8Array(buffer).set(new Uint8Array(data));
      this._ramStore.set(chunkId, buffer);
      this._ramUsed += chunk.sizeBytes;
    }

    chunk.lastAccessMs = Date.now();
    log.verbose(MODULE, `Promoted ${chunkId} to ${targetTier}`);
  }

    async demote(chunkId, targetTier) {
    const chunk = this._chunks.get(chunkId);
    if (!chunk) {
      throw new Error(`Chunk ${chunkId} not found`);
    }

    if (chunk.locked) {
      throw new Error(`Cannot demote locked chunk ${chunkId}`);
    }

    const tierOrder = { opfs: 0, ram: 1, vram: 2 };
    if (tierOrder[targetTier] >= tierOrder[chunk.tier]) {
      return; // Already at or below target tier
    }

    // Read current data
    const result = await this.read(chunkId);
    const data = result.data;

    // Free from current tier
    await this._freeFromTier(chunkId, chunk);

    // Write to target tier
    chunk.tier = targetTier;
    if (targetTier === 'ram') {
      const buffer = new ArrayBuffer(chunk.sizeBytes);
      new Uint8Array(buffer).set(new Uint8Array(data));
      this._ramStore.set(chunkId, buffer);
      this._ramUsed += chunk.sizeBytes;
    } else if (targetTier === 'opfs') {
      await this._writeToOpfs(chunkId, chunk.partition, data);
      this._opfsUsed += chunk.sizeBytes;
    }

    log.verbose(MODULE, `Demoted ${chunkId} to ${targetTier}`);
  }

    async _freeFromTier(chunkId, chunk) {
    if (chunk.tier === 'vram') {
      const pool = getBufferPool();
      const gpuBuffer = this._vramStore.get(chunkId);
      if (gpuBuffer) {
        pool.release(gpuBuffer);
        this._vramStore.delete(chunkId);
        this._vramUsed -= chunk.sizeBytes;
      }
    } else if (chunk.tier === 'ram') {
      this._ramStore.delete(chunkId);
      this._ramUsed -= chunk.sizeBytes;
    } else {
      await this._deleteFromOpfs(chunkId, chunk.partition);
      this._opfsUsed -= chunk.sizeBytes;
    }
  }

    getChunkInfo(chunkId) {
    return this._chunks.get(chunkId) || null;
  }

    listChunks(partition) {
    const result = [];
    for (const [id, chunk] of this._chunks) {
      if (chunk.partition === partition) {
        result.push(id);
      }
    }
    return result;
  }

    getStats() {
    const partitionStats = [];

    for (const [name, config] of this._partitions) {
      let allocated = 0;
      let count = 0;
      for (const chunk of this._chunks.values()) {
        if (chunk.partition === name) {
          allocated += chunk.sizeBytes;
          count++;
        }
      }
      partitionStats.push({
        name,
        allocatedBytes: allocated,
        chunkCount: count,
      });
    }

    return {
      totalChunks: this._chunks.size,
      totalAllocatedBytes: this._vramUsed + this._ramUsed + this._opfsUsed,
      vramUsedBytes: this._vramUsed,
      ramUsedBytes: this._ramUsed,
      opfsUsedBytes: this._opfsUsed,
      vramBudgetBytes: this.vramBudgetBytes,
      ramBudgetBytes: this.ramBudgetBytes,
      evictionCount: this._evictionCount,
      totalBytesEvicted: this._totalBytesEvicted,
      partitionStats,
    };
  }

    async evict(tier, bytesNeeded) {
    // Collect evictable chunks
    const evictable = [];
    for (const [id, chunk] of this._chunks) {
      if (chunk.tier === tier && !chunk.locked) {
        evictable.push({ id, chunk });
      }
    }

    // Sort by LRU
    evictable.sort((a, b) => a.chunk.lastAccessMs - b.chunk.lastAccessMs);

    let freed = 0;
    const targetTier = tier === 'vram' ? 'ram' : 'opfs';

    for (const { id, chunk } of evictable) {
      if (freed >= bytesNeeded) break;

      await this.demote(id, targetTier);
      freed += chunk.sizeBytes;
      this._evictionCount++;
      this._totalBytesEvicted += chunk.sizeBytes;
    }

    return freed;
  }

    async _readFromOpfs(chunkId, partition) {
    const dir = this._opfsDirs.get(partition);
    if (!dir) {
      throw new Error(`OPFS partition ${partition} not found`);
    }

    const fileHandle = await dir.getFileHandle(`${chunkId}.bin`);
    const file = await fileHandle.getFile();
    return file.arrayBuffer();
  }

    async _writeToOpfs(chunkId, partition, data) {
    const dir = this._opfsDirs.get(partition);
    if (!dir) {
      // Fallback to RAM if OPFS not available
      this._ramStore.set(chunkId, data);
      this._ramUsed += data.byteLength;
      return;
    }

    const fileHandle = await dir.getFileHandle(`${chunkId}.bin`, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(data);
    await writable.close();
  }

    async _deleteFromOpfs(chunkId, partition) {
    const dir = this._opfsDirs.get(partition);
    if (!dir) return;

    try {
      await dir.removeEntry(`${chunkId}.bin`);
    } catch (err) {
      // Ignore
    }
  }

    async destroy() {
    const pool = getBufferPool();

    // Release all VRAM buffers
    for (const gpuBuffer of this._vramStore.values()) {
      pool.release(gpuBuffer);
    }

    this._vramStore.clear();
    this._ramStore.clear();
    this._chunks.clear();
    this._partitions.clear();
    this._opfsDirs.clear();

    this._vramUsed = 0;
    this._ramUsed = 0;
    this._opfsUsed = 0;
    this._initialized = false;

    log.verbose(MODULE, 'EmulatedVramStore destroyed');
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

export function createEmulatedVramStore(rootPath) {
  return new EmulatedVramStore(rootPath, DEFAULT_VRAM_BUDGET, DEFAULT_RAM_BUDGET);
}

export async function detectLocalResources() {
  let vramBytes = 2 * 1024 * 1024 * 1024; // Default 2GB
  let ramBytes = 8 * 1024 * 1024 * 1024;  // Default 8GB
  let storageBytes = 100 * 1024 * 1024 * 1024; // Default 100GB

  if (typeof navigator === 'undefined') {
    return { vramBytes, ramBytes, storageBytes };
  }

  // Try to detect actual VRAM from WebGPU adapter
  try {
    if (navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        // WebGPU doesn't expose VRAM directly, but we can use maxBufferSize as a hint
        const device = await adapter.requestDevice();
        vramBytes = Math.min(device.limits.maxBufferSize, 8 * 1024 * 1024 * 1024);
        device.destroy();
      }
    }
  } catch (err) {
    log.warn(MODULE, `Failed to detect VRAM: ${err.message}`);
  }

  // Try to estimate available RAM
  // @ts-ignore - navigator.deviceMemory is not in TypeScript types
  if (navigator.deviceMemory) {
    // @ts-ignore
    ramBytes = navigator.deviceMemory * 1024 * 1024 * 1024;
  }

  // Try to detect storage quota
  try {
    if (navigator.storage && navigator.storage.estimate) {
      const estimate = await navigator.storage.estimate();
      storageBytes = estimate.quota || storageBytes;
    }
  } catch (err) {
    log.warn(MODULE, `Failed to detect storage: ${err.message}`);
  }

  return { vramBytes, ramBytes, storageBytes };
}
