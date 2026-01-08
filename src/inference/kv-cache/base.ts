/**
 * KV Cache Base - Core KV cache implementation
 *
 * Implements efficient key-value cache for transformer inference.
 * Supports both contiguous and paged memory layouts.
 * GPU-native storage to avoid CPU readbacks during inference.
 *
 * @module inference/kv-cache/base
 */

import { getDevice } from '../../gpu/device.js';
import { allowReadback } from '../../gpu/perf-guards.js';
import { log } from '../../debug/index.js';
import { getRuntimeConfig } from '../../config/runtime.js';
import {
  type KVCacheConfig,
  type ContiguousLayerCache,
  type PagedLayerCache,
  type LayerCache,
  type PageLocation,
  type KVGetResult,
  type GPUBuffersResult,
  type MemoryStats,
  type GPUContext,
  isContiguousLayer,
  isPagedLayer,
  f32ToF16Array,
  f16ToF32Array,
} from './types.js';

// ============================================================================
// KVCache Class
// ============================================================================

export class KVCache {
  readonly numLayers: number;
  readonly numHeads: number;
  readonly headDim: number;
  readonly maxSeqLen: number;
  readonly layout: 'contiguous' | 'paged';
  readonly pageSize: number;
  readonly kvDtype: 'f16' | 'f32';
  readonly bytesPerElem: number;
  readonly kvSize: number;
  readonly windowSize?: number;  // For subclass compatibility

  useGPU: boolean;
  layers: LayerCache[];
  currentSeqLen: number;
  memoryUsage: number;
  gpuContext: GPUContext | null;

  /**
   * @param config - KV cache configuration
   */
  constructor(config: KVCacheConfig) {
    const runtimeKV = getRuntimeConfig().kvcache;
    this.numLayers = config.numLayers;
    this.numHeads = config.numHeads;
    this.headDim = config.headDim;
    // Use config defaults from schema
    this.maxSeqLen = config.maxSeqLen || runtimeKV.maxSeqLen;
    this.useGPU = config.useGPU || false;
    this.layout = config.layout || runtimeKV.layout;
    this.pageSize = config.pageSize || runtimeKV.pageSize;
    this.kvDtype = config.kvDtype || runtimeKV.kvDtype;
    this.bytesPerElem = this.kvDtype === 'f16' ? 2 : 4;

    // Size of one KV pair per position
    this.kvSize = this.numHeads * this.headDim;

    // Initialize layer caches
    this.layers = new Array(this.numLayers);
    this.currentSeqLen = 0;

    // Memory usage tracking
    this.memoryUsage = 0;

    // GPU context (set externally)
    this.gpuContext = null;

    // Initialize storage
    this._initializeStorage();
  }

  // ==========================================================================
  // Storage Initialization
  // ==========================================================================

  /**
   * Initialize storage for all layers
   */
  private _initializeStorage(): void {
    if (this.layout === 'paged') {
      this._initializePagedStorage();
    } else {
      this._initializeContiguousStorage();
    }
  }

  /**
   * Initialize contiguous storage (pre-allocated)
   */
  private _initializeContiguousStorage(): void {
    const sizePerLayer = this.maxSeqLen * this.kvSize;
    const bytesPerLayer = sizePerLayer * this.bytesPerElem * 2; // K + V

    const device = this.useGPU ? getDevice() : null;

    for (let l = 0; l < this.numLayers; l++) {
      if (device && this.useGPU) {
        // GPU-native storage
        const keysGPU = device.createBuffer({
          label: `kv_cache_keys_layer_${l}`,
          size: sizePerLayer * this.bytesPerElem,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        const valuesGPU = device.createBuffer({
          label: `kv_cache_values_layer_${l}`,
          size: sizePerLayer * this.bytesPerElem,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        this.layers[l] = {
          keysGPU,
          valuesGPU,
          // Use empty CPU arrays in GPU mode - saves ~2.7GB for 9B models
          // CPU shadows are only allocated on-demand if migrateFromGPU() is called
          keys: new Float32Array(0),
          values: new Float32Array(0),
          seqLen: 0
        };
      } else {
        // CPU-only storage
        this.layers[l] = {
          keys: new Float32Array(sizePerLayer),
          values: new Float32Array(sizePerLayer),
          keysGPU: null,
          valuesGPU: null,
          seqLen: 0
        };
      }
      this.memoryUsage += bytesPerLayer;
    }
  }

  /**
   * Initialize paged storage (lazy allocation)
   */
  private _initializePagedStorage(): void {
    const numPages = Math.ceil(this.maxSeqLen / this.pageSize);

    for (let l = 0; l < this.numLayers; l++) {
      this.layers[l] = {
        keyPages: new Array(numPages).fill(null),
        valuePages: new Array(numPages).fill(null),
        allocatedPages: 0,
        seqLen: 0
      };
    }
  }

  // ==========================================================================
  // Paged Storage Helpers
  // ==========================================================================

  /**
   * Allocate a new page for paged storage
   */
  private _allocatePage(): Float32Array {
    const pageElements = this.pageSize * this.kvSize;
    const page = new Float32Array(pageElements);
    this.memoryUsage += pageElements * 4;
    return page;
  }

  /**
   * Get the page index and offset for a sequence position
   */
  private _getPageLocation(pos: number): PageLocation {
    const pageIdx = Math.floor(pos / this.pageSize);
    const offset = (pos % this.pageSize) * this.kvSize;
    return { pageIdx, offset };
  }

  /**
   * Ensure pages are allocated up to the given position
   */
  private _ensurePagesAllocated(layerIdx: number, pos: number): void {
    if (this.layout !== 'paged') return;

    const layer = this.layers[layerIdx] as PagedLayerCache;
    const neededPage = Math.floor(pos / this.pageSize);

    for (let p = layer.allocatedPages; p <= neededPage; p++) {
      if (!layer.keyPages[p]) {
        layer.keyPages[p] = this._allocatePage();
        layer.valuePages[p] = this._allocatePage();
        layer.allocatedPages = p + 1;
      }
    }
  }

  // ==========================================================================
  // Update Methods
  // ==========================================================================

  /**
   * Update cache with new key-value pairs for a layer
   * @param layerIdx - Layer index
   * @param keys - New keys [batchSize, numHeads, headDim]
   * @param values - New values [batchSize, numHeads, headDim]
   * @param startPos - Starting position in sequence
   */
  update(
    layerIdx: number,
    keys: Float32Array | GPUBuffer,
    values: Float32Array | GPUBuffer,
    startPos: number = this.currentSeqLen
  ): void {
    const numNewTokens = keys instanceof GPUBuffer
      ? keys.size / (this.kvSize * this.bytesPerElem)
      : keys.length / this.kvSize;

    if (startPos + numNewTokens > this.maxSeqLen) {
      throw new Error(
        `Cache overflow: ${startPos + numNewTokens} > ${this.maxSeqLen}`
      );
    }

    const layer = this.layers[layerIdx];

    if (this.layout === 'paged') {
      if (keys instanceof GPUBuffer || values instanceof GPUBuffer) {
        throw new Error('Paged layout does not support GPU buffer inputs');
      }
      this._updatePaged(layer as PagedLayerCache, keys, values, startPos, numNewTokens);
    } else {
      this._updateContiguous(layer as ContiguousLayerCache, keys, values, startPos, numNewTokens);
    }

    layer.seqLen = Math.max(layer.seqLen, startPos + numNewTokens);

    // Update global sequence length if this is the last layer
    if (layerIdx === this.numLayers - 1) {
      this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numNewTokens);
    }
  }

  /**
   * Update cache directly from GPU buffers (zero-copy)
   */
  updateFromGPU(
    layerIdx: number,
    keysBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    startPos: number,
    numTokens: number
  ): void {
    const layer = this.layers[layerIdx] as ContiguousLayerCache;
    const device = getDevice();

    if (!device || !layer.keysGPU) {
      throw new Error('GPU cache not initialized');
    }

    if (startPos + numTokens > this.maxSeqLen) {
      throw new Error(
        `Cache overflow: ${startPos + numTokens} > ${this.maxSeqLen}`
      );
    }

    const byteOffset = startPos * this.kvSize * this.bytesPerElem;
    const byteSize = numTokens * this.kvSize * this.bytesPerElem;

    // Copy directly from source buffers to cache buffers
    const encoder = device.createCommandEncoder({ label: 'kv_cache_update' });
    encoder.copyBufferToBuffer(keysBuffer, 0, layer.keysGPU, byteOffset, byteSize);
    encoder.copyBufferToBuffer(valuesBuffer, 0, layer.valuesGPU!, byteOffset, byteSize);
    device.queue.submit([encoder.finish()]);

    layer.seqLen = Math.max(layer.seqLen, startPos + numTokens);

    if (layerIdx === this.numLayers - 1) {
      this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numTokens);
    }
  }

  /**
   * Record KV cache update to an external encoder (for batched GPU operations).
   * Does NOT submit - caller is responsible for submitting the encoder.
   */
  recordUpdateFromGPU(
    encoder: GPUCommandEncoder,
    layerIdx: number,
    keysBuffer: GPUBuffer,
    valuesBuffer: GPUBuffer,
    startPos: number,
    numTokens: number
  ): void {
    const layer = this.layers[layerIdx] as ContiguousLayerCache;

    if (!layer.keysGPU) {
      throw new Error('GPU cache not initialized');
    }

    if (startPos + numTokens > this.maxSeqLen) {
      throw new Error(
        `Cache overflow: ${startPos + numTokens} > ${this.maxSeqLen}`
      );
    }

    const byteOffset = startPos * this.kvSize * this.bytesPerElem;
    const byteSize = numTokens * this.kvSize * this.bytesPerElem;

    // Record copy operations to the provided encoder (no submit)
    encoder.copyBufferToBuffer(keysBuffer, 0, layer.keysGPU, byteOffset, byteSize);
    encoder.copyBufferToBuffer(valuesBuffer, 0, layer.valuesGPU!, byteOffset, byteSize);

    // Update seqLen metadata (this happens immediately, copies happen when encoder is submitted)
    layer.seqLen = Math.max(layer.seqLen, startPos + numTokens);

    if (layerIdx === this.numLayers - 1) {
      this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numTokens);
    }
  }

  /**
   * Update contiguous storage
   */
  private _updateContiguous(
    layer: ContiguousLayerCache,
    keys: Float32Array | GPUBuffer,
    values: Float32Array | GPUBuffer,
    startPos: number,
    numNewTokens: number
  ): void {
    const offset = startPos * this.kvSize;
    const device = getDevice();

    // Handle GPU buffer inputs
    if (keys instanceof GPUBuffer) {
      // For GPU inputs, copy to GPU cache directly
      if (layer.keysGPU && device) {
        const byteOffset = offset * this.bytesPerElem;
        const byteSize = numNewTokens * this.kvSize * this.bytesPerElem;
        const encoder = device.createCommandEncoder({ label: 'kv_update_gpu' });
        encoder.copyBufferToBuffer(keys, 0, layer.keysGPU, byteOffset, byteSize);
        encoder.copyBufferToBuffer(values as GPUBuffer, 0, layer.valuesGPU!, byteOffset, byteSize);
        device.queue.submit([encoder.finish()]);
      }
      return;
    }

    // CPU path
    layer.keys.set(keys, offset);
    layer.values.set(values as Float32Array, offset);

    // Also update GPU if available
    if (layer.keysGPU && device) {
      const byteOffset = offset * this.bytesPerElem;
      if (this.kvDtype === 'f16') {
        const keysF16 = f32ToF16Array(keys as Float32Array);
        const valuesF16 = f32ToF16Array(values as Float32Array);
        device.queue.writeBuffer(layer.keysGPU, byteOffset, keysF16);
        device.queue.writeBuffer(layer.valuesGPU!, byteOffset, valuesF16);
      } else {
        device.queue.writeBuffer(layer.keysGPU, byteOffset, keys as Float32Array);
        device.queue.writeBuffer(layer.valuesGPU!, byteOffset, values as Float32Array);
      }
    }
  }

  /**
   * Update paged storage
   */
  private _updatePaged(
    layer: PagedLayerCache,
    keys: Float32Array,
    values: Float32Array,
    startPos: number,
    numNewTokens: number
  ): void {
    for (let t = 0; t < numNewTokens; t++) {
      const pos = startPos + t;
      this._ensurePagesAllocated(this.layers.indexOf(layer), pos);

      const { pageIdx, offset } = this._getPageLocation(pos);
      const srcOffset = t * this.kvSize;

      layer.keyPages[pageIdx]!.set(
        keys.subarray(srcOffset, srcOffset + this.kvSize),
        offset
      );
      layer.valuePages[pageIdx]!.set(
        values.subarray(srcOffset, srcOffset + this.kvSize),
        offset
      );
    }
  }

  // ==========================================================================
  // Get Methods
  // ==========================================================================

  /**
   * Get cached keys and values for a layer
   */
  get(layerIdx: number, startPos: number = 0, endPos?: number): KVGetResult {
    const layer = this.layers[layerIdx];
    const actualEndPos = endPos ?? layer.seqLen;

    if (this.layout === 'paged') {
      return this._getPaged(layer as PagedLayerCache, startPos, actualEndPos);
    } else {
      return this._getContiguous(layer as ContiguousLayerCache, startPos, actualEndPos);
    }
  }

  /**
   * Get key cache buffer (GPU or CPU)
   */
  getKeyCache(layerIdx: number): GPUBuffer | Float32Array | null {
    const layer = this.layers[layerIdx];
    if (isContiguousLayer(layer)) {
      return layer.keysGPU || layer.keys;
    }
    return null;
  }

  /**
   * Get value cache buffer (GPU or CPU)
   */
  getValueCache(layerIdx: number): GPUBuffer | Float32Array | null {
    const layer = this.layers[layerIdx];
    if (isContiguousLayer(layer)) {
      return layer.valuesGPU || layer.values;
    }
    return null;
  }

  /**
   * Get GPU buffers for a layer (for GPU-native attention)
   */
  getGPUBuffers(layerIdx: number): GPUBuffersResult | null {
    const layer = this.layers[layerIdx];

    if (!isContiguousLayer(layer) || !layer.keysGPU || !layer.valuesGPU) {
      return null;
    }

    return {
      keysGPU: layer.keysGPU,
      valuesGPU: layer.valuesGPU,
      seqLen: layer.seqLen,
    };
  }

  /**
   * Check if GPU cache is available
   */
  hasGPUCache(): boolean {
    const firstLayer = this.layers[0];
    return this.useGPU && isContiguousLayer(firstLayer) && firstLayer.keysGPU != null;
  }

  /**
   * Get from contiguous storage
   */
  private _getContiguous(
    layer: ContiguousLayerCache,
    startPos: number,
    endPos: number
  ): KVGetResult {
    const startOffset = startPos * this.kvSize;
    const endOffset = endPos * this.kvSize;

    return {
      keys: layer.keys.subarray(startOffset, endOffset),
      values: layer.values.subarray(startOffset, endOffset)
    };
  }

  /**
   * Get from paged storage
   */
  private _getPaged(
    layer: PagedLayerCache,
    startPos: number,
    endPos: number
  ): KVGetResult {
    const length = (endPos - startPos) * this.kvSize;
    const keys = new Float32Array(length);
    const values = new Float32Array(length);

    let destOffset = 0;
    for (let pos = startPos; pos < endPos; pos++) {
      const { pageIdx, offset } = this._getPageLocation(pos);

      keys.set(
        layer.keyPages[pageIdx]!.subarray(offset, offset + this.kvSize),
        destOffset
      );
      values.set(
        layer.valuePages[pageIdx]!.subarray(offset, offset + this.kvSize),
        destOffset
      );

      destOffset += this.kvSize;
    }

    return { keys, values };
  }

  // ==========================================================================
  // Cache Management
  // ==========================================================================

  /**
   * Clear cache for all layers
   */
  clear(): void {
    this.currentSeqLen = 0;

    for (let l = 0; l < this.numLayers; l++) {
      const layer = this.layers[l];
      layer.seqLen = 0;

      if (this.layout === 'paged') {
        // Don't deallocate pages, just reset length
        // Pages will be reused
      } else {
        // Zero out contiguous arrays
        const contiguousLayer = layer as ContiguousLayerCache;
        contiguousLayer.keys.fill(0);
        contiguousLayer.values.fill(0);
      }
    }
  }

  /**
   * Clone the cache (for speculative decoding rollback)
   */
  clone(): KVCache {
    const cloned = new KVCache({
      numLayers: this.numLayers,
      numHeads: this.numHeads,
      headDim: this.headDim,
      maxSeqLen: this.maxSeqLen,
      useGPU: false, // Always clone to CPU
      layout: 'contiguous', // Simplify for clone
      pageSize: this.pageSize
    });

    cloned.currentSeqLen = this.currentSeqLen;

    for (let l = 0; l < this.numLayers; l++) {
      const { keys, values } = this.get(l);
      const clonedLayer = cloned.layers[l] as ContiguousLayerCache;
      clonedLayer.keys.set(keys);
      clonedLayer.values.set(values);
      clonedLayer.seqLen = this.layers[l].seqLen;
    }

    return cloned;
  }

  /**
   * Truncate cache to a specific length (for rollback)
   */
  truncate(length: number): void {
    if (length >= this.currentSeqLen) return;

    this.currentSeqLen = length;
    for (let l = 0; l < this.numLayers; l++) {
      this.layers[l].seqLen = Math.min(this.layers[l].seqLen, length);
    }
  }

  /**
   * Get memory usage statistics
   */
  getMemoryStats(): MemoryStats {
    const theoretical = this.numLayers * 2 * this.maxSeqLen * this.kvSize * this.bytesPerElem;
    const actual = this.memoryUsage;
    const used = this.numLayers * 2 * this.currentSeqLen * this.kvSize * this.bytesPerElem;

    return {
      theoretical: theoretical,
      allocated: actual,
      used: used,
      efficiency: used / actual,
      seqLen: this.currentSeqLen,
      maxSeqLen: this.maxSeqLen,
      layout: this.layout
    };
  }

  // ==========================================================================
  // GPU Migration
  // ==========================================================================

  /**
   * Set GPU context for GPU-based caching
   */
  setGPUContext(gpuContext: GPUContext): void {
    this.gpuContext = gpuContext;

    // Migrate existing cache to GPU if we have data
    if (this.currentSeqLen > 0 && gpuContext?.device) {
      this._migrateToGPU(gpuContext.device);
    }
  }

  /**
   * Migrate existing CPU cache data to GPU buffers
   */
  private _migrateToGPU(device: GPUDevice): void {
    if (this.layout === 'paged') {
      log.warn('KVCache', 'GPU migration not supported for paged layout');
      return;
    }

    log.info('KVCache', `Migrating ${this.currentSeqLen} positions to GPU...`);
    const sizePerLayer = this.maxSeqLen * this.kvSize;
    const bytesPerLayer = sizePerLayer * this.bytesPerElem;

    for (let l = 0; l < this.numLayers; l++) {
      const layer = this.layers[l] as ContiguousLayerCache;

      // Create GPU buffers if they don't exist
      if (!layer.keysGPU) {
        layer.keysGPU = device.createBuffer({
          label: `kv_cache_keys_layer_${l}`,
          size: bytesPerLayer,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
      }
      if (!layer.valuesGPU) {
        layer.valuesGPU = device.createBuffer({
          label: `kv_cache_values_layer_${l}`,
          size: bytesPerLayer,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
      }

      // Upload existing CPU data to GPU
      const usedElems = layer.seqLen * this.kvSize;
      const usedSize = usedElems * this.bytesPerElem;
      if (usedSize > 0) {
        if (this.kvDtype === 'f16') {
          const keysF16 = f32ToF16Array(layer.keys.subarray(0, usedElems));
          const valuesF16 = f32ToF16Array(layer.values.subarray(0, usedElems));
          device.queue.writeBuffer(layer.keysGPU, 0, keysF16);
          device.queue.writeBuffer(layer.valuesGPU, 0, valuesF16);
        } else {
          device.queue.writeBuffer(
            layer.keysGPU,
            0,
            layer.keys.buffer,
            layer.keys.byteOffset,
            usedSize
          );
          device.queue.writeBuffer(
            layer.valuesGPU,
            0,
            layer.values.buffer,
            layer.values.byteOffset,
            usedSize
          );
        }
      }
    }

    this.useGPU = true;
    log.info('KVCache', 'Migration complete');
  }

  /**
   * Sync GPU cache back to CPU (for debugging or fallback)
   */
  async syncToCPU(): Promise<void> {
    if (!this.useGPU || this.layout === 'paged') return;
    if (!allowReadback('kv-cache.syncToCPU')) return;

    const device = getDevice();
    if (!device) return;

    const sizePerLayer = this.maxSeqLen * this.kvSize;

    for (let l = 0; l < this.numLayers; l++) {
      const layer = this.layers[l] as ContiguousLayerCache;
      if (!layer.keysGPU || !layer.valuesGPU) continue;

      const usedSize = layer.seqLen * this.kvSize * this.bytesPerElem;
      if (usedSize === 0) continue;

      // Allocate CPU arrays on-demand if empty (lazy allocation from GPU mode)
      if (layer.keys.length === 0) {
        layer.keys = new Float32Array(sizePerLayer);
        layer.values = new Float32Array(sizePerLayer);
      }

      // Create staging buffers for readback
      const keysStaging = device.createBuffer({
        size: usedSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const valuesStaging = device.createBuffer({
        size: usedSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      // Copy from GPU cache to staging
      const encoder = device.createCommandEncoder({ label: 'kv_cache_sync' });
      encoder.copyBufferToBuffer(layer.keysGPU, 0, keysStaging, 0, usedSize);
      encoder.copyBufferToBuffer(layer.valuesGPU, 0, valuesStaging, 0, usedSize);
      device.queue.submit([encoder.finish()]);

      // Map and copy to CPU arrays
      await keysStaging.mapAsync(GPUMapMode.READ);
      await valuesStaging.mapAsync(GPUMapMode.READ);

      if (this.kvDtype === 'f16') {
        const keysRaw = new Uint16Array(keysStaging.getMappedRange().slice(0));
        const valuesRaw = new Uint16Array(valuesStaging.getMappedRange().slice(0));
        const keysData = f16ToF32Array(keysRaw);
        const valuesData = f16ToF32Array(valuesRaw);
        layer.keys.set(keysData);
        layer.values.set(valuesData);
      } else {
        const keysData = new Float32Array(keysStaging.getMappedRange().slice(0));
        const valuesData = new Float32Array(valuesStaging.getMappedRange().slice(0));
        layer.keys.set(keysData);
        layer.values.set(valuesData);
      }

      keysStaging.unmap();
      valuesStaging.unmap();
      keysStaging.destroy();
      valuesStaging.destroy();
    }
  }

  /**
   * Destroy GPU resources
   */
  destroy(): void {
    for (let l = 0; l < this.numLayers; l++) {
      const layer = this.layers[l];
      if (isContiguousLayer(layer)) {
        if (layer.keysGPU) {
          layer.keysGPU.destroy();
          layer.keysGPU = null;
        }
        if (layer.valuesGPU) {
          layer.valuesGPU.destroy();
          layer.valuesGPU = null;
        }
      }
    }
  }
}
