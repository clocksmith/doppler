

import { getDevice } from '../../gpu/device.js';
import { allowReadback } from '../../gpu/perf-guards.js';
import { log } from '../../debug/index.js';
import {
  isContiguousLayer,
  isPagedLayer,
  f32ToF16Array,
  f16ToF32Array,
} from './types.js';

// ============================================================================
// KVCache Class
// ============================================================================

export class KVCache {
  
  constructor(config) {
    this.numLayers = config.numLayers;
    
    this.numHeads = config.numHeads;
    
    this.headDim = config.headDim;
    if (!Number.isFinite(config.maxSeqLen) || config.maxSeqLen <= 0) {
      throw new Error('KVCache requires a valid maxSeqLen.');
    }
    if (config.useGPU == null) {
      throw new Error('KVCache requires useGPU to be set.');
    }
    if (!config.layout) {
      throw new Error('KVCache requires a layout.');
    }
    if (!Number.isFinite(config.pageSize) || config.pageSize <= 0) {
      throw new Error('KVCache requires a valid pageSize.');
    }
    if (!config.kvDtype) {
      throw new Error('KVCache requires kvDtype.');
    }

    this.maxSeqLen = config.maxSeqLen;
    
    this.useGPU = config.useGPU;
    
    this.layout = config.layout;
    
    this.pageSize = config.pageSize;
    
    this.kvDtype = config.kvDtype;
    
    this.bytesPerElem = this.kvDtype === 'f16' ? 2 : 4;

    // Size of one KV pair per position
    
    this.kvSize = this.numHeads * this.headDim;

    // Initialize layer caches
    
    this.layers = new Array(this.numLayers);
    
    this.currentSeqLen = 0;
    
    this.totalTokensSeen = 0;

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

  
  _initializeStorage() {
    if (this.layout === 'paged') {
      this._initializePagedStorage();
    } else {
      this._initializeContiguousStorage();
    }
  }

  
  _initializeContiguousStorage() {
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

  
  _initializePagedStorage() {
    const numPages = Math.ceil(this.maxSeqLen / this.pageSize);
    const device = this.useGPU ? getDevice() : null;
    const sizePerLayer = this.maxSeqLen * this.kvSize;
    const bytesPerLayer = sizePerLayer * this.bytesPerElem;
    const pageTableBytes = numPages * 4;

    for (let l = 0; l < this.numLayers; l++) {
      if (device && this.useGPU) {
        const keysGPU = device.createBuffer({
          label: `kv_cache_keys_paged_layer_${l}`,
          size: bytesPerLayer,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        const valuesGPU = device.createBuffer({
          label: `kv_cache_values_paged_layer_${l}`,
          size: bytesPerLayer,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        const pageTable = new Uint32Array(numPages);
        for (let i = 0; i < numPages; i++) {
          pageTable[i] = i;
        }
        const pageTableGPU = device.createBuffer({
          label: `kv_cache_page_table_layer_${l}`,
          size: pageTableBytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(pageTableGPU, 0, pageTable);

        this.layers[l] = {
          keyPages: new Array(numPages).fill(null),
          valuePages: new Array(numPages).fill(null),
          allocatedPages: 0,
          seqLen: 0,
          keysGPU,
          valuesGPU,
          pageTable,
          pageTableGPU,
        };
        this.memoryUsage += (bytesPerLayer * 2) + pageTableBytes;
      } else {
        this.layers[l] = {
          keyPages: new Array(numPages).fill(null),
          valuePages: new Array(numPages).fill(null),
          allocatedPages: 0,
          seqLen: 0,
          keysGPU: null,
          valuesGPU: null,
          pageTable: null,
          pageTableGPU: null,
        };
      }
    }
  }

  // ==========================================================================
  // Paged Storage Helpers
  // ==========================================================================

  
  _allocatePage() {
    const pageElements = this.pageSize * this.kvSize;
    const page = new Float32Array(pageElements);
    this.memoryUsage += pageElements * 4;
    return page;
  }

  
  _getPageLocation(pos) {
    const pageIdx = Math.floor(pos / this.pageSize);
    const offset = (pos % this.pageSize) * this.kvSize;
    return { pageIdx, offset };
  }

  
  _ensurePagesAllocated(layerIdx, pos) {
    if (this.layout !== 'paged') return;

    const layer =  (this.layers[layerIdx]);
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

  
  _assertLayerIndex(layerIdx) {
    if (!Number.isInteger(layerIdx) || layerIdx < 0 || layerIdx >= this.numLayers) {
      throw new Error(`KVCache layer index out of range: ${layerIdx}`);
    }
  }

  
  _assertStartPos(startPos) {
    if (!Number.isInteger(startPos) || startPos < 0) {
      throw new Error('KVCache startPos must be a non-negative integer.');
    }
  }

  
  _resolveTokenCount(keys, values) {
    const bytesPerToken = this.kvSize * this.bytesPerElem;
    const keysIsGPU = keys instanceof GPUBuffer;
    const valuesIsGPU = values instanceof GPUBuffer;

    if (keysIsGPU || valuesIsGPU) {
      if (!keysIsGPU || !valuesIsGPU) {
        throw new Error('KVCache update requires both keys and values as GPU buffers.');
      }
      if (!Number.isFinite(keys.size) || !Number.isFinite(values.size)) {
        throw new Error('KVCache update requires GPU buffer sizes.');
      }
      if (keys.size !== values.size) {
        throw new Error('KVCache update requires matching GPU buffer sizes.');
      }
      if (keys.size % bytesPerToken !== 0) {
        throw new Error('KVCache update GPU buffer size must align to kvSize.');
      }
      return { numNewTokens: keys.size / bytesPerToken, isGPU: true };
    }

    if (!Number.isFinite(keys?.length) || !Number.isFinite(values?.length)) {
      throw new Error('KVCache update requires typed array inputs.');
    }
    if (keys.length !== values.length) {
      throw new Error('KVCache update requires matching keys and values lengths.');
    }
    if (keys.length % this.kvSize !== 0) {
      throw new Error('KVCache update array length must align to kvSize.');
    }
    return { numNewTokens: keys.length / this.kvSize, isGPU: false };
  }

  
  update(
    layerIdx,
    keys,
    values,
    startPos = this.currentSeqLen
  ) {
    this._assertLayerIndex(layerIdx);
    this._assertStartPos(startPos);

    const { numNewTokens } = this._resolveTokenCount(keys, values);
    if (!Number.isInteger(numNewTokens) || numNewTokens < 0) {
      throw new Error('KVCache update requires a non-negative integer token count.');
    }
    if (numNewTokens === 0) {
      return;
    }

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
      this._updatePaged(layerIdx,  (layer), keys, values, startPos, numNewTokens);
    } else {
      this._updateContiguous( (layer), keys, values, startPos, numNewTokens);
    }

    layer.seqLen = Math.max(layer.seqLen, startPos + numNewTokens);
    this.totalTokensSeen = Math.max(this.totalTokensSeen, startPos + numNewTokens);

    // Update global sequence length if this is the last layer
    if (layerIdx === this.numLayers - 1) {
      this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numNewTokens);
    }
  }

  
  updateFromGPU(
    layerIdx,
    keysBuffer,
    valuesBuffer,
    startPos,
    numTokens
  ) {
    this._assertLayerIndex(layerIdx);
    this._assertStartPos(startPos);
    if (!Number.isInteger(numTokens) || numTokens < 0) {
      throw new Error('KVCache updateFromGPU requires a non-negative integer token count.');
    }
    if (numTokens === 0) {
      return;
    }

    const layer =  (this.layers[layerIdx]);
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
    if (byteSize > keysBuffer.size || byteSize > valuesBuffer.size) {
      throw new Error('KVCache updateFromGPU buffer size is smaller than requested write.');
    }

    // Copy directly from source buffers to cache buffers
    const encoder = device.createCommandEncoder({ label: 'kv_cache_update' });
    encoder.copyBufferToBuffer(keysBuffer, 0, layer.keysGPU, byteOffset, byteSize);
    encoder.copyBufferToBuffer(valuesBuffer, 0, layer.valuesGPU, byteOffset, byteSize);
    device.queue.submit([encoder.finish()]);

    if (this.layout === 'paged') {
      const neededPages = Math.ceil((startPos + numTokens) / this.pageSize);
      if (Number.isFinite(layer.allocatedPages)) {
        layer.allocatedPages = Math.max(layer.allocatedPages, neededPages);
      } else {
        layer.allocatedPages = neededPages;
      }
    }

    layer.seqLen = Math.max(layer.seqLen, startPos + numTokens);
    this.totalTokensSeen = Math.max(this.totalTokensSeen, startPos + numTokens);

    if (layerIdx === this.numLayers - 1) {
      this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numTokens);
    }
  }

  
  recordUpdateFromGPU(
    recorder,
    layerIdx,
    keysBuffer,
    valuesBuffer,
    startPos,
    numTokens
  ) {
    this._assertLayerIndex(layerIdx);
    this._assertStartPos(startPos);
    if (!Number.isInteger(numTokens) || numTokens < 0) {
      throw new Error('KVCache recordUpdateFromGPU requires a non-negative integer token count.');
    }
    if (numTokens === 0) {
      return;
    }

    const encoder = recorder.getEncoder();
    const layer =  (this.layers[layerIdx]);

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
    if (byteSize > keysBuffer.size || byteSize > valuesBuffer.size) {
      throw new Error('KVCache recordUpdateFromGPU buffer size is smaller than requested write.');
    }

    // Record copy operations to the provided encoder (no submit)
    encoder.copyBufferToBuffer(keysBuffer, 0, layer.keysGPU, byteOffset, byteSize);
    encoder.copyBufferToBuffer(valuesBuffer, 0, layer.valuesGPU, byteOffset, byteSize);

    if (this.layout === 'paged') {
      const neededPages = Math.ceil((startPos + numTokens) / this.pageSize);
      if (Number.isFinite(layer.allocatedPages)) {
        layer.allocatedPages = Math.max(layer.allocatedPages, neededPages);
      } else {
        layer.allocatedPages = neededPages;
      }
    }

    // Update seqLen metadata (this happens immediately, copies happen when encoder is submitted)
    layer.seqLen = Math.max(layer.seqLen, startPos + numTokens);
    this.totalTokensSeen = Math.max(this.totalTokensSeen, startPos + numTokens);

    if (layerIdx === this.numLayers - 1) {
      this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numTokens);
    }
  }

  
  _updateContiguous(
    layer,
    keys,
    values,
    startPos,
    numNewTokens
  ) {
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
        encoder.copyBufferToBuffer( (values), 0, layer.valuesGPU, byteOffset, byteSize);
        device.queue.submit([encoder.finish()]);
      }
      return;
    }

    // CPU path
    layer.keys.set(keys, offset);
    layer.values.set( (values), offset);

    // Also update GPU if available
    if (layer.keysGPU && device) {
      const byteOffset = offset * this.bytesPerElem;
      if (this.kvDtype === 'f16') {
        const keysF16 = f32ToF16Array( (keys));
        const valuesF16 = f32ToF16Array( (values));
        device.queue.writeBuffer(layer.keysGPU, byteOffset, keysF16);
        device.queue.writeBuffer(layer.valuesGPU, byteOffset, valuesF16);
      } else {
        device.queue.writeBuffer(layer.keysGPU, byteOffset,  (keys));
        device.queue.writeBuffer(layer.valuesGPU, byteOffset,  (values));
      }
    }
  }

  
  _updatePaged(
    layerIdx,
    layer,
    keys,
    values,
    startPos,
    numNewTokens
  ) {
    const device = getDevice();
    if (layer.keysGPU && layer.valuesGPU && device) {
      const byteOffset = startPos * this.kvSize * this.bytesPerElem;
      if (this.kvDtype === 'f16') {
        const keysF16 = f32ToF16Array( (keys));
        const valuesF16 = f32ToF16Array( (values));
        device.queue.writeBuffer(layer.keysGPU, byteOffset, keysF16);
        device.queue.writeBuffer(layer.valuesGPU, byteOffset, valuesF16);
      } else {
        device.queue.writeBuffer(layer.keysGPU, byteOffset,  (keys));
        device.queue.writeBuffer(layer.valuesGPU, byteOffset,  (values));
      }
    }

    for (let t = 0; t < numNewTokens; t++) {
      const pos = startPos + t;
      this._ensurePagesAllocated(layerIdx, pos);

      const { pageIdx, offset } = this._getPageLocation(pos);
      const srcOffset = t * this.kvSize;

      layer.keyPages[pageIdx].set(
        keys.subarray(srcOffset, srcOffset + this.kvSize),
        offset
      );
      layer.valuePages[pageIdx].set(
        values.subarray(srcOffset, srcOffset + this.kvSize),
        offset
      );
    }
  }

  // ==========================================================================
  // Get Methods
  // ==========================================================================

  
  get(layerIdx, startPos = 0, endPos) {
    this._assertLayerIndex(layerIdx);
    this._assertStartPos(startPos);
    const layer = this.layers[layerIdx];
    const actualEndPos = endPos ?? layer.seqLen;
    if (!Number.isInteger(actualEndPos) || actualEndPos < startPos) {
      throw new Error('KVCache get requires endPos >= startPos.');
    }
    if (actualEndPos > layer.seqLen) {
      throw new Error('KVCache get range exceeds cached sequence length.');
    }

    if (this.layout === 'paged' && this.useGPU) {
      throw new Error('Paged GPU cache does not support CPU readback via get().');
    }

    if (this.layout === 'paged') {
      return this._getPaged( (layer), startPos, actualEndPos);
    } else {
      return this._getContiguous( (layer), startPos, actualEndPos);
    }
  }

  
  getKeyCache(layerIdx) {
    const layer = this.layers[layerIdx];
    if (isContiguousLayer(layer)) {
      return layer.keysGPU || layer.keys;
    }
    if (this.layout === 'paged') {
      return layer.keysGPU || null;
    }
    return null;
  }

  
  getValueCache(layerIdx) {
    const layer = this.layers[layerIdx];
    if (isContiguousLayer(layer)) {
      return layer.valuesGPU || layer.values;
    }
    if (this.layout === 'paged') {
      return layer.valuesGPU || null;
    }
    return null;
  }

  
  getGPUBuffers(layerIdx) {
    const layer = this.layers[layerIdx];

    if (this.layout === 'paged') {
      if (!layer.keysGPU || !layer.valuesGPU || !layer.pageTableGPU) {
        return null;
      }
      return {
        keysGPU: layer.keysGPU,
        valuesGPU: layer.valuesGPU,
        seqLen: layer.seqLen,
        pageTableGPU: layer.pageTableGPU,
        pageSize: this.pageSize,
      };
    }

    if (!isContiguousLayer(layer) || !layer.keysGPU || !layer.valuesGPU) {
      return null;
    }

    return {
      keysGPU: layer.keysGPU,
      valuesGPU: layer.valuesGPU,
      seqLen: layer.seqLen,
    };
  }

  
  hasGPUCache() {
    const firstLayer = this.layers[0];
    if (!this.useGPU) return false;
    if (this.layout === 'paged') {
      return firstLayer.keysGPU != null && firstLayer.valuesGPU != null;
    }
    return isContiguousLayer(firstLayer) && firstLayer.keysGPU != null;
  }

  
  _getContiguous(
    layer,
    startPos,
    endPos
  ) {
    const startOffset = startPos * this.kvSize;
    const endOffset = endPos * this.kvSize;

    return {
      keys: layer.keys.subarray(startOffset, endOffset),
      values: layer.values.subarray(startOffset, endOffset)
    };
  }

  
  _getPaged(
    layer,
    startPos,
    endPos
  ) {
    const length = (endPos - startPos) * this.kvSize;
    const keys = new Float32Array(length);
    const values = new Float32Array(length);

    let destOffset = 0;
    for (let pos = startPos; pos < endPos; pos++) {
      const { pageIdx, offset } = this._getPageLocation(pos);
      const keyPage = layer.keyPages[pageIdx];
      const valuePage = layer.valuePages[pageIdx];
      if (!keyPage || !valuePage) {
        destOffset += this.kvSize;
        continue;
      }

      keys.set(keyPage.subarray(offset, offset + this.kvSize), destOffset);
      values.set(valuePage.subarray(offset, offset + this.kvSize), destOffset);

      destOffset += this.kvSize;
    }

    return { keys, values };
  }

  // ==========================================================================
  // Cache Management
  // ==========================================================================

  
  clear() {
    this.currentSeqLen = 0;
    this.totalTokensSeen = 0;

    for (let l = 0; l < this.numLayers; l++) {
      const layer = this.layers[l];
      layer.seqLen = 0;

      if (this.layout === 'paged') {
        // Don't deallocate pages, just reset length
        // Pages will be reused
      } else {
        // Zero out contiguous arrays
        const contiguousLayer =  (layer);
        contiguousLayer.keys.fill(0);
        contiguousLayer.values.fill(0);
      }
    }
  }

  
  clone() {
    if (this.layout === 'paged') {
      const cloned = new KVCache({
        numLayers: this.numLayers,
        numHeads: this.numHeads,
        headDim: this.headDim,
        maxSeqLen: this.maxSeqLen,
        useGPU: this.useGPU,
        layout: 'paged',
        pageSize: this.pageSize,
        kvDtype: this.kvDtype
      });

      cloned.currentSeqLen = this.currentSeqLen;
      cloned.totalTokensSeen = this.totalTokensSeen;

      const device = this.useGPU ? getDevice() : null;
      if (this.useGPU && !device) {
        throw new Error('GPU device not initialized');
      }

      for (let l = 0; l < this.numLayers; l++) {
        const src = this.layers[l];
        const dst = cloned.layers[l];

        const inferredPages = Math.ceil(src.seqLen / this.pageSize);
        const allocatedPages = Number.isFinite(src.allocatedPages)
          ? Math.max(src.allocatedPages, inferredPages)
          : inferredPages;
        dst.allocatedPages = allocatedPages;
        dst.seqLen = src.seqLen;

        if (this.useGPU && src.keysGPU && src.valuesGPU && dst.keysGPU && dst.valuesGPU) {
          const usedBytes = src.seqLen * this.kvSize * this.bytesPerElem;
          if (usedBytes > 0) {
            const encoder = device.createCommandEncoder({ label: `kv_cache_clone_paged_${l}` });
            encoder.copyBufferToBuffer(src.keysGPU, 0, dst.keysGPU, 0, usedBytes);
            encoder.copyBufferToBuffer(src.valuesGPU, 0, dst.valuesGPU, 0, usedBytes);
            device.queue.submit([encoder.finish()]);
          }
        }

        if (src.pageTable && dst.pageTable && dst.pageTableGPU && device) {
          dst.pageTable.set(src.pageTable);
          device.queue.writeBuffer(dst.pageTableGPU, 0, src.pageTable);
        }

        if (!this.useGPU) {
          for (let p = 0; p < allocatedPages; p++) {
            const keyPage = src.keyPages?.[p];
            const valuePage = src.valuePages?.[p];
            if (!keyPage || !valuePage) {
              continue;
            }
            const nextKeyPage = cloned._allocatePage();
            nextKeyPage.set(keyPage);
            const nextValuePage = cloned._allocatePage();
            nextValuePage.set(valuePage);
            dst.keyPages[p] = nextKeyPage;
            dst.valuePages[p] = nextValuePage;
          }
        }
      }

      return cloned;
    }

    const cloned = new KVCache({
      numLayers: this.numLayers,
      numHeads: this.numHeads,
      headDim: this.headDim,
      maxSeqLen: this.maxSeqLen,
      useGPU: false, // Always clone to CPU
      layout: 'contiguous', // Simplify for clone
      pageSize: this.pageSize,
      kvDtype: this.kvDtype
    });

    cloned.currentSeqLen = this.currentSeqLen;

    for (let l = 0; l < this.numLayers; l++) {
      const { keys, values } = this.get(l);
      const clonedLayer =  (cloned.layers[l]);
      clonedLayer.keys.set(keys);
      clonedLayer.values.set(values);
      clonedLayer.seqLen = this.layers[l].seqLen;
    }

    return cloned;
  }

  
  truncate(length) {
    if (length >= this.currentSeqLen) return;

    this.currentSeqLen = length;
    this.totalTokensSeen = Math.min(this.totalTokensSeen, length);
    for (let l = 0; l < this.numLayers; l++) {
      this.layers[l].seqLen = Math.min(this.layers[l].seqLen, length);
    }
  }

  
  getMemoryStats() {
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

  
  setGPUContext(gpuContext) {
    this.gpuContext = gpuContext;

    // Migrate existing cache to GPU if we have data
    if (this.currentSeqLen > 0 && gpuContext?.device) {
      this._migrateToGPU(gpuContext.device);
    }
  }

  
  _migrateToGPU(device) {
    if (this.layout === 'paged') {
      log.info('KVCache', `Migrating ${this.currentSeqLen} positions to GPU (paged)...`);
      const numPages = Math.ceil(this.maxSeqLen / this.pageSize);
      const sizePerLayer = this.maxSeqLen * this.kvSize;
      const bytesPerLayer = sizePerLayer * this.bytesPerElem;
      const pageTableBytes = numPages * 4;

      for (let l = 0; l < this.numLayers; l++) {
        const layer =  (this.layers[l]);

        if (!layer.keysGPU) {
          layer.keysGPU = device.createBuffer({
            label: `kv_cache_keys_paged_layer_${l}`,
            size: bytesPerLayer,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
          });
        }
        if (!layer.valuesGPU) {
          layer.valuesGPU = device.createBuffer({
            label: `kv_cache_values_paged_layer_${l}`,
            size: bytesPerLayer,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
          });
        }

        if (!layer.pageTable) {
          layer.pageTable = new Uint32Array(numPages);
          for (let i = 0; i < numPages; i++) {
            layer.pageTable[i] = i;
          }
        }
        if (!layer.pageTableGPU) {
          layer.pageTableGPU = device.createBuffer({
            label: `kv_cache_page_table_layer_${l}`,
            size: pageTableBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
        }
        device.queue.writeBuffer(layer.pageTableGPU, 0, layer.pageTable);

        const allocatedPages = layer.allocatedPages ?? 0;
        if (allocatedPages > 0) {
          const pageElems = this.pageSize * this.kvSize;
          const pageBytes = pageElems * this.bytesPerElem;
          for (let p = 0; p < allocatedPages; p++) {
            const keyPage = layer.keyPages?.[p];
            const valuePage = layer.valuePages?.[p];
            if (!keyPage || !valuePage) continue;
            const byteOffset = p * pageBytes;
            if (this.kvDtype === 'f16') {
              const keysF16 = f32ToF16Array(keyPage);
              const valuesF16 = f32ToF16Array(valuePage);
              device.queue.writeBuffer(layer.keysGPU, byteOffset, keysF16);
              device.queue.writeBuffer(layer.valuesGPU, byteOffset, valuesF16);
            } else {
              device.queue.writeBuffer(layer.keysGPU, byteOffset, keyPage);
              device.queue.writeBuffer(layer.valuesGPU, byteOffset, valuePage);
            }
          }
        }
      }

      this.useGPU = true;
      log.info('KVCache', 'Paged migration complete');
      return;
    }

    log.info('KVCache', `Migrating ${this.currentSeqLen} positions to GPU...`);
    const sizePerLayer = this.maxSeqLen * this.kvSize;
    const bytesPerLayer = sizePerLayer * this.bytesPerElem;

    for (let l = 0; l < this.numLayers; l++) {
      const layer =  (this.layers[l]);

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

  
  async syncToCPU() {
    if (!this.useGPU || this.layout === 'paged') return;
    if (!allowReadback('kv-cache.syncToCPU')) return;

    const device = getDevice();
    if (!device) return;

    const sizePerLayer = this.maxSeqLen * this.kvSize;

    for (let l = 0; l < this.numLayers; l++) {
      const layer =  (this.layers[l]);
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

  
  destroy() {
    for (let l = 0; l < this.numLayers; l++) {
      const layer = this.layers[l];
      if (this.layout === 'paged') {
        if (layer.keysGPU) {
          layer.keysGPU.destroy();
          layer.keysGPU = null;
        }
        if (layer.valuesGPU) {
          layer.valuesGPU.destroy();
          layer.valuesGPU = null;
        }
        if (layer.pageTableGPU) {
          layer.pageTableGPU.destroy();
          layer.pageTableGPU = null;
        }
        continue;
      }
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
