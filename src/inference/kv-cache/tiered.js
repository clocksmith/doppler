
import { getDevice } from '../../gpu/device.js';
import { runKVQuantize, recordKVQuantize } from '../../gpu/kernel-selector.js';
import { KVCache } from './base.js';
import { SlidingWindowKVCache } from './sliding-window.js';

// ============================================================================
// TieredKVCache (hot ring + cold paged)
// ============================================================================


export class TieredKVCache {
  
  constructor(config, caches = null) {
    if (!config) {
      throw new Error('TieredKVCache requires a config.');
    }
    if (config.layout !== 'tiered') {
      throw new Error('TieredKVCache requires layout="tiered".');
    }
    if (!config.tiering) {
      throw new Error('TieredKVCache requires tiering config.');
    }
    const tiering = config.tiering;
    if (!Number.isFinite(tiering.hotWindow) || tiering.hotWindow <= 0) {
      throw new Error('TieredKVCache requires a positive tiering.hotWindow.');
    }
    if (!Number.isFinite(tiering.coldPageSize) || tiering.coldPageSize <= 0) {
      throw new Error('TieredKVCache requires a positive tiering.coldPageSize.');
    }

    this.numLayers = config.numLayers;
    
    this.numHeads = config.numHeads;
    
    this.headDim = config.headDim;
    
    this.maxSeqLen = config.maxSeqLen;
    
    this.useGPU = config.useGPU;
    
    this.layout = 'tiered';
    
    this.kvDtype = config.kvDtype;
    
    this.bytesPerElem = this.kvDtype === 'f16' ? 2 : 4;
    
    this.kvSize = this.numHeads * this.headDim;
    
    this.hotWindow = tiering.hotWindow;
    
    this.coldPageSize = tiering.coldPageSize;
    
    this.coldDtype = tiering.coldDtype ?? this.kvDtype;
    
    this.tieringMode = tiering.mode;
    
    const defaultCompressionMode = tiering.mode === 'int8'
      ? 'int8'
      : (tiering.mode === 'int4' ? 'int4' : 'none');
    this.compression = tiering.compression ?? { mode: defaultCompressionMode, blockSize: 1 };
    
    this.gating = tiering.gating ?? { mode: 'auto', minAluBwRatio: 0.0 };
    
    this.currentSeqLen = 0;
    
    this.totalTokensSeen = 0;
    
    this.memoryUsage = 0;
    
    this.gpuContext = null;
    
    this.coldStore = caches?.coldStore ?? null;
    
    this.coldStorePartition = caches?.coldStorePartition ?? 'kv-cache';
    
    this.coldStoreRegistered = false;
    
    this.coldStoreChunks = [];

    if (this.kvDtype !== 'f16' || this.coldDtype !== 'f16') {
      throw new Error('TieredKVCache currently requires f16 KV storage.');
    }

    this.coldQuantMode = this._resolveCompressionMode(this.compression, this.gating);
    if (this.coldQuantMode !== 'none' && this.compression.blockSize !== 1) {
      throw new Error('TieredKVCache compression.blockSize must be 1 (per-token) for int8/int4 cold tiers.');
    }
    this.coldPackedStride = this.coldQuantMode === 'int4'
      ? Math.ceil(this.headDim / 8)
      : Math.ceil(this.headDim / 4);

    if (this.coldQuantMode !== 'none' && !this.useGPU) {
      throw new Error('TieredKVCache quantization requires GPU.');
    }
    if (this.coldQuantMode !== 'none' && this.headDim > 256) {
      throw new Error('TieredKVCache quantization requires headDim <= 256.');
    }

    if (caches) {
      this.hotCache = caches.hotCache;
      this.coldCache = caches.coldCache ?? null;
      this.coldLayers = caches.coldLayers ?? null;
    } else {
      this.hotCache = new SlidingWindowKVCache({
        numLayers: config.numLayers,
        numHeads: config.numHeads,
        headDim: config.headDim,
        maxSeqLen: this.hotWindow,
        useGPU: config.useGPU,
        layout: 'contiguous',
        pageSize: config.pageSize,
        kvDtype: config.kvDtype,
        windowSize: this.hotWindow,
      });

      if (this.coldQuantMode === 'none') {
        this.coldCache = new KVCache({
          numLayers: config.numLayers,
          numHeads: config.numHeads,
          headDim: config.headDim,
          maxSeqLen: config.maxSeqLen,
          useGPU: config.useGPU,
          layout: 'paged',
          pageSize: this.coldPageSize,
          kvDtype: this.coldDtype,
        });
        this.coldLayers = null;
      } else {
        this.coldCache = null;
        this.coldLayers = this._createColdQuantizedLayers();
      }
    }

    if (this.coldCache) {
      this.memoryUsage = this.hotCache.memoryUsage + this.coldCache.memoryUsage;
    } else {
      this.memoryUsage = this.hotCache.memoryUsage + this._coldQuantizedBytes();
    }
  }

  
  _resolveCompressionMode(compression, gating) {
    const requested = compression?.mode ?? 'none';
    if (gating?.mode === 'force_off') return 'none';
    if (gating?.mode === 'force_on') return requested;
    if (gating?.mode === 'auto' && gating.minAluBwRatio > 0) {
      const ratio = 1.0;
      if (ratio < gating.minAluBwRatio) return 'none';
    }
    return requested;
  }

  
  _coldQuantizedBytes() {
    if (this.coldQuantMode === 'none') return 0;
    const packedStride = this.coldPackedStride;
    const packedBytesPerToken = this.numHeads * packedStride * 4;
    const scaleBytesPerToken = this.numHeads * 2;
    return this.numLayers * this.maxSeqLen * ((packedBytesPerToken * 2) + (scaleBytesPerToken * 2));
  }

  
  _createColdQuantizedLayers() {
    const device = getDevice();
    if (!device) {
      throw new Error('GPU device not initialized.');
    }
    const layers = new Array(this.numLayers);
    const packedStride = this.coldPackedStride;
    const packedBytes = this.maxSeqLen * this.numHeads * packedStride * 4;
    const scalesBytes = this.maxSeqLen * this.numHeads * 2;

    for (let l = 0; l < this.numLayers; l++) {
      const keysPackedGPU = device.createBuffer({
        label: `kv_cache_cold_keys_packed_layer_${l}`,
        size: packedBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      const valuesPackedGPU = device.createBuffer({
        label: `kv_cache_cold_values_packed_layer_${l}`,
        size: packedBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      const scalesKGPU = device.createBuffer({
        label: `kv_cache_cold_scales_k_layer_${l}`,
        size: scalesBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      const scalesVGPU = device.createBuffer({
        label: `kv_cache_cold_scales_v_layer_${l}`,
        size: scalesBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      layers[l] = {
        keysPackedGPU,
        valuesPackedGPU,
        scalesKGPU,
        scalesVGPU,
        seqLen: 0,
      };
    }
    return layers;
  }

  
  _getColdStoreBytes() {
    if (this.coldCache) {
      return this.coldCache.getMemoryStats().theoretical;
    }
    return this._coldQuantizedBytes();
  }

  
  async _registerColdStoreBuffers() {
    if (!this.coldStore || this.coldStoreRegistered) return;

    if (typeof this.coldStore.initialize === 'function') {
      await this.coldStore.initialize();
    }
    if (typeof this.coldStore.createPartition === 'function') {
      await this.coldStore.createPartition({
        name: this.coldStorePartition,
        maxBytes: this._getColdStoreBytes(),
        opfsPath: this.coldStorePartition,
      });
    }

    if (typeof this.coldStore.registerVramBuffer !== 'function') {
      this.coldStoreRegistered = true;
      return;
    }

    const register = (buffer, label) => {
      if (!buffer) return;
      const sizeBytes = buffer.size;
      if (!Number.isFinite(sizeBytes) || sizeBytes <= 0) {
        throw new Error('TieredKVCache cold store requires GPU buffer sizes.');
      }
      const id = this.coldStore.registerVramBuffer(
        this.coldStorePartition,
        buffer,
        sizeBytes,
        label,
        { locked: true }
      );
      this.coldStoreChunks.push(id);
    };

    if (this.coldCache) {
      for (let l = 0; l < this.numLayers; l++) {
        const layer = this.coldCache.layers[l];
        register(layer.keysGPU, `kv_cache_cold_keys_${l}`);
        register(layer.valuesGPU, `kv_cache_cold_values_${l}`);
      }
    } else if (this.coldLayers) {
      for (let l = 0; l < this.numLayers; l++) {
        const layer = this.coldLayers[l];
        register(layer.keysPackedGPU, `kv_cache_cold_keys_packed_${l}`);
        register(layer.valuesPackedGPU, `kv_cache_cold_values_packed_${l}`);
        register(layer.scalesKGPU, `kv_cache_cold_scales_k_${l}`);
        register(layer.scalesVGPU, `kv_cache_cold_scales_v_${l}`);
      }
    }

    this.coldStoreRegistered = true;
  }

  
  clear() {
    this.hotCache.clear();
    if (this.coldCache) {
      this.coldCache.clear();
    } else if (this.coldLayers) {
      for (const layer of this.coldLayers) {
        layer.seqLen = 0;
      }
    }
    this.currentSeqLen = 0;
    this.totalTokensSeen = 0;
  }

  
  update(layerIdx, keys, values, startPos = this.currentSeqLen) {
    if (!this.coldCache) {
      throw new Error('TieredKVCache quantized mode requires GPU update paths.');
    }
    this.coldCache.update(layerIdx, keys, values, startPos);
    this.hotCache.update(layerIdx, keys, values);
    this.currentSeqLen = this.coldCache.currentSeqLen;
    this.totalTokensSeen = this.coldCache.totalTokensSeen;
  }

  
  async updateFromGPU(layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
    if (!Number.isInteger(startPos) || startPos < 0) {
      throw new Error('TieredKVCache updateFromGPU requires a non-negative startPos.');
    }
    if (!Number.isInteger(numTokens) || numTokens < 0) {
      throw new Error('TieredKVCache updateFromGPU requires a non-negative integer token count.');
    }
    if (numTokens === 0) {
      return;
    }
    if (startPos + numTokens > this.maxSeqLen) {
      throw new Error(
        `Cache overflow: ${startPos + numTokens} > ${this.maxSeqLen}`
      );
    }

    await this._registerColdStoreBuffers();

    if (this.coldCache) {
      this.coldCache.updateFromGPU(layerIdx, keysBuffer, valuesBuffer, startPos, numTokens);
      this.currentSeqLen = this.coldCache.currentSeqLen;
      this.totalTokensSeen = this.coldCache.totalTokensSeen;
    } else {
      const layer = this.coldLayers[layerIdx];
      await runKVQuantize(
        keysBuffer,
        valuesBuffer,
        layer.keysPackedGPU,
        layer.valuesPackedGPU,
        layer.scalesKGPU,
        layer.scalesVGPU,
        {
          numKVHeads: this.numHeads,
          headDim: this.headDim,
          startPos,
          numTokens,
          packedStride: this.coldPackedStride,
          mode: this.coldQuantMode,
        }
      );
      layer.seqLen = Math.max(layer.seqLen, startPos + numTokens);
      this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numTokens);
      this.totalTokensSeen = Math.max(this.totalTokensSeen, startPos + numTokens);
    }
    this.hotCache.updateFromGPU(layerIdx, keysBuffer, valuesBuffer, startPos, numTokens);
  }

  
  async recordUpdateFromGPU(recorder, layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
    if (!Number.isInteger(startPos) || startPos < 0) {
      throw new Error('TieredKVCache recordUpdateFromGPU requires a non-negative startPos.');
    }
    if (!Number.isInteger(numTokens) || numTokens < 0) {
      throw new Error('TieredKVCache recordUpdateFromGPU requires a non-negative integer token count.');
    }
    if (numTokens === 0) {
      return;
    }
    if (startPos + numTokens > this.maxSeqLen) {
      throw new Error(
        `Cache overflow: ${startPos + numTokens} > ${this.maxSeqLen}`
      );
    }

    await this._registerColdStoreBuffers();

    if (this.coldCache) {
      this.coldCache.recordUpdateFromGPU(recorder, layerIdx, keysBuffer, valuesBuffer, startPos, numTokens);
    } else {
      const layer = this.coldLayers[layerIdx];
      await recordKVQuantize(
        recorder,
        keysBuffer,
        valuesBuffer,
        layer.keysPackedGPU,
        layer.valuesPackedGPU,
        layer.scalesKGPU,
        layer.scalesVGPU,
        {
          numKVHeads: this.numHeads,
          headDim: this.headDim,
          startPos,
          numTokens,
          packedStride: this.coldPackedStride,
          mode: this.coldQuantMode,
        }
      );
      layer.seqLen = Math.max(layer.seqLen, startPos + numTokens);
    }
    this.hotCache.recordUpdateFromGPU(recorder, layerIdx, keysBuffer, valuesBuffer, startPos, numTokens);
    this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numTokens);
    this.totalTokensSeen = Math.max(this.totalTokensSeen, startPos + numTokens);
  }

  
  get(layerIdx) {
    if (!this.coldCache) {
      throw new Error('TieredKVCache quantized mode does not support CPU reads.');
    }
    return this.coldCache.get(layerIdx);
  }

  
  getGPUBuffers(layerIdx) {
    const hot = this.hotCache.getGPUBuffers(layerIdx);
    if (!hot) return null;

    const totalSeqLen = this.currentSeqLen;
    const hotLen = Math.min(totalSeqLen, this.hotWindow);
    const hotStart = totalSeqLen > hotLen ? (totalSeqLen - hotLen) : 0;
    const coldLen = totalSeqLen - hotLen;

    if (this.coldCache) {
      const cold = this.coldCache.getGPUBuffers(layerIdx);
      if (!cold) return null;
      return {
        layout: 'tiered',
        seqLen: totalSeqLen,
        hotKeysGPU: hot.keysGPU,
        hotValuesGPU: hot.valuesGPU,
        hotSeqLen: hotLen,
        hotStart,
        hotWindow: this.hotWindow,
        coldKeysGPU: cold.keysGPU,
        coldValuesGPU: cold.valuesGPU,
        coldSeqLen: coldLen,
        coldPageTableGPU: cold.pageTableGPU,
        coldPageSize: cold.pageSize ?? this.coldPageSize,
        coldQuantMode: 'none',
      };
    }

    const coldLayer = this.coldLayers[layerIdx];
    return {
      layout: 'tiered',
      seqLen: totalSeqLen,
      hotKeysGPU: hot.keysGPU,
      hotValuesGPU: hot.valuesGPU,
      hotSeqLen: hotLen,
      hotStart,
      hotWindow: this.hotWindow,
      coldKeysGPU: coldLayer.keysPackedGPU,
      coldValuesGPU: coldLayer.valuesPackedGPU,
      coldScalesKGPU: coldLayer.scalesKGPU,
      coldScalesVGPU: coldLayer.scalesVGPU,
      coldSeqLen: coldLen,
      coldPageTableGPU: null,
      coldPageSize: 0,
      coldPackedStride: this.coldPackedStride,
      coldQuantMode: this.coldQuantMode,
    };
  }

  
  hasGPUCache() {
    if (!this.hotCache.hasGPUCache()) return false;
    if (this.coldCache) return this.coldCache.hasGPUCache();
    return Array.isArray(this.coldLayers);
  }

  
  truncate(length) {
    if (this.coldCache) {
      this.coldCache.truncate(length);
      this.currentSeqLen = this.coldCache.currentSeqLen;
    } else {
      this.currentSeqLen = Math.min(this.currentSeqLen, length);
      if (this.coldLayers) {
        for (const layer of this.coldLayers) {
          layer.seqLen = Math.min(layer.seqLen, length);
        }
      }
    }
    this.totalTokensSeen = Math.min(this.totalTokensSeen, this.currentSeqLen);
  }

  
  getMemoryStats() {
    const hotStats = this.hotCache.getMemoryStats();
    const coldStats = this.coldCache ? this.coldCache.getMemoryStats() : {
      theoretical: this._coldQuantizedBytes(),
      allocated: this._coldQuantizedBytes(),
      used: this._coldQuantizedBytes(),
      efficiency: 1.0,
      seqLen: this.currentSeqLen,
      maxSeqLen: this.maxSeqLen,
      layout: 'paged',
    };
    return {
      theoretical: hotStats.theoretical + coldStats.theoretical,
      allocated: hotStats.allocated + coldStats.allocated,
      used: hotStats.used + coldStats.used,
      efficiency: (hotStats.used + coldStats.used) / (hotStats.allocated + coldStats.allocated),
      seqLen: this.currentSeqLen,
      maxSeqLen: this.maxSeqLen,
      layout: this.layout,
      hot: hotStats,
      cold: coldStats,
    };
  }

  
  setGPUContext(gpuContext) {
    this.gpuContext = gpuContext;
    this.hotCache.setGPUContext(gpuContext);
    if (this.coldCache) {
      this.coldCache.setGPUContext(gpuContext);
    }
  }

  
  destroy() {
    this.hotCache.destroy();
    if (this.coldCache) {
      this.coldCache.destroy();
    } else if (this.coldLayers) {
      for (const layer of this.coldLayers) {
        layer.keysPackedGPU.destroy();
        layer.valuesPackedGPU.destroy();
        layer.scalesKGPU.destroy();
        layer.scalesVGPU.destroy();
      }
    }
  }

  
  clone() {
    const hotClone = this.hotCache.clone();
    const coldClone = this.coldCache ? this.coldCache.clone() : null;
    const cloned = new TieredKVCache({
      numLayers: this.numLayers,
      numHeads: this.numHeads,
      headDim: this.headDim,
      maxSeqLen: this.maxSeqLen,
      useGPU: this.useGPU,
      layout: 'tiered',
      kvDtype: this.kvDtype,
      pageSize: this.coldPageSize,
      tiering: {
        mode: this.tieringMode,
        hotWindow: this.hotWindow,
        coldPageSize: this.coldPageSize,
        coldDtype: this.coldDtype,
        compression: this.compression,
        gating: this.gating,
      },
    }, { hotCache: hotClone, coldCache: coldClone, coldLayers: null });

    if (!coldClone && this.coldLayers) {
      cloned.coldLayers = cloned._createColdQuantizedLayers();
      const device = getDevice();
      if (!device) {
        throw new Error('GPU device not initialized');
      }
      const packedStride = this.coldPackedStride;
      const packedBytesPerToken = this.numHeads * packedStride * 4;
      const scalesBytesPerToken = this.numHeads * 2;
      for (let l = 0; l < this.numLayers; l++) {
        const src = this.coldLayers[l];
        const dst = cloned.coldLayers[l];
        const usedTokens = src.seqLen;
        if (usedTokens > 0) {
          const packedBytes = usedTokens * packedBytesPerToken;
          const scalesBytes = usedTokens * scalesBytesPerToken;
          const encoder = device.createCommandEncoder({ label: `kv_cache_cold_clone_${l}` });
          encoder.copyBufferToBuffer(src.keysPackedGPU, 0, dst.keysPackedGPU, 0, packedBytes);
          encoder.copyBufferToBuffer(src.valuesPackedGPU, 0, dst.valuesPackedGPU, 0, packedBytes);
          encoder.copyBufferToBuffer(src.scalesKGPU, 0, dst.scalesKGPU, 0, scalesBytes);
          encoder.copyBufferToBuffer(src.scalesVGPU, 0, dst.scalesVGPU, 0, scalesBytes);
          device.queue.submit([encoder.finish()]);
          dst.seqLen = src.seqLen;
        }
      }
    }

    cloned.currentSeqLen = this.currentSeqLen;
    cloned.totalTokensSeen = this.totalTokensSeen;
    return cloned;
  }
}
