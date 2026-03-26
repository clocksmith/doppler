
import { getDevice } from '../../gpu/device.js';
import { runKVQuantize, recordKVQuantize } from '../../gpu/kernel-selector.js';
import { KVCache } from './base.js';
import { SlidingWindowKVCache } from './sliding-window.js';
import {
  computePackedStride,
  retainTurboQuantSharedBuffers,
} from '../../gpu/kernels/turboquant-codebook.js';

function isTurboQuantMode(mode) {
  return mode === 'turboquant' || mode === 'turboquant_prod';
}

function assertSupportedTurboQuantMode(mode, label) {
  if (mode !== 'turboquant_outlier') {
    return mode;
  }
  throw new Error(
    `TieredKVCache ${label}="${mode}" is not supported yet. ` +
    'TurboQuant outlier high-precision buffers and decode kernels are not wired end to end.'
  );
}

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
    
    const turboQuantModes = ['turboquant', 'turboquant_prod'];
    const defaultCompressionMode = turboQuantModes.includes(tiering.mode)
      ? tiering.mode
      : (tiering.mode === 'int8' ? 'int8' : (tiering.mode === 'int4' ? 'int4' : 'none'));
    this.compression = tiering.compression ?? { mode: defaultCompressionMode, blockSize: 1 };
    
    this.gating = tiering.gating ?? { mode: 'force_off', minAluBwRatio: 0.0 };
    
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

    assertSupportedTurboQuantMode(this.tieringMode, 'tiering.mode');
    assertSupportedTurboQuantMode(this.compression?.mode ?? 'none', 'tiering.compression.mode');
    this.coldQuantMode = this._resolveCompressionMode(this.compression, this.gating);
    assertSupportedTurboQuantMode(this.coldQuantMode, 'resolved cold quant mode');
    if (this.coldQuantMode !== 'none' && this.compression.blockSize !== 1) {
      throw new Error('TieredKVCache compression.blockSize must be 1 (per-token) for quantized cold tiers.');
    }

    this.isTurboQuant = isTurboQuantMode(this.coldQuantMode);
    this.isProdMode = this.coldQuantMode === 'turboquant_prod';
    this.turboQuantBitWidth = this.compression.bitWidth ?? 4;

    if (this.isTurboQuant) {
      this.coldPackedStride = computePackedStride(this.headDim, this.turboQuantBitWidth);
      if (this.isProdMode) {
        const mseBitWidth = this.turboQuantBitWidth - 1;
        this.msePackedStride = computePackedStride(this.headDim, mseBitWidth);
        this.residualPackedStride = Math.ceil(this.headDim / 32);
      }
    } else {
      this.coldPackedStride = this.coldQuantMode === 'int4'
        ? Math.ceil(this.headDim / 8)
        : Math.ceil(this.headDim / 4);
    }

    // Shared TurboQuant GPU buffers (allocated lazily in _createColdQuantizedLayers)
    this.rotationMatrixBuffer = null;
    this.codebookCentroidsBuffer = null;
    this.codebookBoundariesBuffer = null;
    this.qjlMatrixBuffer = null;
    this.releaseSharedBuffers = null;

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
      throw new Error(
        'TieredKVCache auto compression gating requires an explicit measured ALU/BW ratio. ' +
        'Use gating.mode="force_on"/"force_off" or set minAluBwRatio to 0.'
      );
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

    // Upload shared TurboQuant buffers if needed
    if (this.isTurboQuant) {
      const sharedBuffers = retainTurboQuantSharedBuffers(device, {
        headDim: this.headDim,
        bitWidth: this.turboQuantBitWidth,
        prodMode: this.isProdMode,
      });
      this.rotationMatrixBuffer = sharedBuffers.rotationMatrixBuffer;
      this.codebookCentroidsBuffer = sharedBuffers.codebookCentroidsBuffer;
      this.codebookBoundariesBuffer = sharedBuffers.codebookBoundariesBuffer;
      this.qjlMatrixBuffer = sharedBuffers.qjlMatrixBuffer;
      this.releaseSharedBuffers = sharedBuffers.release;
    }

    const layers = new Array(this.numLayers);
    const scalesBytes = this.maxSeqLen * this.numHeads * 2;
    const bufUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

    if (this.isProdMode) {
      const msePackedBytes = this.maxSeqLen * this.numHeads * this.msePackedStride * 4;
      const resPackedBytes = this.maxSeqLen * this.numHeads * this.residualPackedStride * 4;

      for (let l = 0; l < this.numLayers; l++) {
        layers[l] = {
          keysPackedGPU: device.createBuffer({ label: `kv_cache_cold_keys_mse_${l}`, size: msePackedBytes, usage: bufUsage }),
          valuesPackedGPU: device.createBuffer({ label: `kv_cache_cold_values_mse_${l}`, size: msePackedBytes, usage: bufUsage }),
          residualKGPU: device.createBuffer({ label: `kv_cache_cold_res_k_${l}`, size: resPackedBytes, usage: bufUsage }),
          residualVGPU: device.createBuffer({ label: `kv_cache_cold_res_v_${l}`, size: resPackedBytes, usage: bufUsage }),
          residualNormsKGPU: device.createBuffer({ label: `kv_cache_cold_rnorm_k_${l}`, size: scalesBytes, usage: bufUsage }),
          residualNormsVGPU: device.createBuffer({ label: `kv_cache_cold_rnorm_v_${l}`, size: scalesBytes, usage: bufUsage }),
          scalesKGPU: device.createBuffer({ label: `kv_cache_cold_scales_k_${l}`, size: scalesBytes, usage: bufUsage }),
          scalesVGPU: device.createBuffer({ label: `kv_cache_cold_scales_v_${l}`, size: scalesBytes, usage: bufUsage }),
          seqLen: 0,
        };
      }
    } else {
      const packedStride = this.isTurboQuant ? this.coldPackedStride : this.coldPackedStride;
      const packedBytes = this.maxSeqLen * this.numHeads * packedStride * 4;

      for (let l = 0; l < this.numLayers; l++) {
        layers[l] = {
          keysPackedGPU: device.createBuffer({ label: `kv_cache_cold_keys_packed_layer_${l}`, size: packedBytes, usage: bufUsage }),
          valuesPackedGPU: device.createBuffer({ label: `kv_cache_cold_values_packed_layer_${l}`, size: packedBytes, usage: bufUsage }),
          scalesKGPU: device.createBuffer({ label: `kv_cache_cold_scales_k_layer_${l}`, size: scalesBytes, usage: bufUsage }),
          scalesVGPU: device.createBuffer({ label: `kv_cache_cold_scales_v_layer_${l}`, size: scalesBytes, usage: bufUsage }),
          seqLen: 0,
        };
      }
    }
    return layers;
  }

  
  _buildQuantizeOptions(layer, startPos, numTokens) {
    const base = {
      numKVHeads: this.numHeads,
      headDim: this.headDim,
      startPos,
      numTokens,
      mode: this.coldQuantMode,
    };

    if (this.isProdMode) {
      return {
        outputKeysBuffer: layer.keysPackedGPU,
        outputValuesBuffer: layer.valuesPackedGPU,
        options: {
          ...base,
          packedStride: this.msePackedStride,
          rotationMatrixBuffer: this.rotationMatrixBuffer,
          codebookCentroidsBuffer: this.codebookCentroidsBuffer,
          codebookBoundariesBuffer: this.codebookBoundariesBuffer,
          qjlMatrixBuffer: this.qjlMatrixBuffer,
          residualKBuffer: layer.residualKGPU,
          residualVBuffer: layer.residualVGPU,
          residualNormsKBuffer: layer.residualNormsKGPU,
          residualNormsVBuffer: layer.residualNormsVGPU,
          residualPackedStride: this.residualPackedStride,
          bitWidth: this.turboQuantBitWidth - 1,
        },
      };
    }

    if (this.isTurboQuant) {
      return {
        outputKeysBuffer: layer.keysPackedGPU,
        outputValuesBuffer: layer.valuesPackedGPU,
        options: {
          ...base,
          packedStride: this.coldPackedStride,
          rotationMatrixBuffer: this.rotationMatrixBuffer,
          codebookCentroidsBuffer: this.codebookCentroidsBuffer,
          codebookBoundariesBuffer: this.codebookBoundariesBuffer,
          bitWidth: this.turboQuantBitWidth,
        },
      };
    }

    return {
      outputKeysBuffer: layer.keysPackedGPU,
      outputValuesBuffer: layer.valuesPackedGPU,
      options: {
        ...base,
        packedStride: this.coldPackedStride,
      },
    };
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
      const quantOpts = this._buildQuantizeOptions(layer, startPos, numTokens);
      await runKVQuantize(
        keysBuffer,
        valuesBuffer,
        quantOpts.outputKeysBuffer,
        quantOpts.outputValuesBuffer,
        layer.scalesKGPU,
        layer.scalesVGPU,
        quantOpts.options
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
      const quantOpts = this._buildQuantizeOptions(layer, startPos, numTokens);
      await recordKVQuantize(
        recorder,
        keysBuffer,
        valuesBuffer,
        quantOpts.outputKeysBuffer,
        quantOpts.outputValuesBuffer,
        layer.scalesKGPU,
        layer.scalesVGPU,
        quantOpts.options
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
    const result = {
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
      coldPackedStride: this.isProdMode ? this.msePackedStride : this.coldPackedStride,
      coldQuantMode: this.coldQuantMode,
    };

    if (this.isTurboQuant) {
      result.rotationMatrixBuffer = this.rotationMatrixBuffer;
      result.codebookCentroidsBuffer = this.codebookCentroidsBuffer;
    }
    if (this.isProdMode) {
      result.residualKGPU = coldLayer.residualKGPU;
      result.residualVGPU = coldLayer.residualVGPU;
      result.residualNormsKGPU = coldLayer.residualNormsKGPU;
      result.residualNormsVGPU = coldLayer.residualNormsVGPU;
      result.qjlMatrixBuffer = this.qjlMatrixBuffer;
      result.residualPackedStride = this.residualPackedStride;
    }

    return result;
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
        layer.keysPackedGPU?.destroy();
        layer.valuesPackedGPU?.destroy();
        layer.scalesKGPU?.destroy();
        layer.scalesVGPU?.destroy();
        layer.residualKGPU?.destroy();
        layer.residualVGPU?.destroy();
        layer.residualNormsKGPU?.destroy();
        layer.residualNormsVGPU?.destroy();
      }
    }
    if (this.releaseSharedBuffers) {
      this.releaseSharedBuffers();
    } else {
      this.rotationMatrixBuffer?.destroy();
      this.codebookCentroidsBuffer?.destroy();
      this.codebookBoundariesBuffer?.destroy();
      this.qjlMatrixBuffer?.destroy();
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
      const mseBytesPerToken = this.isProdMode ? this.numHeads * this.msePackedStride * 4 : packedBytesPerToken;
      const residualBytesPerToken = this.isProdMode ? this.numHeads * this.residualPackedStride * 4 : 0;
      for (let l = 0; l < this.numLayers; l++) {
        const src = this.coldLayers[l];
        const dst = cloned.coldLayers[l];
        const usedTokens = src.seqLen;
        if (usedTokens > 0) {
          const packedBytes = usedTokens * mseBytesPerToken;
          const scalesBytes = usedTokens * scalesBytesPerToken;
          const encoder = device.createCommandEncoder({ label: `kv_cache_cold_clone_${l}` });
          encoder.copyBufferToBuffer(src.keysPackedGPU, 0, dst.keysPackedGPU, 0, packedBytes);
          encoder.copyBufferToBuffer(src.valuesPackedGPU, 0, dst.valuesPackedGPU, 0, packedBytes);
          encoder.copyBufferToBuffer(src.scalesKGPU, 0, dst.scalesKGPU, 0, scalesBytes);
          encoder.copyBufferToBuffer(src.scalesVGPU, 0, dst.scalesVGPU, 0, scalesBytes);
          if (this.isProdMode) {
            const residualBytes = usedTokens * residualBytesPerToken;
            encoder.copyBufferToBuffer(src.residualKGPU, 0, dst.residualKGPU, 0, residualBytes);
            encoder.copyBufferToBuffer(src.residualVGPU, 0, dst.residualVGPU, 0, residualBytes);
            encoder.copyBufferToBuffer(src.residualNormsKGPU, 0, dst.residualNormsKGPU, 0, scalesBytes);
            encoder.copyBufferToBuffer(src.residualNormsVGPU, 0, dst.residualNormsVGPU, 0, scalesBytes);
          }
          device.queue.submit([encoder.finish()]);
        }
        dst.seqLen = src.seqLen;
      }
    }

    cloned.currentSeqLen = this.currentSeqLen;
    cloned.totalTokensSeen = this.totalTokensSeen;
    return cloned;
  }
}
