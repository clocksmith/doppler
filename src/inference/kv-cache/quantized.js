
import { getDevice } from '../../gpu/device.js';
import { runKVQuantize, recordKVQuantize } from '../../gpu/kernel-selector.js';

// =============================================================================
// QuantizedKVCache — Contiguous quantized KV cache for full-attention models
//
// All KV entries are quantized (no hot/cold split). For models like Qwen 3.5
// and LFM that force contiguous layout due to full global attention.
// =============================================================================

export class QuantizedKVCache {

  constructor(config) {
    if (!config) {
      throw new Error('QuantizedKVCache requires a config.');
    }
    if (!config.quantMode) {
      throw new Error('QuantizedKVCache requires a quantMode.');
    }

    this.numLayers = config.numLayers;
    this.numHeads = config.numHeads;
    this.headDim = config.headDim;
    this.maxSeqLen = config.maxSeqLen;
    this.useGPU = config.useGPU;
    this.layout = 'contiguous_quantized';
    this.kvDtype = config.kvDtype ?? 'f16';
    this.quantMode = config.quantMode;
    this.bitWidth = config.bitWidth ?? 4;
    this.prodMode = config.prodMode === true;
    this.currentSeqLen = 0;
    this.totalTokensSeen = 0;

    if (this.kvDtype !== 'f16') {
      throw new Error('QuantizedKVCache requires f16 KV input.');
    }
    if (this.quantMode === 'turboquant_outlier') {
      throw new Error(
        'QuantizedKVCache quantMode="turboquant_outlier" is not supported yet. ' +
        'TurboQuant outlier high-precision buffers and decode kernels are not wired end to end.'
      );
    }
    if (!this.useGPU) {
      throw new Error('QuantizedKVCache requires GPU.');
    }
    if (this.headDim > 256) {
      throw new Error('QuantizedKVCache requires headDim <= 256.');
    }

    // Packing parameters
    this.packFactor = Math.floor(32 / this.bitWidth);
    this.packedStride = Math.ceil(this.headDim / this.packFactor);

    // Prod-mode packing (b-1 bits for MSE, 1-bit for residual)
    if (this.prodMode) {
      this.mseBitWidth = this.bitWidth - 1;
      this.msePackFactor = Math.floor(32 / this.mseBitWidth);
      this.msePackedStride = Math.ceil(this.headDim / this.msePackFactor);
      this.residualPackedStride = Math.ceil(this.headDim / 32);
    }

    // Shared buffers (rotation matrix, codebook, QJL matrix)
    this.rotationMatrixBuffer = null;
    this.codebookCentroidsBuffer = null;
    this.codebookBoundariesBuffer = null;
    this.qjlMatrixBuffer = null;
    this.releaseSharedBuffers = null;

    // Per-layer quantized storage
    this.layers = this._createLayers();
    this.memoryUsage = this._computeMemoryUsage();
  }

  _createLayers() {
    const device = getDevice();
    if (!device) {
      throw new Error('GPU device not initialized.');
    }

    const layers = new Array(this.numLayers);
    for (let l = 0; l < this.numLayers; l++) {
      layers[l] = this._createLayerBuffers(device, l);
    }
    return layers;
  }

  _createLayerBuffers(device, layerIdx) {
    const layer = { seqLen: 0 };

    if (this.prodMode) {
      // MSE stage buffers
      const msePacked = this.maxSeqLen * this.numHeads * this.msePackedStride * 4;
      layer.keysPackedMSE = device.createBuffer({
        label: `qkv_cache_keys_mse_L${layerIdx}`,
        size: msePacked,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      layer.valuesPackedMSE = device.createBuffer({
        label: `qkv_cache_values_mse_L${layerIdx}`,
        size: msePacked,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });

      // Residual buffers (1-bit packed)
      const resPacked = this.maxSeqLen * this.numHeads * this.residualPackedStride * 4;
      layer.residualK = device.createBuffer({
        label: `qkv_cache_res_k_L${layerIdx}`,
        size: resPacked,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      layer.residualV = device.createBuffer({
        label: `qkv_cache_res_v_L${layerIdx}`,
        size: resPacked,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });

      // Residual norms
      const normBytes = this.maxSeqLen * this.numHeads * 2;
      layer.residualNormsK = device.createBuffer({
        label: `qkv_cache_rnorm_k_L${layerIdx}`,
        size: normBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      layer.residualNormsV = device.createBuffer({
        label: `qkv_cache_rnorm_v_L${layerIdx}`,
        size: normBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    } else {
      // MSE-only buffers
      const packedBytes = this.maxSeqLen * this.numHeads * this.packedStride * 4;
      layer.keysPacked = device.createBuffer({
        label: `qkv_cache_keys_packed_L${layerIdx}`,
        size: packedBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      layer.valuesPacked = device.createBuffer({
        label: `qkv_cache_values_packed_L${layerIdx}`,
        size: packedBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    }

    // Scales (shared between MSE-only and prod)
    const scalesBytes = this.maxSeqLen * this.numHeads * 2;
    layer.scalesK = device.createBuffer({
      label: `qkv_cache_scales_k_L${layerIdx}`,
      size: scalesBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    layer.scalesV = device.createBuffer({
      label: `qkv_cache_scales_v_L${layerIdx}`,
      size: scalesBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    return layer;
  }

  _computeMemoryUsage() {
    let bytesPerToken;
    if (this.prodMode) {
      const mseBytes = this.numHeads * this.msePackedStride * 4 * 2; // K + V
      const resBytes = this.numHeads * this.residualPackedStride * 4 * 2;
      const normBytes = this.numHeads * 2 * 4; // scales + residual norms, K + V
      bytesPerToken = mseBytes + resBytes + normBytes;
    } else {
      const packedBytes = this.numHeads * this.packedStride * 4 * 2;
      const scaleBytes = this.numHeads * 2 * 2;
      bytesPerToken = packedBytes + scaleBytes;
    }
    return this.numLayers * this.maxSeqLen * bytesPerToken;
  }

  /**
   * Set shared TurboQuant buffers (rotation matrix, codebook, QJL).
   * Must be called before first updateFromGPU.
   */
  setSharedBuffers(buffers) {
    this.rotationMatrixBuffer = buffers.rotationMatrixBuffer;
    this.codebookCentroidsBuffer = buffers.codebookCentroidsBuffer;
    this.codebookBoundariesBuffer = buffers.codebookBoundariesBuffer;
    this.releaseSharedBuffers = typeof buffers.release === 'function' ? buffers.release : null;
    if (this.prodMode) {
      this.qjlMatrixBuffer = buffers.qjlMatrixBuffer;
    }
  }

  async updateFromGPU(layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
    if (!Number.isInteger(startPos) || startPos < 0) {
      throw new Error('QuantizedKVCache updateFromGPU requires non-negative startPos.');
    }
    if (!Number.isInteger(numTokens) || numTokens <= 0) {
      return;
    }
    if (startPos + numTokens > this.maxSeqLen) {
      throw new Error(`Cache overflow: ${startPos + numTokens} > ${this.maxSeqLen}`);
    }

    const layer = this.layers[layerIdx];
    const quantOpts = this._buildQuantizeOptions(layer, startPos, numTokens);

    await runKVQuantize(
      keysBuffer, valuesBuffer,
      quantOpts.outputKeysBuffer, quantOpts.outputValuesBuffer,
      layer.scalesK, layer.scalesV,
      quantOpts.options
    );

    layer.seqLen = Math.max(layer.seqLen, startPos + numTokens);
    this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numTokens);
    this.totalTokensSeen = Math.max(this.totalTokensSeen, startPos + numTokens);
  }

  async recordUpdateFromGPU(recorder, layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
    if (!Number.isInteger(startPos) || startPos < 0) {
      throw new Error('QuantizedKVCache recordUpdateFromGPU requires non-negative startPos.');
    }
    if (!Number.isInteger(numTokens) || numTokens <= 0) {
      return;
    }
    if (startPos + numTokens > this.maxSeqLen) {
      throw new Error(`Cache overflow: ${startPos + numTokens} > ${this.maxSeqLen}`);
    }

    const layer = this.layers[layerIdx];
    const quantOpts = this._buildQuantizeOptions(layer, startPos, numTokens);

    await recordKVQuantize(
      recorder,
      keysBuffer, valuesBuffer,
      quantOpts.outputKeysBuffer, quantOpts.outputValuesBuffer,
      layer.scalesK, layer.scalesV,
      quantOpts.options
    );

    layer.seqLen = Math.max(layer.seqLen, startPos + numTokens);
    this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numTokens);
    this.totalTokensSeen = Math.max(this.totalTokensSeen, startPos + numTokens);
  }

  _buildQuantizeOptions(layer, startPos, numTokens) {
    if (this.prodMode) {
      return {
        outputKeysBuffer: layer.keysPackedMSE,
        outputValuesBuffer: layer.valuesPackedMSE,
        options: {
          numKVHeads: this.numHeads,
          headDim: this.headDim,
          startPos,
          numTokens,
          packedStride: this.msePackedStride,
          mode: 'turboquant_prod',
          rotationMatrixBuffer: this.rotationMatrixBuffer,
          codebookCentroidsBuffer: this.codebookCentroidsBuffer,
          codebookBoundariesBuffer: this.codebookBoundariesBuffer,
          qjlMatrixBuffer: this.qjlMatrixBuffer,
          residualKBuffer: layer.residualK,
          residualVBuffer: layer.residualV,
          residualNormsKBuffer: layer.residualNormsK,
          residualNormsVBuffer: layer.residualNormsV,
          residualPackedStride: this.residualPackedStride,
          bitWidth: this.mseBitWidth,
        },
      };
    }

    return {
      outputKeysBuffer: layer.keysPacked,
      outputValuesBuffer: layer.valuesPacked,
      options: {
        numKVHeads: this.numHeads,
        headDim: this.headDim,
        startPos,
        numTokens,
        packedStride: this.packedStride,
        mode: this.quantMode,
        rotationMatrixBuffer: this.rotationMatrixBuffer,
        codebookCentroidsBuffer: this.codebookCentroidsBuffer,
        codebookBoundariesBuffer: this.codebookBoundariesBuffer,
        bitWidth: this.bitWidth,
      },
    };
  }

  getGPUBuffers(layerIdx) {
    const layer = this.layers[layerIdx];
    if (!layer) return null;

    const base = {
      layout: 'contiguous_quantized',
      seqLen: this.currentSeqLen,
      quantMode: this.quantMode,
      prodMode: this.prodMode,
      packedStride: this.prodMode ? this.msePackedStride : this.packedStride,
      scalesKGPU: layer.scalesK,
      scalesVGPU: layer.scalesV,
      rotationMatrixBuffer: this.rotationMatrixBuffer,
      codebookCentroidsBuffer: this.codebookCentroidsBuffer,
    };

    if (this.prodMode) {
      return {
        ...base,
        keysPackedGPU: layer.keysPackedMSE,
        valuesPackedGPU: layer.valuesPackedMSE,
        residualKGPU: layer.residualK,
        residualVGPU: layer.residualV,
        residualNormsKGPU: layer.residualNormsK,
        residualNormsVGPU: layer.residualNormsV,
        residualPackedStride: this.residualPackedStride,
        qjlMatrixBuffer: this.qjlMatrixBuffer,
      };
    }

    return {
      ...base,
      keysPackedGPU: layer.keysPacked,
      valuesPackedGPU: layer.valuesPacked,
    };
  }

  hasGPUCache() {
    return Array.isArray(this.layers);
  }

  clear() {
    for (const layer of this.layers) {
      layer.seqLen = 0;
    }
    this.currentSeqLen = 0;
    this.totalTokensSeen = 0;
  }

  truncate(length) {
    this.currentSeqLen = Math.min(this.currentSeqLen, length);
    for (const layer of this.layers) {
      layer.seqLen = Math.min(layer.seqLen, length);
    }
    this.totalTokensSeen = Math.min(this.totalTokensSeen, this.currentSeqLen);
  }

  getMemoryStats() {
    return {
      theoretical: this.memoryUsage,
      allocated: this.memoryUsage,
      used: this.memoryUsage,
      efficiency: 1.0,
      seqLen: this.currentSeqLen,
      maxSeqLen: this.maxSeqLen,
      layout: this.layout,
      kvDtype: this.kvDtype,
      counters: null,
    };
  }

  setGPUContext() { /* no-op for quantized cache */ }

  destroy() {
    for (const layer of this.layers) {
      if (this.prodMode) {
        layer.keysPackedMSE?.destroy();
        layer.valuesPackedMSE?.destroy();
        layer.residualK?.destroy();
        layer.residualV?.destroy();
        layer.residualNormsK?.destroy();
        layer.residualNormsV?.destroy();
      } else {
        layer.keysPacked?.destroy();
        layer.valuesPacked?.destroy();
      }
      layer.scalesK?.destroy();
      layer.scalesV?.destroy();
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
}
