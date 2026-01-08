/**
 * Sliding Window KV Cache - For long sequences with limited memory
 *
 * Only keeps the most recent N tokens in the cache, using ring-buffer
 * semantics for GPU storage.
 *
 * @module inference/kv-cache/sliding-window
 */

import { getDevice } from '../../gpu/device.js';
import { getRuntimeConfig } from '../../config/runtime.js';
import { KVCache } from './base.js';

// ============================================================================
// SlidingWindowKVCache Class
// ============================================================================

/**
 * Sliding Window KV Cache for long sequences
 * Only keeps the most recent N tokens
 */
export class SlidingWindowKVCache extends KVCache {
  /**
   * @param {import('./types.js').KVCacheConfig & { windowSize?: number }} config - Configuration with windowSize
   */
  constructor(config) {
    super(config);
    /** @readonly @type {number} */
    this.windowSize = config.windowSize || getRuntimeConfig().kvcache.windowSize;
    /** @type {number} */
    this.totalTokensSeen = 0;
  }

  /**
   * Update with sliding window logic
   * @param {number} layerIdx
   * @param {Float32Array | GPUBuffer} keys
   * @param {Float32Array | GPUBuffer} values
   * @param {number} [startPos]
   */
  update(
    layerIdx,
    keys,
    values,
    startPos = this.currentSeqLen
  ) {
    if (keys instanceof GPUBuffer || values instanceof GPUBuffer) {
      throw new Error('Use updateFromGPU for GPU buffer inputs');
    }

    const numNewTokens = keys.length / this.kvSize;
    this.totalTokensSeen += numNewTokens;

    // Check if we need to slide the window
    if (this.currentSeqLen + numNewTokens > this.windowSize) {
      this._slideWindow(numNewTokens);
    }

    // Add new tokens
    super.update(layerIdx, keys, values, this.currentSeqLen);
  }

  /**
   * GPU-native update with ring-buffer semantics.
   * Keeps the last `windowSize` tokens in GPU memory while allowing
   * unbounded absolute positions for RoPE.
   * @param {number} layerIdx
   * @param {GPUBuffer} keysBuffer
   * @param {GPUBuffer} valuesBuffer
   * @param {number} startPos
   * @param {number} numTokens
   */
  updateFromGPU(
    layerIdx,
    keysBuffer,
    valuesBuffer,
    startPos,
    numTokens
  ) {
    const layer = /** @type {import('./types.js').ContiguousLayerCache} */ (this.layers[layerIdx]);
    const device = getDevice();

    if (!device || !layer.keysGPU) {
      throw new Error('GPU cache not initialized');
    }

    const windowSize = this.windowSize;
    const bytesPerToken = this.kvSize * this.bytesPerElem;
    const writePos = startPos % windowSize;

    const firstChunkTokens = Math.min(numTokens, windowSize - writePos);
    const firstChunkBytes = firstChunkTokens * bytesPerToken;
    const secondChunkTokens = numTokens - firstChunkTokens;
    const secondChunkBytes = secondChunkTokens * bytesPerToken;

    const encoder = device.createCommandEncoder({ label: 'kv_cache_update_sliding' });

    const destByteOffset1 = writePos * bytesPerToken;
    encoder.copyBufferToBuffer(keysBuffer, 0, layer.keysGPU, destByteOffset1, firstChunkBytes);
    encoder.copyBufferToBuffer(valuesBuffer, 0, layer.valuesGPU, destByteOffset1, firstChunkBytes);

    if (secondChunkTokens > 0) {
      const srcByteOffset2 = firstChunkBytes;
      encoder.copyBufferToBuffer(keysBuffer, srcByteOffset2, layer.keysGPU, 0, secondChunkBytes);
      encoder.copyBufferToBuffer(valuesBuffer, srcByteOffset2, layer.valuesGPU, 0, secondChunkBytes);
    }

    device.queue.submit([encoder.finish()]);

    const seen = Math.max(this.totalTokensSeen, startPos + numTokens);
    this.totalTokensSeen = seen;
    const storedLen = Math.min(windowSize, seen);

    layer.seqLen = Math.max(layer.seqLen || 0, storedLen);
    if (layerIdx === this.numLayers - 1) {
      this.currentSeqLen = storedLen;
    }
  }

  /**
   * Record KV cache update with ring-buffer semantics to an external encoder.
   * Does NOT submit - caller is responsible for submitting the encoder.
   * @param {GPUCommandEncoder} encoder
   * @param {number} layerIdx
   * @param {GPUBuffer} keysBuffer
   * @param {GPUBuffer} valuesBuffer
   * @param {number} startPos
   * @param {number} numTokens
   */
  recordUpdateFromGPU(
    encoder,
    layerIdx,
    keysBuffer,
    valuesBuffer,
    startPos,
    numTokens
  ) {
    const layer = /** @type {import('./types.js').ContiguousLayerCache} */ (this.layers[layerIdx]);

    if (!layer.keysGPU) {
      throw new Error('GPU cache not initialized');
    }

    const windowSize = this.windowSize;
    const bytesPerToken = this.kvSize * this.bytesPerElem;
    const writePos = startPos % windowSize;

    const firstChunkTokens = Math.min(numTokens, windowSize - writePos);
    const firstChunkBytes = firstChunkTokens * bytesPerToken;
    const secondChunkTokens = numTokens - firstChunkTokens;
    const secondChunkBytes = secondChunkTokens * bytesPerToken;

    const destByteOffset1 = writePos * bytesPerToken;
    encoder.copyBufferToBuffer(keysBuffer, 0, layer.keysGPU, destByteOffset1, firstChunkBytes);
    encoder.copyBufferToBuffer(valuesBuffer, 0, layer.valuesGPU, destByteOffset1, firstChunkBytes);

    if (secondChunkTokens > 0) {
      const srcByteOffset2 = firstChunkBytes;
      encoder.copyBufferToBuffer(keysBuffer, srcByteOffset2, layer.keysGPU, 0, secondChunkBytes);
      encoder.copyBufferToBuffer(valuesBuffer, srcByteOffset2, layer.valuesGPU, 0, secondChunkBytes);
    }

    // Update metadata (copies happen when encoder is submitted)
    const seen = Math.max(this.totalTokensSeen, startPos + numTokens);
    this.totalTokensSeen = seen;
    const storedLen = Math.min(windowSize, seen);

    layer.seqLen = Math.max(layer.seqLen || 0, storedLen);
    if (layerIdx === this.numLayers - 1) {
      this.currentSeqLen = storedLen;
    }
  }

  /**
   * Slide the window to make room for new tokens
   * @private
   * @param {number} numNewTokens
   */
  _slideWindow(numNewTokens) {
    const shiftAmount = Math.min(
      this.currentSeqLen,
      this.currentSeqLen + numNewTokens - this.windowSize
    );

    if (shiftAmount <= 0) return;

    // Shift cache contents for each layer
    for (let l = 0; l < this.numLayers; l++) {
      const layer = /** @type {import('./types.js').ContiguousLayerCache} */ (this.layers[l]);
      const keepFrom = shiftAmount * this.kvSize;
      const keepLength = (layer.seqLen - shiftAmount) * this.kvSize;

      // Shift keys and values
      layer.keys.copyWithin(0, keepFrom, keepFrom + keepLength);
      layer.values.copyWithin(0, keepFrom, keepFrom + keepLength);
      layer.seqLen -= shiftAmount;
    }

    this.currentSeqLen -= shiftAmount;
  }

  /**
   * @returns {import('./types.js').MemoryStats & { windowSize: number; totalTokensSeen: number }}
   */
  getMemoryStats() {
    const stats = super.getMemoryStats();
    return {
      ...stats,
      windowSize: this.windowSize,
      totalTokensSeen: this.totalTokensSeen
    };
  }
}
