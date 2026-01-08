

import { getDevice } from '../../gpu/device.js';
import { KVCache } from './base.js';

// ============================================================================
// SlidingWindowKVCache Class
// ============================================================================


export class SlidingWindowKVCache extends KVCache {
  
  constructor(config) {
    super(config);

    if (!Number.isFinite(config.windowSize) || !Number.isInteger(config.windowSize) || config.windowSize <= 0) {
      throw new Error('SlidingWindowKVCache requires a positive windowSize.');
    }
    this.windowSize = config.windowSize;

    this.totalTokensSeen = 0;
  }

  
  clear() {
    super.clear();
    this.totalTokensSeen = 0;
  }

  
  update(
    layerIdx,
    keys,
    values,
    startPos = this.currentSeqLen
  ) {
    if (keys instanceof GPUBuffer || values instanceof GPUBuffer) {
      throw new Error('Use updateFromGPU for GPU buffer inputs');
    }

    const { numNewTokens } = this._resolveTokenCount(keys, values);
    if (!Number.isInteger(numNewTokens) || numNewTokens < 0) {
      throw new Error('SlidingWindowKVCache update requires a non-negative integer token count.');
    }
    if (numNewTokens === 0) {
      return;
    }
    this.totalTokensSeen += numNewTokens;

    // Check if we need to slide the window
    if (this.currentSeqLen + numNewTokens > this.windowSize) {
      this._slideWindow(numNewTokens);
    }

    // Add new tokens
    super.update(layerIdx, keys, values, this.currentSeqLen);
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
      throw new Error('SlidingWindowKVCache updateFromGPU requires a non-negative integer token count.');
    }
    if (numTokens === 0) {
      return;
    }

    const layer =  (this.layers[layerIdx]);
    const device = getDevice();

    if (!device || !layer.keysGPU) {
      throw new Error('GPU cache not initialized');
    }

    const windowSize = this.windowSize;
    const bytesPerToken = this.kvSize * this.bytesPerElem;
    const fullStart = startPos;
    const fullTokens = numTokens;
    let srcByteOffset = 0;

    if (numTokens > windowSize) {
      const dropTokens = numTokens - windowSize;
      startPos += dropTokens;
      numTokens = windowSize;
      srcByteOffset = dropTokens * bytesPerToken;
    }
    const writePos = startPos % windowSize;
    const bytesNeeded = srcByteOffset + (numTokens * bytesPerToken);
    if (bytesNeeded > keysBuffer.size || bytesNeeded > valuesBuffer.size) {
      throw new Error('SlidingWindowKVCache updateFromGPU buffer size is smaller than requested write.');
    }

    const firstChunkTokens = Math.min(numTokens, windowSize - writePos);
    const firstChunkBytes = firstChunkTokens * bytesPerToken;
    const secondChunkTokens = numTokens - firstChunkTokens;
    const secondChunkBytes = secondChunkTokens * bytesPerToken;

    const encoder = device.createCommandEncoder({ label: 'kv_cache_update_sliding' });

    const destByteOffset1 = writePos * bytesPerToken;
    encoder.copyBufferToBuffer(keysBuffer, srcByteOffset, layer.keysGPU, destByteOffset1, firstChunkBytes);
    encoder.copyBufferToBuffer(valuesBuffer, srcByteOffset, layer.valuesGPU, destByteOffset1, firstChunkBytes);

    if (secondChunkTokens > 0) {
      const srcByteOffset2 = srcByteOffset + firstChunkBytes;
      encoder.copyBufferToBuffer(keysBuffer, srcByteOffset2, layer.keysGPU, 0, secondChunkBytes);
      encoder.copyBufferToBuffer(valuesBuffer, srcByteOffset2, layer.valuesGPU, 0, secondChunkBytes);
    }

    device.queue.submit([encoder.finish()]);

    const seen = Math.max(this.totalTokensSeen, fullStart + fullTokens);
    const storedLen = Math.min(windowSize, Math.max(0, seen));
    this.totalTokensSeen = Math.max(this.totalTokensSeen, seen);
    const prevLayerSeqLen = Math.min(Math.max(layer.seqLen || 0, 0), windowSize);

    layer.seqLen = Math.max(prevLayerSeqLen, storedLen);
    if (layerIdx === this.numLayers - 1) {
      this.currentSeqLen = Math.min(this.currentSeqLen, windowSize);
      this.currentSeqLen = Math.max(this.currentSeqLen, storedLen);
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
      throw new Error('SlidingWindowKVCache recordUpdateFromGPU requires a non-negative integer token count.');
    }
    if (numTokens === 0) {
      return;
    }

    const layer =  (this.layers[layerIdx]);
    const encoder = recorder.getEncoder();

    if (!layer.keysGPU) {
      throw new Error('GPU cache not initialized');
    }

    const windowSize = this.windowSize;
    const bytesPerToken = this.kvSize * this.bytesPerElem;
    const fullStart = startPos;
    const fullTokens = numTokens;
    let srcByteOffset = 0;

    if (numTokens > windowSize) {
      const dropTokens = numTokens - windowSize;
      startPos += dropTokens;
      numTokens = windowSize;
      srcByteOffset = dropTokens * bytesPerToken;
    }
    const writePos = startPos % windowSize;
    const bytesNeeded = srcByteOffset + (numTokens * bytesPerToken);
    if (bytesNeeded > keysBuffer.size || bytesNeeded > valuesBuffer.size) {
      throw new Error('SlidingWindowKVCache recordUpdateFromGPU buffer size is smaller than requested write.');
    }

    const firstChunkTokens = Math.min(numTokens, windowSize - writePos);
    const firstChunkBytes = firstChunkTokens * bytesPerToken;
    const secondChunkTokens = numTokens - firstChunkTokens;
    const secondChunkBytes = secondChunkTokens * bytesPerToken;

    const destByteOffset1 = writePos * bytesPerToken;
    encoder.copyBufferToBuffer(keysBuffer, srcByteOffset, layer.keysGPU, destByteOffset1, firstChunkBytes);
    encoder.copyBufferToBuffer(valuesBuffer, srcByteOffset, layer.valuesGPU, destByteOffset1, firstChunkBytes);

    if (secondChunkTokens > 0) {
      const srcByteOffset2 = srcByteOffset + firstChunkBytes;
      encoder.copyBufferToBuffer(keysBuffer, srcByteOffset2, layer.keysGPU, 0, secondChunkBytes);
      encoder.copyBufferToBuffer(valuesBuffer, srcByteOffset2, layer.valuesGPU, 0, secondChunkBytes);
    }

    // Update metadata (copies happen when encoder is submitted)
    const seen = Math.max(this.totalTokensSeen, fullStart + fullTokens);
    const storedLen = Math.min(windowSize, Math.max(0, seen));
    this.totalTokensSeen = Math.max(this.totalTokensSeen, seen);
    const prevLayerSeqLen = Math.min(Math.max(layer.seqLen || 0, 0), windowSize);

    layer.seqLen = Math.max(prevLayerSeqLen, storedLen);
    if (layerIdx === this.numLayers - 1) {
      this.currentSeqLen = Math.min(this.currentSeqLen, windowSize);
      this.currentSeqLen = Math.max(this.currentSeqLen, storedLen);
    }
  }

  
  _slideWindow(numNewTokens) {
    if (!Number.isInteger(numNewTokens) || numNewTokens <= 0) {
      return;
    }

    const safeCurrentSeqLen = Math.min(Math.max(this.currentSeqLen, 0), this.windowSize);
    const shiftAmount = Math.min(
      safeCurrentSeqLen,
      safeCurrentSeqLen + numNewTokens - this.windowSize
    );

    if (shiftAmount <= 0) return;

    // Shift cache contents for each layer
    for (let l = 0; l < this.numLayers; l++) {
      const layer =  (this.layers[l]);
      const keepFrom = shiftAmount * this.kvSize;
      const safeLayerSeqLen = Math.min(Math.max(layer.seqLen || 0, 0), this.windowSize);
      const keepLengthTokens = Math.max(0, safeLayerSeqLen - shiftAmount);
      const keepLength = keepLengthTokens * this.kvSize;
      layer.seqLen = keepLengthTokens;

      // Shift keys and values
      layer.keys.copyWithin(0, keepFrom, keepFrom + keepLength);
      layer.values.copyWithin(0, keepFrom, keepFrom + keepLength);
    }

    this.currentSeqLen = Math.max(0, safeCurrentSeqLen - shiftAmount);
  }

  
  getMemoryStats() {
    const stats = super.getMemoryStats();
    return {
      ...stats,
      windowSize: this.windowSize,
      totalTokensSeen: this.totalTokensSeen
    };
  }
}
