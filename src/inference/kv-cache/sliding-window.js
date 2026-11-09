

import { getDevice } from '../../gpu/device.js';
import { recordKVCacheWriteF32ToF16 } from '../../gpu/kernels/kv-cache-write.js';
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

  
  // Ring-buffer writes share one plan: which tokens survive when more arrive
  // than the window holds, where the write starts and how it wraps. Immediate
  // and recorded paths differ only in who owns the encoder and submits it;
  // window bookkeeping always follows the encoded commands.
  _beginGpuWrite(name, layerIdx, startPos, numTokens) {
    this._assertLayerIndex(layerIdx);
    this._assertStartPos(startPos);
    if (!Number.isInteger(numTokens) || numTokens < 0) {
      throw new Error(`SlidingWindowKVCache ${name} requires a non-negative integer token count.`);
    }
    return numTokens === 0 ? null : this.layers[layerIdx];
  }

  _planRingWrite(name, keysBuffer, valuesBuffer, startPos, numTokens, unitsPerToken, bytesPerUnit) {
    const windowSize = this.windowSize;
    const fullStart = startPos;
    const fullTokens = numTokens;
    let srcOffset = 0;
    if (numTokens > windowSize) {
      const dropTokens = numTokens - windowSize;
      startPos += dropTokens;
      numTokens = windowSize;
      srcOffset = dropTokens * unitsPerToken;
    }
    const writePos = startPos % windowSize;
    const bytesNeeded = (srcOffset + (numTokens * unitsPerToken)) * bytesPerUnit;
    if (bytesNeeded > keysBuffer.size || bytesNeeded > valuesBuffer.size) {
      throw new Error(`SlidingWindowKVCache ${name} buffer size is smaller than requested write.`);
    }
    const firstChunkTokens = Math.min(numTokens, windowSize - writePos);
    return { fullStart, fullTokens, srcOffset, writePos, firstChunkTokens, secondChunkTokens: numTokens - firstChunkTokens, unitsPerToken };
  }

  _encodeRingCopy(encoder, layer, keysBuffer, valuesBuffer, plan) {
    const firstBytes = plan.firstChunkTokens * plan.unitsPerToken;
    const destOffset = plan.writePos * plan.unitsPerToken;
    encoder.copyBufferToBuffer(keysBuffer, plan.srcOffset, layer.keysGPU, destOffset, firstBytes);
    encoder.copyBufferToBuffer(valuesBuffer, plan.srcOffset, layer.valuesGPU, destOffset, firstBytes);
    if (plan.secondChunkTokens > 0) {
      const secondBytes = plan.secondChunkTokens * plan.unitsPerToken;
      const wrapOffset = plan.srcOffset + firstBytes;
      encoder.copyBufferToBuffer(keysBuffer, wrapOffset, layer.keysGPU, 0, secondBytes);
      encoder.copyBufferToBuffer(valuesBuffer, wrapOffset, layer.valuesGPU, 0, secondBytes);
    }
  }

  _noteRingWrite(layerIdx, layer, plan) {
    const windowSize = this.windowSize;
    const seen = Math.max(this.totalTokensSeen, plan.fullStart + plan.fullTokens);
    const storedLen = Math.min(windowSize, Math.max(0, seen));
    this.totalTokensSeen = Math.max(this.totalTokensSeen, seen);
    const prevLayerSeqLen = Math.min(Math.max(layer.seqLen || 0, 0), windowSize);
    layer.seqLen = Math.max(prevLayerSeqLen, storedLen);
    if (layerIdx === this.numLayers - 1) {
      this.currentSeqLen = Math.min(this.currentSeqLen, windowSize);
      this.currentSeqLen = Math.max(this.currentSeqLen, storedLen);
    }
  }

  updateFromGPU(layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
    const layer = this._beginGpuWrite('updateFromGPU', layerIdx, startPos, numTokens);
    if (!layer) return;
    const device = getDevice();
    if (!device || !layer.keysGPU) {
      throw new Error('GPU cache not initialized');
    }
    const plan = this._planRingWrite('updateFromGPU', keysBuffer, valuesBuffer, startPos, numTokens, this.kvSize * this.bytesPerElem, 1);
    const encoder = device.createCommandEncoder({ label: 'kv_cache_update_sliding' });
    this._encodeRingCopy(encoder, layer, keysBuffer, valuesBuffer, plan);
    device.queue.submit([encoder.finish()]);
    this._noteRingWrite(layerIdx, layer, plan);
  }

  recordUpdateFromGPU(recorder, layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
    const layer = this._beginGpuWrite('recordUpdateFromGPU', layerIdx, startPos, numTokens);
    if (!layer) return;
    const encoder = recorder.getEncoder();
    if (!layer.keysGPU) {
      throw new Error('GPU cache not initialized');
    }
    const plan = this._planRingWrite('recordUpdateFromGPU', keysBuffer, valuesBuffer, startPos, numTokens, this.kvSize * this.bytesPerElem, 1);
    this._encodeRingCopy(encoder, layer, keysBuffer, valuesBuffer, plan);
    this._noteRingWrite(layerIdx, layer, plan);
  }

  async recordUpdateF32ToF16FromGPU(recorder, layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
    this._assertLayerIndex(layerIdx);
    this._assertStartPos(startPos);
    if (this.kvDtype !== 'f16') {
      throw new Error('SlidingWindowKVCache recordUpdateF32ToF16FromGPU requires an f16 KV cache.');
    }
    const layer = this._beginGpuWrite('recordUpdateF32ToF16FromGPU', layerIdx, startPos, numTokens);
    if (!layer) return;
    if (!layer.keysGPU) {
      throw new Error('GPU cache not initialized');
    }
    const plan = this._planRingWrite('recordUpdateF32ToF16FromGPU', keysBuffer, valuesBuffer, startPos, numTokens, this.kvSize, 4);
    await recordKVCacheWriteF32ToF16(recorder, keysBuffer, valuesBuffer, layer.keysGPU, layer.valuesGPU,
      { srcOffset: plan.srcOffset, dstOffset: plan.writePos * this.kvSize, elementCount: plan.firstChunkTokens * this.kvSize });
    if (plan.secondChunkTokens > 0) {
      await recordKVCacheWriteF32ToF16(recorder, keysBuffer, valuesBuffer, layer.keysGPU, layer.valuesGPU,
        { srcOffset: plan.srcOffset + (plan.firstChunkTokens * this.kvSize), dstOffset: 0, elementCount: plan.secondChunkTokens * this.kvSize });
    }
    this._noteRingWrite(layerIdx, layer, plan);
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
