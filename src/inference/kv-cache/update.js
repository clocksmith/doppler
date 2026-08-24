import { getDevice } from '../../gpu/device.js';
import { recordKVCacheWriteF32ToF16 } from '../../gpu/kernel-selector.js';
import {
  isContiguousLayer,
} from './types.js';

export function update(layerIdx, keys, values, startPos = this.currentSeqLen) {
    this._assertLayerIndex(layerIdx);
    this._assertStartPos(startPos);

    const { numNewTokens } = this._resolveTokenCount(keys, values);
    if (!Number.isInteger(numNewTokens) || numNewTokens < 0) {
      throw new Error('KVCache update requires a non-negative integer token count.');
    }
    if (numNewTokens === 0) {
      return;
    }
    this.counters.updateCalls += 1;
    this.counters.tokensWritten += numNewTokens;

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

    this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numNewTokens);
  }

export function updateFromGPU(layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
    this._assertLayerIndex(layerIdx);
    this._assertStartPos(startPos);
    if (!Number.isInteger(numTokens) || numTokens < 0) {
      throw new Error('KVCache updateFromGPU requires a non-negative integer token count.');
    }
    if (numTokens === 0) {
      return;
    }
    this.counters.gpuUpdateCalls += 1;
    this.counters.tokensWritten += numTokens;

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

    this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numTokens);
  }

export function recordUpdateFromGPU(recorder, layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
    this._assertLayerIndex(layerIdx);
    this._assertStartPos(startPos);
    if (!Number.isInteger(numTokens) || numTokens < 0) {
      throw new Error('KVCache recordUpdateFromGPU requires a non-negative integer token count.');
    }
    if (numTokens === 0) {
      return;
    }
    this.counters.recordedGpuUpdateCalls += 1;
    this.counters.tokensWritten += numTokens;

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

    this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numTokens);
  }

export async function recordUpdateF32ToF16FromGPU(recorder, layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
    this._assertLayerIndex(layerIdx);
    this._assertStartPos(startPos);
    if (this.kvDtype !== 'f16') {
      throw new Error('KVCache recordUpdateF32ToF16FromGPU requires an f16 KV cache.');
    }
    if (!Number.isInteger(numTokens) || numTokens < 0) {
      throw new Error('KVCache recordUpdateF32ToF16FromGPU requires a non-negative integer token count.');
    }
    if (numTokens === 0) {
      return;
    }
    this.counters.recordedGpuUpdateCalls += 1;
    this.counters.tokensWritten += numTokens;

    const layer =  (this.layers[layerIdx]);
    if (!layer.keysGPU) {
      throw new Error('GPU cache not initialized');
    }

    if (startPos + numTokens > this.maxSeqLen) {
      throw new Error(
        `Cache overflow: ${startPos + numTokens} > ${this.maxSeqLen}`
      );
    }

    const elementCount = numTokens * this.kvSize;
    await recordKVCacheWriteF32ToF16(
      recorder,
      keysBuffer,
      valuesBuffer,
      layer.keysGPU,
      layer.valuesGPU,
      {
        srcOffset: 0,
        dstOffset: startPos * this.kvSize,
        elementCount,
      }
    );

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

    this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numTokens);
  }

export function recordF16UpdateAlreadyWrittenFromGPU(layerIdx, startPos, numTokens) {
    this._assertLayerIndex(layerIdx);
    this._assertStartPos(startPos);
    if (this.kvDtype !== 'f16') {
      throw new Error('KVCache recordF16UpdateAlreadyWrittenFromGPU requires an f16 KV cache.');
    }
    if (this.layout !== 'contiguous') {
      throw new Error('KVCache recordF16UpdateAlreadyWrittenFromGPU requires contiguous layout.');
    }
    if (!Number.isInteger(numTokens) || numTokens < 0) {
      throw new Error('KVCache recordF16UpdateAlreadyWrittenFromGPU requires a non-negative integer token count.');
    }
    if (numTokens === 0) {
      return;
    }

    const layer =  (this.layers[layerIdx]);
    if (!isContiguousLayer(layer) || !layer.keysGPU || !layer.valuesGPU) {
      throw new Error('GPU cache not initialized');
    }
    if (startPos + numTokens > this.maxSeqLen) {
      throw new Error(
        `Cache overflow: ${startPos + numTokens} > ${this.maxSeqLen}`
      );
    }

    const requiredBytes = (startPos + numTokens) * this.kvSize * this.bytesPerElem;
    if (Number.isFinite(layer.keysGPU.size) && requiredBytes > layer.keysGPU.size) {
      throw new Error('KVCache direct f16 keys write exceeds GPU cache buffer size.');
    }
    if (Number.isFinite(layer.valuesGPU.size) && requiredBytes > layer.valuesGPU.size) {
      throw new Error('KVCache direct f16 values write exceeds GPU cache buffer size.');
    }

    this.counters.recordedGpuUpdateCalls += 1;
    this.counters.tokensWritten += numTokens;
    layer.seqLen = Math.max(layer.seqLen, startPos + numTokens);
    this.totalTokensSeen = Math.max(this.totalTokensSeen, startPos + numTokens);
    this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numTokens);
  }

export function _updateContiguous(layer, keys, values, startPos, numNewTokens) {
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
        throw new Error(
          'KVCache f16 GPU updates require GPU f16 inputs or recordUpdateF32ToF16FromGPU().'
        );
      } else {
        device.queue.writeBuffer(layer.keysGPU, byteOffset,  (keys));
        device.queue.writeBuffer(layer.valuesGPU, byteOffset,  (values));
      }
    }
  }

export function _updatePaged(layerIdx, layer, keys, values, startPos, numNewTokens) {
    const device = getDevice();
    if (layer.keysGPU && layer.valuesGPU && device) {
      const byteOffset = startPos * this.kvSize * this.bytesPerElem;
      if (this.kvDtype === 'f16') {
        throw new Error(
          'Paged KVCache f16 GPU updates require a declared GPU-native write path.'
        );
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
