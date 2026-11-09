import { getDevice } from '../../gpu/device.js';
import { recordKVCacheWriteF32ToF16 } from '../../gpu/kernels/kv-cache-write.js';
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

// Shared mechanics for GPU-side writes into a base cache: span validation and
// counters, the two copy commands, and the bookkeeping every write advances.
// Immediate and recorded paths differ only in who owns the encoder and when it
// is submitted; metadata always advances after the commands are encoded.
function beginGpuWrite(cache, name, layerIdx, startPos, numTokens, counter, needDevice) {
  cache._assertLayerIndex(layerIdx);
  cache._assertStartPos(startPos);
  if (!Number.isInteger(numTokens) || numTokens < 0) {
    throw new Error(`KVCache ${name} requires a non-negative integer token count.`);
  }
  if (numTokens === 0) {
    return null;
  }
  cache.counters[counter] += 1;
  cache.counters.tokensWritten += numTokens;
  const layer = cache.layers[layerIdx];
  const device = needDevice ? getDevice() : null;
  if ((needDevice && !device) || !layer.keysGPU) {
    throw new Error('GPU cache not initialized');
  }
  if (startPos + numTokens > cache.maxSeqLen) {
    throw new Error(`Cache overflow: ${startPos + numTokens} > ${cache.maxSeqLen}`);
  }
  return { layer, device };
}

function contiguousCopySpan(cache, name, keysBuffer, valuesBuffer, startPos, numTokens) {
  const byteOffset = startPos * cache.kvSize * cache.bytesPerElem;
  const byteSize = numTokens * cache.kvSize * cache.bytesPerElem;
  if (byteSize > keysBuffer.size || byteSize > valuesBuffer.size) {
    throw new Error(`KVCache ${name} buffer size is smaller than requested write.`);
  }
  return { byteOffset, byteSize };
}

function encodeContiguousCopy(encoder, layer, keysBuffer, valuesBuffer, span) {
  encoder.copyBufferToBuffer(keysBuffer, 0, layer.keysGPU, span.byteOffset, span.byteSize);
  encoder.copyBufferToBuffer(valuesBuffer, 0, layer.valuesGPU, span.byteOffset, span.byteSize);
}

function noteWrittenSpan(cache, layer, startPos, numTokens) {
  if (cache.layout === 'paged') {
    const neededPages = Math.ceil((startPos + numTokens) / cache.pageSize);
    layer.allocatedPages = Number.isFinite(layer.allocatedPages)
      ? Math.max(layer.allocatedPages, neededPages)
      : neededPages;
  }
  layer.seqLen = Math.max(layer.seqLen, startPos + numTokens);
  cache.totalTokensSeen = Math.max(cache.totalTokensSeen, startPos + numTokens);
  cache.currentSeqLen = Math.max(cache.currentSeqLen, startPos + numTokens);
}

export function updateFromGPU(layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
  const write = beginGpuWrite(this, 'updateFromGPU', layerIdx, startPos, numTokens, 'gpuUpdateCalls', true);
  if (!write) return;
  const span = contiguousCopySpan(this, 'updateFromGPU', keysBuffer, valuesBuffer, startPos, numTokens);
  const encoder = write.device.createCommandEncoder({ label: 'kv_cache_update' });
  encodeContiguousCopy(encoder, write.layer, keysBuffer, valuesBuffer, span);
  write.device.queue.submit([encoder.finish()]);
  noteWrittenSpan(this, write.layer, startPos, numTokens);
}

export function recordUpdateFromGPU(recorder, layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
  const write = beginGpuWrite(this, 'recordUpdateFromGPU', layerIdx, startPos, numTokens, 'recordedGpuUpdateCalls', false);
  if (!write) return;
  const span = contiguousCopySpan(this, 'recordUpdateFromGPU', keysBuffer, valuesBuffer, startPos, numTokens);
  // The copies join the recorder's encoder and land when it is submitted.
  encodeContiguousCopy(recorder.getEncoder(), write.layer, keysBuffer, valuesBuffer, span);
  noteWrittenSpan(this, write.layer, startPos, numTokens);
}

export async function recordUpdateF32ToF16FromGPU(recorder, layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
  this._assertLayerIndex(layerIdx);
  this._assertStartPos(startPos);
  if (this.kvDtype !== 'f16') {
    throw new Error('KVCache recordUpdateF32ToF16FromGPU requires an f16 KV cache.');
  }
  const write = beginGpuWrite(this, 'recordUpdateF32ToF16FromGPU', layerIdx, startPos, numTokens, 'recordedGpuUpdateCalls', false);
  if (!write) return;
  await recordKVCacheWriteF32ToF16(recorder, keysBuffer, valuesBuffer, write.layer.keysGPU, write.layer.valuesGPU, {
    srcOffset: 0,
    dstOffset: startPos * this.kvSize,
    elementCount: numTokens * this.kvSize,
  });
  noteWrittenSpan(this, write.layer, startPos, numTokens);
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
    noteWrittenSpan(this, layer, startPos, numTokens);
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
        encodeContiguousCopy(encoder, layer, keys, values, { byteOffset, byteSize });
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

// Sequence bookkeeping for tokens whose rows a replayed whole-token decode program
// already wrote on the GPU: the same fields update() advances, without data movement.
// Contiguous base caches only: a subclass that keeps other bookkeeping and does not
// define markAppended itself fails closed, as does a paged layout.
export function markAppended(layerIdx, startPos, numTokens) {
    if (!Object.hasOwn(Object.getPrototypeOf(this), 'markAppended') || this.layout === 'paged') {
      throw new Error('KVCache.markAppended supports contiguous base caches only.');
    }
    const layer = this.layers[layerIdx];
    if (!layer || !Number.isSafeInteger(startPos) || startPos < 0 || !Number.isSafeInteger(numTokens) || numTokens < 1) {
      throw new Error('KVCache.markAppended requires an allocated layer and a positive token span.');
    }
    const end = startPos + numTokens;
    if (end > this.maxSeqLen) {
      throw new Error(`Cache overflow: ${end} > ${this.maxSeqLen}`);
    }
    layer.seqLen = Math.max(layer.seqLen, end);
    this.totalTokensSeen = Math.max(this.totalTokensSeen, end);
    this.currentSeqLen = Math.max(this.currentSeqLen, end);
  }
