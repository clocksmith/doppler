import { getDevice } from '../../gpu/device.js';

function normalizeLayerType(layerType) {
  return typeof layerType === 'string' ? layerType.trim().toLowerCase() : '';
}

function isSlidingLayerType(layerType) {
  const normalized = normalizeLayerType(layerType);
  return normalized === 'sliding_attention'
    || normalized === 'local_attention'
    || normalized === 'local'
    || normalized === 'sliding';
}

function buildLayerSpecs(config, bytesPerElem) {
  const layerTypes = Array.isArray(config.layerTypes) ? config.layerTypes : null;
  if (!layerTypes || layerTypes.length !== config.numLayers) {
    throw new Error(
      'MixedGeometryKVCache requires explicit layerTypes for every layer so per-layer KV geometry stays manifest-owned.'
    );
  }

  const baseHeadDim = Number(config.headDim);
  const globalHeadDim = Number.isFinite(config.globalHeadDim) && config.globalHeadDim > 0
    ? Math.trunc(config.globalHeadDim)
    : baseHeadDim;
  const numHeads = Number(config.numHeads);
  const maxSeqLen = Number(config.maxSeqLen);
  const slidingWindow = Number.isFinite(config.slidingWindow) && config.slidingWindow > 0
    ? Math.min(Math.trunc(config.slidingWindow), maxSeqLen)
    : null;

  if (!Number.isFinite(baseHeadDim) || baseHeadDim <= 0) {
    throw new Error('MixedGeometryKVCache requires a positive headDim.');
  }
  if (!Number.isFinite(numHeads) || numHeads <= 0) {
    throw new Error('MixedGeometryKVCache requires a positive numHeads.');
  }
  if (!Number.isFinite(maxSeqLen) || maxSeqLen <= 0) {
    throw new Error('MixedGeometryKVCache requires a positive maxSeqLen.');
  }

  return layerTypes.map((layerType, layerIdx) => {
    const isSliding = isSlidingLayerType(layerType);
    const headDim = isSliding ? baseHeadDim : globalHeadDim;
    const capacityTokens = isSliding && slidingWindow != null
      ? slidingWindow
      : maxSeqLen;
    const kvSize = numHeads * headDim;
    const bytesPerToken = kvSize * bytesPerElem;
    return {
      layerIdx,
      layerType,
      layout: isSliding && slidingWindow != null ? 'ring' : 'contiguous',
      headDim,
      numHeads,
      kvSize,
      capacityTokens,
      bytesPerToken,
      capacityBytes: capacityTokens * bytesPerToken,
    };
  });
}

function createLayerBuffers(device, spec, kvDtype) {
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const keysGPU = device.createBuffer({
    label: `mixed_kv_keys_layer_${spec.layerIdx}`,
    size: spec.capacityBytes,
    usage,
  });
  const valuesGPU = device.createBuffer({
    label: `mixed_kv_values_layer_${spec.layerIdx}`,
    size: spec.capacityBytes,
    usage,
  });
  return {
    keysGPU,
    valuesGPU,
    seqLen: 0,
    kvDtype,
  };
}

function destroyLayerBuffers(layer) {
  try {
    layer?.keysGPU?.destroy?.();
  } catch {
    // Ignore already-destroyed buffers during rollback.
  }
  try {
    layer?.valuesGPU?.destroy?.();
  } catch {
    // Ignore already-destroyed buffers during rollback.
  }
}

export class MixedGeometryKVCache {
  constructor(config) {
    if (!config || typeof config !== 'object') {
      throw new Error('MixedGeometryKVCache requires a config object.');
    }
    if (config.useGPU !== true) {
      throw new Error(
        'MixedGeometryKVCache requires GPU execution. ' +
        'Mixed-head-dim incremental decode is not supported on CPU.'
      );
    }
    if (!Number.isFinite(config.pageSize) || config.pageSize <= 0) {
      throw new Error('MixedGeometryKVCache requires a positive pageSize.');
    }
    if (config.kvDtype !== 'f16' && config.kvDtype !== 'f32') {
      throw new Error('MixedGeometryKVCache requires kvDtype to be "f16" or "f32".');
    }

    const device = getDevice();
    if (!device) {
      throw new Error('MixedGeometryKVCache requires an initialized GPU device.');
    }

    this.numLayers = Math.trunc(config.numLayers);
    this.numHeads = Math.trunc(config.numHeads);
    this.headDim = Math.max(
      Math.trunc(config.headDim),
      Number.isFinite(config.globalHeadDim) ? Math.trunc(config.globalHeadDim) : Math.trunc(config.headDim)
    );
    this.maxSeqLen = Math.trunc(config.maxSeqLen);
    this.useGPU = true;
    this.layout = 'contiguous';
    this.pageSize = Math.trunc(config.pageSize);
    this.kvDtype = config.kvDtype;
    this.bytesPerElem = this.kvDtype === 'f16' ? 2 : 4;
    this.kvSize = this.numHeads * this.headDim;
    this.windowSize = Number.isFinite(config.slidingWindow) && config.slidingWindow > 0
      ? Math.min(Math.trunc(config.slidingWindow), this.maxSeqLen)
      : undefined;
    this.currentSeqLen = 0;
    this.totalTokensSeen = 0;
    this.memoryUsage = 0;
    this.gpuContext = { device };
    this._config = {
      ...config,
      numLayers: this.numLayers,
      numHeads: this.numHeads,
      headDim: Math.trunc(config.headDim),
      globalHeadDim: Number.isFinite(config.globalHeadDim) ? Math.trunc(config.globalHeadDim) : null,
      maxSeqLen: this.maxSeqLen,
      pageSize: this.pageSize,
    };
    this.layerSpecs = buildLayerSpecs(this._config, this.bytesPerElem);
    this.layers = new Array(this.numLayers);

    try {
      for (let layerIdx = 0; layerIdx < this.numLayers; layerIdx++) {
        const spec = this.layerSpecs[layerIdx];
        this.layers[layerIdx] = createLayerBuffers(device, spec, this.kvDtype);
        this.memoryUsage += spec.capacityBytes * 2;
      }
    } catch (error) {
      for (const layer of this.layers) {
        destroyLayerBuffers(layer);
      }
      throw error;
    }
  }

  _assertLayerIndex(layerIdx) {
    if (!Number.isInteger(layerIdx) || layerIdx < 0 || layerIdx >= this.numLayers) {
      throw new Error(`MixedGeometryKVCache layer index out of range: ${layerIdx}`);
    }
  }

  _assertStartPos(startPos) {
    if (!Number.isInteger(startPos) || startPos < 0) {
      throw new Error('MixedGeometryKVCache startPos must be a non-negative integer.');
    }
  }

  _assertTokenCount(numTokens, label) {
    if (!Number.isInteger(numTokens) || numTokens < 0) {
      throw new Error(`${label} requires a non-negative integer token count.`);
    }
  }

  _recordContiguousCopy(encoder, layer, spec, keysBuffer, valuesBuffer, startPos, numTokens) {
    const endPos = startPos + numTokens;
    if (endPos > spec.capacityTokens) {
      throw new Error(
        `MixedGeometryKVCache overflow at layer ${spec.layerIdx}: ${endPos} > ${spec.capacityTokens}.`
      );
    }
    const byteOffset = startPos * spec.bytesPerToken;
    const byteSize = numTokens * spec.bytesPerToken;
    if (byteSize > keysBuffer.size || byteSize > valuesBuffer.size) {
      throw new Error('MixedGeometryKVCache contiguous copy buffer is smaller than the requested write.');
    }
    encoder.copyBufferToBuffer(keysBuffer, 0, layer.keysGPU, byteOffset, byteSize);
    encoder.copyBufferToBuffer(valuesBuffer, 0, layer.valuesGPU, byteOffset, byteSize);
  }

  _recordRingCopy(encoder, layer, spec, keysBuffer, valuesBuffer, startPos, numTokens) {
    const windowSize = spec.capacityTokens;
    const bytesPerToken = spec.bytesPerToken;
    const fullStart = startPos;
    const fullTokens = numTokens;
    let srcByteOffset = 0;

    if (numTokens > windowSize) {
      const dropTokens = numTokens - windowSize;
      startPos += dropTokens;
      numTokens = windowSize;
      srcByteOffset = dropTokens * bytesPerToken;
    }

    const bytesNeeded = srcByteOffset + (numTokens * bytesPerToken);
    if (bytesNeeded > keysBuffer.size || bytesNeeded > valuesBuffer.size) {
      throw new Error('MixedGeometryKVCache ring copy buffer is smaller than the requested write.');
    }

    const writePos = startPos % windowSize;
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

    const seen = Math.max(this.totalTokensSeen, fullStart + fullTokens);
    layer.seqLen = Math.min(windowSize, Math.max(layer.seqLen, seen));
  }

  _updateMetadata(layer, spec, startPos, numTokens) {
    if (spec.layout !== 'ring') {
      layer.seqLen = Math.max(layer.seqLen, Math.min(startPos + numTokens, spec.capacityTokens));
    }
    this.totalTokensSeen = Math.max(this.totalTokensSeen, startPos + numTokens);
    this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numTokens);
  }

  update() {
    throw new Error(
      'MixedGeometryKVCache requires GPU-buffer updateFromGPU()/recordUpdateFromGPU() writes. ' +
      'CPU shadow copies are intentionally unsupported.'
    );
  }

  updateFromGPU(layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
    this._assertLayerIndex(layerIdx);
    this._assertStartPos(startPos);
    this._assertTokenCount(numTokens, 'MixedGeometryKVCache updateFromGPU');
    if (numTokens === 0) {
      return;
    }

    const device = getDevice();
    if (!device) {
      throw new Error('MixedGeometryKVCache updateFromGPU requires an initialized GPU device.');
    }

    const spec = this.layerSpecs[layerIdx];
    const layer = this.layers[layerIdx];
    const encoder = device.createCommandEncoder({ label: `mixed_kv_update_${layerIdx}` });
    if (spec.layout === 'ring') {
      this._recordRingCopy(encoder, layer, spec, keysBuffer, valuesBuffer, startPos, numTokens);
    } else {
      this._recordContiguousCopy(encoder, layer, spec, keysBuffer, valuesBuffer, startPos, numTokens);
    }
    device.queue.submit([encoder.finish()]);
    this._updateMetadata(layer, spec, startPos, numTokens);
  }

  recordUpdateFromGPU(recorder, layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
    this._assertLayerIndex(layerIdx);
    this._assertStartPos(startPos);
    this._assertTokenCount(numTokens, 'MixedGeometryKVCache recordUpdateFromGPU');
    if (numTokens === 0) {
      return;
    }

    const spec = this.layerSpecs[layerIdx];
    const layer = this.layers[layerIdx];
    const encoder = recorder.getEncoder();
    if (spec.layout === 'ring') {
      this._recordRingCopy(encoder, layer, spec, keysBuffer, valuesBuffer, startPos, numTokens);
    } else {
      this._recordContiguousCopy(encoder, layer, spec, keysBuffer, valuesBuffer, startPos, numTokens);
    }
    this._updateMetadata(layer, spec, startPos, numTokens);
  }

  get() {
    throw new Error('MixedGeometryKVCache CPU get() is not implemented.');
  }

  getKeyCache(layerIdx) {
    this._assertLayerIndex(layerIdx);
    return this.layers[layerIdx]?.keysGPU ?? null;
  }

  getValueCache(layerIdx) {
    this._assertLayerIndex(layerIdx);
    return this.layers[layerIdx]?.valuesGPU ?? null;
  }

  getGPUBuffers(layerIdx) {
    this._assertLayerIndex(layerIdx);
    const layer = this.layers[layerIdx];
    if (!layer?.keysGPU || !layer?.valuesGPU) {
      return null;
    }
    const spec = this.layerSpecs[layerIdx];
    return {
      layout: spec.layout,
      keysGPU: layer.keysGPU,
      valuesGPU: layer.valuesGPU,
      seqLen: layer.seqLen,
    };
  }

  hasGPUCache() {
    return this.layers[0]?.keysGPU != null && this.layers[0]?.valuesGPU != null;
  }

  clear() {
    this.currentSeqLen = 0;
    this.totalTokensSeen = 0;
    for (let layerIdx = 0; layerIdx < this.numLayers; layerIdx++) {
      this.layers[layerIdx].seqLen = 0;
    }
  }

  clone() {
    const cloned = new MixedGeometryKVCache(this._config);
    cloned.currentSeqLen = this.currentSeqLen;
    cloned.totalTokensSeen = this.totalTokensSeen;

    const device = getDevice();
    if (!device) {
      throw new Error('MixedGeometryKVCache clone requires an initialized GPU device.');
    }

    const encoder = device.createCommandEncoder({ label: 'mixed_kv_clone' });
    let hasCopies = false;
    for (let layerIdx = 0; layerIdx < this.numLayers; layerIdx++) {
      const spec = this.layerSpecs[layerIdx];
      const srcLayer = this.layers[layerIdx];
      const dstLayer = cloned.layers[layerIdx];
      dstLayer.seqLen = srcLayer.seqLen;
      const copyBytes = spec.layout === 'ring'
        ? spec.capacityBytes
        : srcLayer.seqLen * spec.bytesPerToken;
      if (copyBytes <= 0) {
        continue;
      }
      encoder.copyBufferToBuffer(srcLayer.keysGPU, 0, dstLayer.keysGPU, 0, copyBytes);
      encoder.copyBufferToBuffer(srcLayer.valuesGPU, 0, dstLayer.valuesGPU, 0, copyBytes);
      hasCopies = true;
    }

    if (hasCopies) {
      device.queue.submit([encoder.finish()]);
    }

    return cloned;
  }

  truncate(length) {
    if (!Number.isInteger(length) || length < 0) {
      throw new Error('MixedGeometryKVCache truncate length must be a non-negative integer.');
    }

    this.currentSeqLen = Math.min(length, this.maxSeqLen);
    this.totalTokensSeen = Math.min(this.totalTokensSeen, this.currentSeqLen);
    for (let layerIdx = 0; layerIdx < this.numLayers; layerIdx++) {
      const spec = this.layerSpecs[layerIdx];
      this.layers[layerIdx].seqLen = Math.min(this.currentSeqLen, spec.capacityTokens);
    }
  }

  getMemoryStats() {
    let used = 0;
    for (let layerIdx = 0; layerIdx < this.numLayers; layerIdx++) {
      const spec = this.layerSpecs[layerIdx];
      const layer = this.layers[layerIdx];
      used += Math.min(layer.seqLen, spec.capacityTokens) * spec.bytesPerToken * 2;
    }
    return {
      theoretical: this.memoryUsage,
      allocated: this.memoryUsage,
      used,
      efficiency: this.memoryUsage > 0 ? used / this.memoryUsage : 1,
      seqLen: this.currentSeqLen,
      maxSeqLen: this.maxSeqLen,
      layout: this.layout,
      kvDtype: this.kvDtype,
      counters: null,
    };
  }

  setGPUContext(gpuContext) {
    this.gpuContext = gpuContext ?? null;
  }

  async syncToCPU() {
    throw new Error(
      'MixedGeometryKVCache does not support syncToCPU(). ' +
      'Use layer readback helpers directly when debugging mixed-geometry KV state.'
    );
  }

  destroy() {
    for (const layer of this.layers) {
      destroyLayerBuffers(layer);
    }
  }
}
