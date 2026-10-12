// Test-only readback of the allocations returned to attention. No model state
// is substituted. Copies are submitted synchronously before the next step.
export function observePhysicalKVCache({ KVCache, getDevice, layers, allocationLayerIndices = null }) {
  const original = KVCache.prototype.getGPUBuffers;
  const latest = new Map(), identities = new WeakMap();
  let nextId = 1;
  const identity = buffer => {
    if (!identities.has(buffer)) identities.set(buffer, nextId++);
    return identities.get(buffer);
  };
  const reads = [];
  KVCache.prototype.getGPUBuffers = function (layer) {
    const result = original.call(this, layer);
    if (allocationLayerIndices && allocationLayerIndices.length !== this.numLayers) {
      throw new Error('Physical cache allocation mapping does not match actual cache.');
    }
    const modelLayer = allocationLayerIndices ? allocationLayerIndices[layer] : layer;
    if (layers.includes(modelLayer) && result) {
      const row = { layer: modelLayer, allocationLayer: layer, seqLen: result.seqLen, keysId: identity(result.keysGPU), valuesId: identity(result.valuesGPU) };
      reads.push(row);
      latest.set(modelLayer, { cache: this, buffers: result, read: row });
    }
    return result;
  };
  return {
    restore() { KVCache.prototype.getGPUBuffers = original; },
    capture(step, expectedSeqLen) {
      const device = getDevice(), staging = [], rows = [];
      try {
        const encoder = device.createCommandEncoder({ label: 'physical_cache_diagnostic' });
        for (const layer of layers) {
          const observed = latest.get(layer);
          if (!observed) throw new Error(`Physical cache was not read at layer ${layer}`);
          const { cache, buffers, read } = observed;
          if (cache.layout !== 'contiguous' || cache.kvDtype !== 'f16'
            || buffers.seqLen !== expectedSeqLen || cache.layers[read.allocationLayer].seqLen !== expectedSeqLen) {
            throw new Error(`Unexpected physical cache geometry at layer ${layer}`);
          }
          const bytes = expectedSeqLen * cache.numHeads * cache.headDim * 2;
          const row = { ...read, step, layout: cache.layout, dtype: cache.kvDtype,
            shape: [expectedSeqLen, cache.numHeads, cache.headDim], byteOffset: 0, byteLength: bytes,
            reads: reads.filter(value => value.layer === layer), values: {} };
          rows.push(row);
          for (const [name, source] of [['k', buffers.keysGPU], ['v', buffers.valuesGPU]]) {
            if (bytes % 4 || bytes > source.size) throw new Error('Physical cache copy exceeds storage');
            const buffer = device.createBuffer({ label: 'physical_cache_readback', size: bytes,
              usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            staging.push({ buffer, row, name });
            encoder.copyBufferToBuffer(source, 0, buffer, 0, bytes);
          }
        }
        device.queue.submit([encoder.finish()]);
        reads.length = 0;
      } catch (error) {
        for (const { buffer } of staging) buffer.destroy();
        throw error;
      }
      return Promise.allSettled(staging.map(async ({ buffer, row, name }) => {
        try {
          await buffer.mapAsync(GPUMapMode.READ);
          const bytes = new Uint8Array(buffer.getMappedRange());
          let binary = '';
          for (let offset = 0; offset < bytes.length; offset += 32768) {
            binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
          }
          row.values[name] = { encoding: 'base64-f16le', dataBase64: btoa(binary) };
        } finally {
          if (buffer.mapState === 'mapped') buffer.unmap();
          buffer.destroy();
        }
      })).then(results => {
        const failed = results.find(result => result.status === 'rejected');
        if (failed) throw failed.reason;
        return rows;
      });
    }
  };
}
