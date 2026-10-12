const INDEX_ARGUMENT = Object.freeze({
  update: 0, updateFromGPU: 0, recordUpdateFromGPU: 1,
  recordUpdateF32ToF16FromGPU: 1, recordF16UpdateAlreadyWrittenFromGPU: 0,
  get: 0, getKeyCache: 0, getValueCache: 0, getGPUBuffers: 0,
});

/** @type {import('./layer-range.js').resolveKVCacheLayerCount} */
export function resolveKVCacheLayerCount(numLayers, layerRange) {
  if (layerRange === null) return numLayers;
  if (!Array.isArray(layerRange) || layerRange.length !== 2
    || !layerRange.every(Number.isSafeInteger) || layerRange[0] < 0
    || layerRange[1] < layerRange[0] || layerRange[1] >= numLayers) {
    throw new Error('KV cache layerRange must be an inclusive range inside the model.');
  }
  return layerRange[1] - layerRange[0] + 1;
}

/** @type {import('./layer-range.js').scopeKVCacheLayerRange} */
export function scopeKVCacheLayerRange(cache, firstLayer) {
  if (!Number.isSafeInteger(firstLayer) || firstLayer < 0) {
    throw new Error('KV cache firstLayer must be a non-negative integer.');
  }
  const lastLayer = firstLayer + cache.numLayers - 1;
  if (!Number.isSafeInteger(lastLayer) || lastLayer < firstLayer) throw new Error('Resident KV range is invalid.');
  const indices = Array.from({ length: cache.numLayers }, (_, index) => firstLayer + index);
  return scopeIndices(cache, indices, layer =>
    `Layer ${layer} is outside resident KV range [${firstLayer}, ${lastLayer}].`);
}

/** @type {import('./layer-range.js').scopeKVCacheLayerIndices} */
export function scopeKVCacheLayerIndices(cache, layerIndices) {
  if (!Array.isArray(layerIndices) || layerIndices.length !== cache.numLayers
    || layerIndices.some((layer, index) => !Number.isSafeInteger(layer) || layer < 0
      || (index > 0 && layer <= layerIndices[index - 1]))) {
    throw new Error('KV cache layer indices must match allocations in ascending model order.');
  }
  return scopeIndices(cache, [...layerIndices], layer => `Layer ${layer} has no allocated KV cache.`);
}

/**
 * @template {import('./base.js').KVCache} T
 * @param {T} cache
 * @param {readonly number[]} indices
 * @param {(layer: unknown) => string} describeMissing
 * @returns {T}
 */
function scopeIndices(cache, indices, describeMissing) {
  const modelIndices = Object.freeze([...indices]);
  const localIndices = new Map(indices.map((layer, index) => [layer, index]));
  const methods = new Map();
  // Bind to the allocation owner so internal method calls keep local indices.
  // The view preserves cache subclass identity (including sliding-window masks).
  return new Proxy(cache, {
    get(target, key) {
      if (key === 'layerIndices') return modelIndices;
      const value = Reflect.get(target, key, target);
      if (typeof value !== 'function' || key === 'constructor') return value;
      if (!methods.has(key)) {
        const position = Object.hasOwn(INDEX_ARGUMENT, key) ? Reflect.get(INDEX_ARGUMENT, key) : null;
        methods.set(key, (/** @type {unknown[]} */ ...args) => {
          if (position !== null) {
            const layer = args[position];
            if (typeof layer !== 'number' || !localIndices.has(layer)) throw new RangeError(describeMissing(layer));
            args[position] = localIndices.get(layer);
          }
          const result = value.apply(target, args);
          return key === 'clone' ? scopeIndices(result, indices, describeMissing) : result;
        });
      }
      return methods.get(key);
    },
    set() { throw new Error('Resident KV cache properties are owned by the cache.'); },
    defineProperty() { throw new Error('Resident KV cache properties are owned by the cache.'); },
    deleteProperty() { throw new Error('Resident KV cache properties are owned by the cache.'); },
  });
}
