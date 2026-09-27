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
  const methods = new Map();
  // Bind to the allocation owner so internal method calls keep local indices.
  // The view preserves cache subclass identity (including sliding-window masks).
  return new Proxy(cache, {
    get(target, key) {
      const value = Reflect.get(target, key, target);
      if (typeof value !== 'function' || key === 'constructor') return value;
      if (!methods.has(key)) {
        const position = Object.hasOwn(INDEX_ARGUMENT, key) ? Reflect.get(INDEX_ARGUMENT, key) : null;
        methods.set(key, (/** @type {unknown[]} */ ...args) => {
          if (position !== null) {
            const layer = args[position];
            if (typeof layer !== 'number' || !Number.isSafeInteger(layer) || layer < firstLayer || layer > lastLayer) {
              throw new RangeError(`Layer ${layer} is outside resident KV range [${firstLayer}, ${lastLayer}].`);
            }
            args[position] = layer - firstLayer;
          }
          return value.apply(target, args);
        });
      }
      return methods.get(key);
    },
    set() { throw new Error('Resident KV cache properties are owned by the cache.'); },
    defineProperty() { throw new Error('Resident KV cache properties are owned by the cache.'); },
    deleteProperty() { throw new Error('Resident KV cache properties are owned by the cache.'); },
  });
}
