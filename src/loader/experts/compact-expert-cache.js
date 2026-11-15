import { validateQwen35RoutedConfig } from '../../config/schema/loading.schema.js';
import { createWeightBuffer } from '../../gpu/weight-buffer.js';
import { readRoutedExpertRange } from './routed-expert-source.js';

/** @type {typeof import('./compact-expert-cache.js').compactExpertCapacity} */
export function compactExpertCapacity(plan, options) {
  const layers = options.compactLayerCount;
  if (!Number.isSafeInteger(layers) || layers < 1) throw new Error('Compact experts require the declared layer count.');
  const quotas = options.policy.compactLayerBytes;
  if (quotas !== null && (Object.keys(quotas).length !== layers || !Object.hasOwn(quotas, plan.layer)
    || Object.keys(quotas).some(layer => Number(layer) >= plan.numLayers))) {
    throw new Error('compactLayerBytes must cover every assigned routed layer at its original index.');
  }
  const layerBytes = quotas === null ? Math.floor(options.policy.cacheBytes / layers) : quotas[plan.layer];
  const quota = layerBytes - plan.numExperts * 4;
  const limit = Math.min(options.device.limits.maxBufferSize, options.device.limits.maxStorageBufferBindingSize);
  const capacity = Math.min(plan.numExperts, Math.floor(quota / plan.expertBytes),
    ...plan.planes.map(plane => Math.floor(limit / plane.expertBytes)));
  if (capacity < 1 || plan.numExperts * 4 > limit) throw new Error('Compact expert quota cannot hold a slot and source map.');
  return capacity;
}

export class CompactExpertCache {
  /** @type {import('./compact-expert-cache.js').CompactExpertCacheOptions} */
  #options;
  /** @type {import('./routed-expert-source.js').RoutedExpertCounters} */
  #stats;
  /** @type {Map<number, import('./compact-expert-cache.js').CompactExpertPool>} */
  #pools = new Map();
  /** @type {Set<GPUBuffer>} */
  #owned = new Set();
  #bytes = 0;
  #staging = 0;

  /** @param {import('./compact-expert-cache.js').CompactExpertCacheOptions} options
   * @param {import('./routed-expert-source.js').RoutedExpertCounters} stats */
  constructor(options, stats) {
    validateQwen35RoutedConfig(options.policy);
    this.#options = options;
    this.#stats = stats;
  }

  /** @param {number} bytes @param {string} label */
  #allocate(bytes, label) {
    const { device, policy } = this.#options;
    if (bytes > device.limits.maxBufferSize || bytes > device.limits.maxStorageBufferBindingSize
      || this.#options.otherGpuBytes() + this.#stats.residentBytes + bytes > policy.gpuBudgetBytes - policy.reserveBytes) {
      throw new Error('Compact expert allocation exceeds the owning device or GPU reserve.');
    }
    const buffer = device.createBuffer({ size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label });
    this.#owned.add(buffer);
    this.#bytes += buffer.size;
    this.#stats.residentBytes += buffer.size;
    this.#stats.peakResidentBytes = Math.max(this.#stats.peakResidentBytes, this.#stats.residentBytes);
    this.#stats.peakAccountedGpuBytes = Math.max(this.#stats.peakAccountedGpuBytes,
      this.#options.otherGpuBytes() + this.#stats.residentBytes);
    return buffer;
  }

  /** @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan */
  #pool(plan) {
    let pool = this.#pools.get(plan.layer);
    if (pool) return pool;
    const capacity = compactExpertCapacity(plan, this.#options);
    const allocated = [];
    try {
      const planes = new Map();
      for (const plane of plan.planes) {
        const buffer = this.#allocate(capacity * plane.expertBytes, `compact_${plan.layer}_${plane.role}`);
        allocated.push(buffer); planes.set(plane.role, buffer);
      }
      const map = this.#allocate(plan.numExperts * 4, `compact_${plan.layer}_source_map`);
      allocated.push(map);
      pool = { capacity, planes, map, stats: { cacheHits: 0, cacheMisses: 0, evictions: 0, sourceReadBytes: 0, uploadBytes: 0, sourceReadWaitMs: 0, uploadEnqueueMs: 0 }, entries: new Map(), free: Array.from({ length: capacity }, (_, i) => capacity - i - 1) };
      this.#pools.set(plan.layer, pool);
      return pool;
    } catch (error) {
      for (const buffer of allocated) this.#retire(buffer);
      throw error;
    }
  }

  /** @param {GPUBuffer} buffer */
  #retire(buffer) {
    buffer.destroy();
    if (this.#owned.delete(buffer)) {
      this.#bytes -= buffer.size;
      this.#stats.residentBytes -= buffer.size;
    }
  }

  /** @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan
   * @param {import('./compact-expert-cache.js').CompactExpertPool} pool
   * @param {number} expert @param {number} slot @param {AbortSignal} signal */
  async #fill(plan, pool, expert, slot, signal) {
    const chunkBytes = Math.floor(this.#options.policy.stagingBytes / (2 * this.#options.policy.readConcurrency * 4)) * 4;
    for (const plane of plan.planes) {
      for (let offset = 0; offset < plane.expertBytes;) {
        signal.throwIfAborted();
        const length = Math.min(chunkBytes, plane.expertBytes - offset);
        const staging = (this.#options.readRangeInto ? 1 : 2) * length;
        this.#staging += staging;
        this.#stats.peakStagingBytes = Math.max(this.#stats.peakStagingBytes, this.#staging);
        const start = performance.now();
        try {
          const data = await readRoutedExpertRange(this.#options, this.#stats, plane, expert, offset, length, signal);
          pool.stats.sourceReadBytes += data.byteLength;
          pool.stats.sourceReadWaitMs += performance.now() - start;
          signal.throwIfAborted();
          const uploadStart = performance.now();
          const buffer = pool.planes.get(plane.role);
          if (!buffer) throw new Error('Compact expert plane is missing.');
          this.#options.device.queue.writeBuffer(buffer, slot * plane.expertBytes + offset, data);
          const uploadMs = performance.now() - uploadStart;
          this.#stats.uploadEnqueueMs += uploadMs;
          pool.stats.uploadEnqueueMs += uploadMs;
          pool.stats.uploadBytes += length;
          this.#stats.uploadBytes += length;
        } finally {
          this.#stats.ioWaitMs += performance.now() - start;
          this.#staging -= staging;
        }
        offset += length;
      }
    }
  }

  /** @type {import('./compact-expert-cache.js').CompactExpertCache['prepare']} */
  async prepare(plan, ids, signal) {
    const pool = this.#pool(plan);
    if (ids.length > pool.capacity) throw new Error('Compact expert selection exceeds slot capacity; reduce route chunks.');
    const selected = new Set(ids);
    /** @type {Array<{expert: number, slot: number}>} */
    const misses = [];
    const slots = new Map();
    for (const expert of ids) {
      const slot = pool.entries.get(expert);
      if (slot !== undefined) {
        pool.entries.delete(expert); pool.entries.set(expert, slot);
        slots.set(expert, slot); this.#stats.cacheHits++; pool.stats.cacheHits++;
      }
    }
    for (const expert of ids) {
      if (slots.has(expert)) continue;
      if (!pool.free.length) {
        const victim = [...pool.entries].find(([key]) => !selected.has(key));
        if (!victim) throw new Error('Compact selected experts exhausted pinned slots.');
        pool.entries.delete(victim[0]); pool.free.push(victim[1]); this.#stats.evictions++; pool.stats.evictions++;
      }
      const slot = pool.free.pop();
      if (slot === undefined) throw new Error('Compact expert slot allocation failed.');
      slots.set(expert, slot); misses.push({ expert, slot }); this.#stats.cacheMisses++; pool.stats.cacheMisses++;
    }
    let cursor = 0;
    const controller = new AbortController();
    const abort = () => controller.abort(signal.reason);
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
    try {
      const outcomes = await Promise.allSettled(Array.from({ length: this.#options.policy.readConcurrency }, async () => {
        while (cursor < misses.length) {
          const { expert, slot } = misses[cursor++];
          try { await this.#fill(plan, pool, expert, slot, controller.signal); }
          catch (error) { controller.abort(error); throw error; }
        }
      }));
      const failed = outcomes.find(result => result.status === 'rejected');
      if (failed) throw failed.reason;
      signal.throwIfAborted();
      for (const { expert, slot } of misses) pool.entries.set(expert, slot);
      const sourceMap = new Uint32Array(plan.numExperts).fill(0xffffffff);
      for (const [expert, slot] of slots) sourceMap[expert] = slot;
      this.#options.device.queue.writeBuffer(pool.map, 0, sourceMap);
      const weights = Object.fromEntries(plan.planes.map(plane => {
        const buffer = pool.planes.get(plane.role);
        if (!buffer) throw new Error('Compact expert plane is missing.');
        return [plane.role, createWeightBuffer(buffer, plane.dtype, 'row',
          [pool.capacity, ...plane.location.shape.slice(1)], plane.name)];
      }));
      return { expertIdx: 0, expertFormat: 'qwen3_5', numExperts: pool.capacity,
        expertIntermediateSize: plan.intermediateSize, sourceNumExperts: plan.numExperts,
        sourceExpertToSlot: pool.map, ...weights, down: weights.down };
    } catch (error) {
      let cleanup;
      try { await this.#options.device.queue.onSubmittedWorkDone(); } catch (failure) { cleanup = failure; }
      for (const { expert, slot } of misses) { pool.entries.delete(expert); pool.free.push(slot); }
      if (cleanup) throw new AggregateError([error, cleanup], 'Compact expert loading and drain failed.');
      throw error;
    } finally { signal.removeEventListener('abort', abort); }
  }

  getStats() {
    return { cacheBytes: this.#bytes, workingBytes: 0,
      compactLayers: Object.fromEntries([...this.#pools].map(([layer, pool]) => [layer, { capacity: pool.capacity, ...pool.stats }])),
      experts: [...this.#pools.values()].reduce((sum, pool) => sum + pool.entries.size, 0) };
  }

  async destroy() {
    const errors = [];
    try { await this.#options.device.queue.onSubmittedWorkDone(); } catch (error) { errors.push(error); }
    for (const buffer of this.#owned) {
      try { this.#retire(buffer); } catch (error) { errors.push(error); }
    }
    if (!this.#owned.size) this.#pools.clear();
    if (errors.length) throw new AggregateError(errors, 'Compact expert cleanup failed.');
  }
}
