import { DEFAULT_EXPERT_CACHE_CONFIG, validateQwen35RoutedConfig } from "../../config/schema/runtime/loading.schema.js";
import { createWeightBuffer } from '../../gpu/weight-buffer.js';
import { isDeviceLost } from '../../memory/device-residency.js';
import { readRoutedExpertRange, readRoutedExpertPayload, readRoutedExpertsLease, readRoutedExpertPlaneLeases, recordRoutedExpertSourceTimings, assertExpertReadShape } from './routed-expert-source.js';

/** @type {typeof import('./compact-expert-cache.js').compactExpertCapacity} */
export function compactExpertCapacity(plan, options) {
  const layers = options.compactLayerCount;
  if (!Number.isSafeInteger(layers) || layers < 1) throw new Error('Compact experts require the declared layer count.');
  const quotas = options.policy.compactLayerBytes;
  if (quotas !== null && (Object.keys(quotas).length !== layers || !Object.hasOwn(quotas, plan.layer)
    || Object.keys(quotas).some(layer => Number(layer) >= plan.numLayers))) {
    throw new Error('compactLayerBytes must cover every assigned routed layer at its original index.');
  }
  const layerBytes = quotas === null ? Math.floor((options.policy.cacheBytes - options.policy.compactPrefillBytes) / layers) : quotas[plan.layer];
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
  /** @type {Array<{ bytes: Uint8Array | null }>} */
  #readers = [];

  /** @param {import('./compact-expert-cache.js').CompactExpertCacheOptions} options
   * @param {import('./routed-expert-source.js').RoutedExpertCounters} stats */
  constructor(options, stats) {
    validateQwen35RoutedConfig(options.policy);
    this.#options = options;
    this.#stats = stats;
    const lease = options.rangeLease;
    if (lease && (!Number.isSafeInteger(lease.concurrency ?? 1) || (lease.concurrency ?? 1) < 1
      || !Number.isSafeInteger(lease.maxPartBytes) || lease.maxPartBytes < 4
      || !Number.isSafeInteger(lease.memoryBytes) || lease.memoryBytes < lease.maxPartBytes * (lease.concurrency ?? 1)
      || lease.memoryBytes > options.policy.stagingBytes - 8 * options.policy.readConcurrency)) {
      throw new Error('Owned expert ranges must fit the aggregate staging budget with fallback scratch.');
    }
    this.#staging = lease?.memoryBytes ?? 0;
    this.#stats.peakStagingBytes = Math.max(this.#stats.peakStagingBytes, this.#staging);
  }

  #chunkBytes() {
    const { stagingBytes, readConcurrency } = this.#options.policy;
    return Math.floor((stagingBytes - (this.#options.rangeLease?.memoryBytes ?? 0)) / (2 * readConcurrency * 4)) * 4;
  }

  #clearReaderStaging() {
    for (const reader of this.#readers) this.#staging -= reader.bytes?.byteLength ?? 0;
    this.#readers = [];
  }

  /** @param {number} bytes @param {string} label */
  #allocate(bytes, label) {
    const { device, policy } = this.#options;
    if (bytes > device.limits.maxBufferSize || bytes > device.limits.maxStorageBufferBindingSize
      || this.#options.otherGpuBytes() + this.#stats.residentBytes + bytes > policy.gpuBudgetBytes - policy.reserveBytes) {
      throw new Error('Compact expert allocation exceeds the owning device or GPU reserve.');
    }
    const buffer = device.createBuffer({ size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label });
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

  /** Authenticate every plane before any upload, while hashing on separate workers.
   * @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan
   * @param {import('./compact-expert-cache.js').CompactExpertPool} pool
   * @param {{expert: number, slot: number}} miss @param {AbortSignal} signal */
  async #fillSinglePlanes(plan, pool, { expert, slot }, signal) {
    const started = performance.now();
    const leases = await readRoutedExpertPlaneLeases(this.#options, this.#stats, plan, expert, signal, pool.stats);
    let failure;
    try {
      signal.throwIfAborted();
      for (const [index, plane] of plan.planes.entries()) {
        signal.throwIfAborted();
        const buffer = pool.planes.get(plane.role);
        if (!buffer) throw new Error('Compact expert plane is missing.');
        const uploadStart = performance.now();
        this.#options.device.queue.writeBuffer(buffer, slot * plane.expertBytes, leases[index].bytes);
        const elapsed = performance.now() - uploadStart;
        this.#stats.uploadEnqueueMs += elapsed;
        pool.stats.uploadEnqueueMs += elapsed;
        this.#stats.uploadBytes += plane.expertBytes;
        pool.stats.uploadBytes += plane.expertBytes;
      }
    } catch (error) { failure = error; throw error; }
    finally {
      const errors = [];
      for (const lease of leases) {
        try { lease.release(); } catch (error) { errors.push(error); }
      }
      this.#stats.ioWaitMs += performance.now() - started;
      if (errors.length) throw new AggregateError(failure ? [failure, ...errors] : errors,
        'Parallel expert upload and lease release failed.');
    }
  }

  /** @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan
   * @param {import('./compact-expert-cache.js').CompactExpertPool} pool
   * @param {number} expert @param {number} slot @param {AbortSignal} signal
   * @param {{ bytes: Uint8Array | null }} reader
   * @param {import('./routed-expert-plan.js').RoutedExpertPlane | null} selectedPlane */
  async #fill(plan, pool, expert, slot, signal, reader, selectedPlane) {
    if (this.#options.policy.sourceReadMode === 'expert-lease-batch') {
      return this.#fillBatch(plan, pool, [{ expert, slot }], signal);
    }
    if (this.#copyCached(plan, pool, expert, slot, signal)) return;
    const chunkBytes = this.#chunkBytes();
    if (this.#options.policy.sourceReadMode === 'expert-parallel') {
      if (!reader.bytes) {
        reader.bytes = new Uint8Array(chunkBytes);
        this.#staging += chunkBytes;
        this.#stats.stagingAllocations++;
      } else this.#stats.stagingReuses++;
      this.#stats.peakStagingBytes = Math.max(this.#stats.peakStagingBytes, this.#staging);
      const destination = reader.bytes.subarray(0, plan.expertBytes);
      const started = performance.now();
      try {
        await readRoutedExpertPayload(this.#options, this.#stats, plan, expert, signal, destination, pool.stats);
        signal.throwIfAborted();
        let offset = 0;
        for (const plane of plan.planes) {
          const buffer = pool.planes.get(plane.role);
          if (!buffer) throw new Error('Compact expert plane is missing.');
          const uploadStart = performance.now();
          this.#options.device.queue.writeBuffer(buffer, slot * plane.expertBytes,
            destination.subarray(offset, offset + plane.expertBytes));
          const uploadMs = performance.now() - uploadStart;
          this.#stats.uploadEnqueueMs += uploadMs;
          pool.stats.uploadEnqueueMs += uploadMs;
          pool.stats.uploadBytes += plane.expertBytes;
          this.#stats.uploadBytes += plane.expertBytes;
          offset += plane.expertBytes;
        }
      } finally { this.#stats.ioWaitMs += performance.now() - started; }
      return;
    }
    for (const plane of selectedPlane === null ? plan.planes : [selectedPlane]) {
      for (let offset = 0; offset < plane.expertBytes;) {
        signal.throwIfAborted();
        const port = this.#options.rangeLease;
        const location = plane.location;
        const physical = location.spans?.length === 1 ? location.spans[0] : location;
        const leased = port && location.sourceTransform == null && (!location.spans || location.spans.length <= 1);
        const length = Math.min(leased ? Math.floor(port.maxPartBytes / 4) * 4 : chunkBytes, plane.expertBytes - offset);
        let destination = null;
        if (!leased && this.#options.readRangeInto && plane.location.sourceTransform == null) {
          if (!reader.bytes) {
            reader.bytes = new Uint8Array(chunkBytes);
            this.#staging += chunkBytes;
            this.#stats.stagingAllocations++;
          } else this.#stats.stagingReuses++;
          // writeBuffer copies before returning, so only this reader may reuse the bytes.
          destination = reader.bytes.subarray(0, length);
        }
        const staging = leased || destination ? 0 : (this.#options.readRangeInto ? 1 : 2) * length;
        this.#staging += staging;
        this.#stats.peakStagingBytes = Math.max(this.#stats.peakStagingBytes, this.#staging);
        const start = performance.now();
        /** @type {import('../../storage/verified-piece-gather.js').VerifiedRangeLease | null} */
        let lease = null;
        try {
          if (leased) {
            const shard = physical.shardIndex;
            if (typeof shard !== 'number' || typeof physical.offset !== 'number' || !Number.isSafeInteger(shard) || shard < 0 || !Number.isSafeInteger(physical.offset)
              || physical.offset < 0 || !Number.isSafeInteger(physical.size) || physical.size < 0
              || !Number.isSafeInteger(expert) || expert < 0 || offset + length > plane.expertBytes || (expert + 1) * plane.expertBytes > physical.size) {
              throw new Error('Owned expert range exceeds its declared physical plane.');
            }
            lease = await port.read(shard, physical.offset + expert * plane.expertBytes + offset, length, signal);
            if (lease.bytes.byteLength !== length) throw new Error('Owned expert range length mismatch.');
            signal.throwIfAborted();
            recordRoutedExpertSourceTimings(this.#stats, [lease]);
            this.#stats.sourceReads++;
            this.#stats.sourceReadBytes += length;
            this.#stats.sourceReadWaitMs += performance.now() - start;
          }
          const data = lease ? lease.bytes : await readRoutedExpertRange(this.#options, this.#stats, plane, expert, offset, length, signal, destination);
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
          // queue.writeBuffer synchronously copies its source, including leased views.
          lease?.release();
          this.#stats.ioWaitMs += performance.now() - start;
          this.#staging -= staging;
        }
        offset += length;
      }
    }
  }

  /** @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan
   * @param {import('./compact-expert-cache.js').CompactExpertPool} pool
   * @param {Array<{expert: number, slot: number}>} misses @param {AbortSignal} signal */
  async #fillBatch(plan, pool, misses, signal) {
    const pending = misses.filter(({ expert, slot }) => !this.#copyCached(plan, pool, expert, slot, signal));
    if (!pending.length) return;
    const started = performance.now();
    const lease = await readRoutedExpertsLease(this.#options, this.#stats, plan, pending.map(row => row.expert), signal, pool.stats);
    try {
      for (const [index, { expert, slot }] of pending.entries()) {
        const bytes = lease.bytes.subarray(index * plan.expertBytes, (index + 1) * plan.expertBytes);
        await this.#options.rememberExpert?.(plan, expert, bytes, signal);
        if (this.#copyCached(plan, pool, expert, slot, signal)) continue;
        let offset = 0;
        for (const plane of plan.planes) {
          signal.throwIfAborted();
          const buffer = pool.planes.get(plane.role);
          if (!buffer) throw new Error('Compact expert plane is missing.');
          const uploadStart = performance.now();
          this.#options.device.queue.writeBuffer(buffer, slot * plane.expertBytes,
            bytes.subarray(offset, offset + plane.expertBytes));
          const elapsed = performance.now() - uploadStart;
          this.#stats.uploadEnqueueMs += elapsed;
          pool.stats.uploadEnqueueMs += elapsed;
          this.#stats.uploadBytes += plane.expertBytes;
          pool.stats.uploadBytes += plane.expertBytes;
          offset += plane.expertBytes;
        }
      }
    } finally {
      lease.release();
      this.#stats.ioWaitMs += performance.now() - started;
    }
  }

  /** @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan
   * @param {import('./compact-expert-cache.js').CompactExpertPool} pool
   * @param {number} expert @param {number} slot @param {AbortSignal} signal */
  #copyCached(plan, pool, expert, slot, signal) {
    if (!this.#options.copyCachedExpert) return false;
    const start = performance.now();
    if (!this.#options.copyCachedExpert(plan, expert, pool.planes, slot, signal)) return false;
    const elapsed = performance.now() - start;
    pool.stats.uploadBytes += plan.expertBytes;
    pool.stats.uploadEnqueueMs += elapsed;
    this.#stats.uploadEnqueueMs += elapsed;
    return true;
  }

  /** @type {import('./compact-expert-cache.js').CompactExpertCache['prepare']} */
  async prepare(plan, ids, signal, residentPlanes = null) {
    assertExpertReadShape(this.#options, plan);
    const chunkBytes = this.#chunkBytes();
    if (!this.#options.readRangeInto || this.#readers.length > this.#options.policy.readConcurrency
      || this.#readers.some(reader => (reader.bytes?.byteLength ?? 0) > chunkBytes)) {
      this.#clearReaderStaging();
    }
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
        /** @type {[number, number] | undefined} */
        let victim;
        for (const entry of pool.entries) {
          if (!selected.has(entry[0])) { victim = entry; break; }
        }
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
      if (residentPlanes !== null) {
        signal.throwIfAborted();
        const encoder = this.#options.device.createCommandEncoder({ label: 'prefill_selected_expert_seed' });
        let bytes = 0;
        for (const { expert, slot } of misses) {
          for (const plane of plan.planes) {
            const source = residentPlanes.get(plane.role);
            const target = pool.planes.get(plane.role);
            if (!source || !target || source.size < plan.numExperts * plane.expertBytes
              || !(source.usage & GPUBufferUsage.COPY_SRC) || !(target.usage & GPUBufferUsage.COPY_DST)) {
              throw new Error('Prefill seed requires complete owned planes with declared copy usage.');
            }
            encoder.copyBufferToBuffer(source, expert * plane.expertBytes,
              target, slot * plane.expertBytes, plane.expertBytes);
            bytes += plane.expertBytes;
          }
        }
        signal.throwIfAborted();
        if (misses.length) {
          this.#options.device.queue.submit([encoder.finish()]);
          this.#stats.prefillSeedBytes += bytes;
          this.#stats.prefillSeedSubmissions++;
        }
      } else if (this.#options.policy.sourceReadMode === 'expert-lease-batch' && misses.length === 1
        && Math.min(this.#options.policy.readConcurrency, this.#options.rangeLease?.concurrency ?? 1) >= plan.planes.length
        && !this.#options.copyCachedExpert && !this.#options.rememberExpert) {
        await this.#fillSinglePlanes(plan, pool, misses[0], controller.signal);
      } else {
        // A small miss set must still use the admitted lease lanes. Flatten only
        // the explicit leased mode; slots stay unpublished until every plane settles.
        const lanes = this.#options.rangeLease
          ? Math.min(this.#options.policy.readConcurrency, this.#options.rangeLease.concurrency ?? 1)
          : this.#options.policy.readConcurrency;
        const parallelPlanes = this.#options.policy.sourceReadMode === 'lease-parallel';
        const tasksPerExpert = parallelPlanes ? plan.planes.length : 1;
        const requestedBatch = this.#options.policy.expertBatchSize ?? DEFAULT_EXPERT_CACHE_CONFIG.qwen35Routed.expertBatchSize;
        // Keep admitted lanes busy before amortizing several experts in one job.
        // A fixed maximum-sized group serializes small miss sets on one worker.
        const batchSize = this.#options.policy.sourceReadMode === 'expert-lease-batch'
          ? Math.max(1, Math.min(requestedBatch, Math.ceil(misses.length / lanes),
            Math.floor(Number(this.#options.rangeLease?.maxPartBytes) / plan.expertBytes))) : 1;
        const taskCount = batchSize > 1 ? Math.ceil(misses.length / batchSize) : misses.length * tasksPerExpert;
        const outcomes = await Promise.allSettled(Array.from({ length: lanes }, async (_, index) => {
          const reader = this.#readers[index] ??= { bytes: null };
          while (cursor < taskCount) {
            const task = cursor++;
            try {
              if (batchSize > 1) await this.#fillBatch(plan, pool, misses.slice(task * batchSize, (task + 1) * batchSize), controller.signal);
              else {
                const { expert, slot } = misses[Math.floor(task / tasksPerExpert)];
                const plane = parallelPlanes ? plan.planes[task % tasksPerExpert] : null;
                await this.#fill(plan, pool, expert, slot, controller.signal, reader, plane);
              }
            }
            catch (error) { controller.abort(error); throw error; }
          }
        }));
        const failed = outcomes.find(result => result.status === 'rejected');
        if (failed) throw failed.reason;
      }
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

  /** @param {number} layer @param {number[]} ids */
  getResidentSlots(layer, ids) {
    const pool = this.#pools.get(layer);
    const slots = new Map();
    for (const expert of ids) {
      const slot = pool?.entries.get(expert);
      if (slot !== undefined) slots.set(expert, slot);
    }
    return slots;
  }

  /** Encode verified resident bytes at their original logical expert offsets.
   * This does not publish working-bank readiness or change compact LRU order.
   * @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan
   * @param {ReadonlyMap<number, number>} slots
   * @param {ReadonlyMap<string, GPUBuffer>} destinations
   * @param {GPUCommandEncoder} encoder */
  encodeResidentCopies(plan, slots, destinations, encoder) {
    const pool = this.#pools.get(plan.layer);
    if (!pool) throw new Error('Resident prefill copy requires the owning compact pool.');
    for (const [expert, slot] of slots) {
      if (pool.entries.get(expert) !== slot) throw new Error('Resident prefill source slot changed.');
      for (const plane of plan.planes) {
        const source = pool.planes.get(plane.role);
        const target = destinations.get(plane.role);
        if (!source || !target || source.size < pool.capacity * plane.expertBytes
            || target.size < plane.bytes || !(source.usage & GPUBufferUsage.COPY_SRC)
            || !(target.usage & GPUBufferUsage.COPY_DST)) {
          throw new Error('Resident prefill plane does not satisfy the declared copy contract.');
        }
        encoder.copyBufferToBuffer(source, slot * plane.expertBytes,
          target, expert * plane.expertBytes, plane.expertBytes);
      }
    }
  }

  /** Release generic fallback scratch before the owning cache changes execution phase. */
  clearStaging() { this.#clearReaderStaging(); }

  /** @param {number} layer */
  hasLayer(layer) { return (this.#pools.get(layer)?.entries.size ?? 0) > 0; }

  get residentBytes() { return this.#bytes; }

  getStats() {
    return { cacheBytes: this.#bytes, workingBytes: 0,
      compactLayers: Object.fromEntries([...this.#pools].map(([layer, pool]) => [layer, { capacity: pool.capacity, ...pool.stats }])),
      experts: [...this.#pools.values()].reduce((sum, pool) => sum + pool.entries.size, 0) };
  }

  async destroy() {
    const errors = [];
    try { await this.#options.device.queue.onSubmittedWorkDone(); }
    catch (error) { if (!isDeviceLost(this.#options.device)) errors.push(error); }
    this.#clearReaderStaging();
    for (const buffer of this.#owned) {
      try { this.#retire(buffer); } catch (error) { errors.push(error); }
    }
    if (!this.#owned.size) this.#pools.clear();
    if (errors.length) throw new AggregateError(errors, 'Compact expert cleanup failed.');
  }
}
