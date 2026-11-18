import { createWeightBuffer } from '../../gpu/weight-buffer.js';
import { isDeviceLost, observeDeviceLoss } from '../../memory/device-residency.js';
import { CompactExpertCache } from './compact-expert-cache.js';
import { MappedExpertUploadCache } from './mapped-expert-upload-cache.js';
import { planTieredExpertPlacement } from './tiered-expert-plan.js';
import { assertExpertReadShape, createRoutedExpertCounters, readRoutedExpertLease, readRoutedExpertPayload } from './routed-expert-source.js';

// Placement is explicit and model-bound. Setup owns resident banks; streamed
// layers keep compact slots in both phases, never a second full prefill bank.
export class TieredExpertCache {
  /** @type {import('./routed-expert-cache.js').RoutedExpertCacheOptions} */
  #options;
  /** @type {import('./tiered-expert-plan.js').TieredExpertPlacement} */
  #placement;
  #stats = createRoutedExpertCounters();
  /** @type {CompactExpertCache | null} */
  #compact = null;
  /** @type {MappedExpertUploadCache | null} */
  #mapped = null;
  /** @type {Map<number, import('./routed-expert-cache.js').RoutedExpertWeights>} */
  #resident = new Map();
  /** @type {Set<GPUBuffer>} */
  #owned = new Set();
  #shutdown = new AbortController();
  #closed = false;
  #ready = false;
  #uploadingBytes = 0;
  /** @type {'prefill' | 'decode' | null} */
  #phase = null;
  /** @type {Promise<unknown> | null} */
  #active = null;

  /** @param {import('./routed-expert-cache.js').RoutedExpertCacheOptions} options
   * @param {readonly import('./routed-expert-plan.js').RoutedExpertBankPlan[]} plans @param {number} minimumSlots */
  constructor(options, plans, minimumSlots) {
    this.#options = options;
    observeDeviceLoss(options.device);
    this.#placement = planTieredExpertPlacement(plans, options.policy, options.device.limits, minimumSlots);
    for (const plan of plans) assertExpertReadShape(options, plan);
    const lease = options.rangeLease;
    if (options.policy.sourceReadMode === 'expert-parallel') {
      if (lease || !options.readRangeInto
        || options.policy.stagingBytes < Math.max(...plans.map(plan => plan.expertBytes)) * 2 * options.policy.readConcurrency) {
        throw new Error('Tiered parallel expert reads require bounded direct scratch within stagingBytes.');
      }
    } else {
      const mappedLimit = options.policy.stagingBytes - (lease?.memoryBytes ?? 0);
      const lanes = Math.min(options.policy.readConcurrency, lease?.concurrency ?? 1);
      const leaseLanes = lease?.concurrency ?? 1;
      if (!lease || !Number.isSafeInteger(leaseLanes) || leaseLanes < 1
        || !Number.isSafeInteger(lease.maxPartBytes) || lease.maxPartBytes < 4 || lease.maxPartBytes % 4
        || !Number.isSafeInteger(lease.memoryBytes) || !Number.isSafeInteger(lease.maxPartBytes * leaseLanes)
        || lease.memoryBytes < lease.maxPartBytes * leaseLanes
        || mappedLimit < Math.max(...plans.map(plan => plan.expertBytes)) * lanes) {
        throw new Error('Tiered expert leases and concurrent mapped writers must fit aggregate stagingBytes.');
      }
    }
    const mappedLimit = options.policy.stagingBytes - (lease?.memoryBytes ?? 0);
    if (options.otherGpuBytes() + this.#placement.cacheUpperBytes > options.policy.gpuBudgetBytes - options.policy.reserveBytes) {
      throw new Error('Tiered expert placement and existing weights exceed the GPU reserve.');
    }
    const mappedBytes = this.#placement.parkedBytes + this.#placement.mappedCacheBytes;
    if (mappedBytes) this.#mapped = new MappedExpertUploadCache({ device: options.device,
      maxBytes: mappedBytes, maxMappedBytes: Math.min(mappedLimit, mappedBytes),
      maxEntries: plans.reduce((sum, plan) => sum + plan.numExperts, 0),
      admission: options.policy.hostCacheAdmission,
      protectedShare: options.policy.tiered?.mappedProtectedShare,
      maxGhostEntries: Math.max(1, Math.floor(this.#placement.mappedCacheBytes / Math.min(...plans.map(plan => plan.expertBytes)))),
      allocate: descriptor => this.#allocate(descriptor), dispose: buffer => this.#retire(buffer) });
    const compactLayerCount = [...this.#placement.layers.values()].filter(row => row.mode !== 'resident').length;
    if (compactLayerCount) {
      const policy = { ...options.policy, storageMode: /** @type {const} */ ('compact-decode'), tiered: null,
        cacheBytes: this.#placement.compactBytes, compactLayerBytes: this.#placement.compactLayerBytes };
      this.#compact = new CompactExpertCache({ ...options, policy, compactLayerCount,
        copyCachedExpert: (plan, expert, planes, slot, signal) => this.#copy(plan, expert, planes, slot, signal),
        rememberExpert: (plan, expert, bytes, signal) => this.#remember(plan, expert, bytes, signal) }, this.#stats);
    }
  }

  /** @param {GPUBufferDescriptor} descriptor */
  #allocate(descriptor) {
    const bytes = Number(descriptor.size), { device, policy } = this.#options;
    if (!Number.isSafeInteger(bytes) || bytes < 4 || bytes > device.limits.maxBufferSize
      || this.#stats.residentBytes + bytes > this.#placement.cacheUpperBytes
      || this.#options.otherGpuBytes() + this.#stats.residentBytes + bytes > policy.gpuBudgetBytes - policy.reserveBytes) {
      throw new Error('Tiered expert allocation exceeds admitted cache or device budget.');
    }
    const buffer = device.createBuffer(descriptor);
    this.#owned.add(buffer);
    this.#stats.residentBytes += buffer.size;
    this.#stats.peakResidentBytes = Math.max(this.#stats.peakResidentBytes, this.#stats.residentBytes);
    this.#stats.peakAccountedGpuBytes = Math.max(this.#stats.peakAccountedGpuBytes,
      this.#options.otherGpuBytes() + this.#stats.residentBytes);
    return buffer;
  }

  /** @param {GPUBuffer} buffer */
  #retire(buffer) {
    buffer.destroy();
    if (this.#owned.delete(buffer)) this.#stats.residentBytes -= buffer.size;
  }

  /** @template T @param {AbortSignal | undefined} external @param {(signal: AbortSignal) => Promise<T>} action */
  #run(external, action) {
    if (this.#closed || this.#active || isDeviceLost(this.#options.device)) throw new Error('Tiered expert cache is closed, lost or already preparing.');
    const controller = new AbortController();
    const abort = () => controller.abort(external?.reason);
    const shutdown = () => controller.abort(this.#shutdown.signal.reason);
    external?.addEventListener('abort', abort, { once: true });
    this.#shutdown.signal.addEventListener('abort', shutdown, { once: true });
    if (external?.aborted) abort();
    const task = Promise.resolve().then(() => { controller.signal.throwIfAborted(); return action(controller.signal); });
    this.#active = task;
    return task.finally(() => {
      external?.removeEventListener('abort', abort);
      this.#shutdown.signal.removeEventListener('abort', shutdown);
      this.#active = null;
    });
  }

  /** @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan @param {number} expert */
  #key(plan, expert) { return `${plan.layer}:${expert}`; }

  /** @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan @param {number} expert
   * @param {Uint8Array} bytes @param {AbortSignal} signal */
  async #remember(plan, expert, bytes, signal) {
    if (!this.#mapped) return;
    const pinned = this.#placement.layers.get(plan.layer)?.mode === 'parked';
    if (!pinned && this.#options.policy.hostCacheAdmissionPhase === 'decode' && this.#phase !== 'decode') return;
    const accepted = await this.#mapped.admit(this.#key(plan, expert), plan.expertBytes, destination => {
      signal.throwIfAborted();
      this.#sampleStaging();
      destination.set(bytes);
      return bytes.byteLength;
    }, { pinned, signal });
    if (pinned && !accepted) throw new Error('Declared parked expert could not be admitted.');
    this.#sampleStaging();
  }

  /** @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan @param {number} expert
   * @param {ReadonlyMap<string, GPUBuffer>} planes @param {number} slot @param {AbortSignal} signal */
  #copy(plan, expert, planes, slot, signal) {
    let offset = 0;
    const ranges = plan.planes.map(plane => {
      const destination = planes.get(plane.role);
      if (!destination) throw new Error('Tiered compact copy requires all declared destination planes.');
      const range = { sourceOffset: offset, destination, destinationOffset: slot * plane.expertBytes, bytes: plane.expertBytes };
      offset += plane.expertBytes;
      return range;
    });
    const copy = this.#mapped?.copyInto(this.#key(plan, expert), ranges, signal);
    if (!copy) return false;
    // Queue ordering makes the data ready for subsequent kernels. The mapped
    // cache independently retains source ownership until this actual fence settles.
    this.#stats.uploadBytes += copy.bytes;
    this.#stats.expertCopySubmissions++;
    return true;
  }

  #sampleStaging() {
    this.#stats.peakStagingBytes = Math.max(this.#stats.peakStagingBytes,
      (this.#options.rangeLease?.memoryBytes ?? 0) + (this.#mapped?.getStats().mappedBytes ?? 0) + this.#uploadingBytes);
  }

  /** @type {import('./tiered-expert-cache.js').TieredExpertCache['preload']} */
  preload(external) {
    return this.#run(external, signal => this.#preload(signal));
  }

  /** @param {AbortSignal} signal */
  async #preload(signal) {
    if (this.#ready) return;
    // Failed setup is retired before retry, never reused as a complete bank.
    if (this.#owned.size) throw new Error('Failed tiered setup requires unload before retry.');
    for (const { plan, mode } of this.#placement.layers.values()) {
      if (mode === 'streamed') continue;
      /** @type {Map<string, GPUBuffer>} */
      const planes = new Map();
      if (mode === 'resident') {
        for (const plane of plan.planes) planes.set(plane.role, this.#allocate({ size: plane.bytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
          label: `tiered_resident_${plan.layer}_${plane.role}` }));
      }
      let cursor = 0;
      const controller = new AbortController();
      const abort = () => controller.abort(signal.reason);
      signal.addEventListener('abort', abort, { once: true });
      if (signal.aborted) abort();
      try {
        const directParallel = this.#options.policy.sourceReadMode === 'expert-parallel';
        const concurrency = directParallel
          ? this.#options.policy.readConcurrency
          : Math.min(this.#options.policy.readConcurrency, this.#options.rangeLease?.concurrency ?? 1);
        const results = await Promise.allSettled(Array.from({ length: concurrency }, async () => {
          const directScratch = directParallel ? new Uint8Array(plan.expertBytes) : null;
          if (directScratch) {
            this.#uploadingBytes += directScratch.byteLength;
            this.#sampleStaging();
          }
          try {
            while (cursor < plan.numExperts) {
              const expert = cursor++;
              try {
                if (directScratch) {
                  await readRoutedExpertPayload(this.#options, this.#stats, plan, expert, controller.signal, directScratch, null);
                  controller.signal.throwIfAborted();
                  this.#uploadingBytes += plan.expertBytes;
                  this.#sampleStaging();
                  try {
                    let offset = 0;
                    for (const plane of plan.planes) {
                      const buffer = planes.get(plane.role);
                      if (!buffer) throw new Error('Tiered resident plane is missing.');
                      this.#options.device.queue.writeBuffer(buffer, expert * plane.expertBytes,
                        directScratch.subarray(offset, offset + plane.expertBytes));
                      this.#stats.uploadBytes += plane.expertBytes;
                      offset += plane.expertBytes;
                    }
                    await this.#options.device.queue.onSubmittedWorkDone();
                    controller.signal.throwIfAborted();
                  } finally { this.#uploadingBytes -= plan.expertBytes; }
                } else {
                  const lease = await readRoutedExpertLease(this.#options, this.#stats, plan, expert, controller.signal, null);
                  try {
                    controller.signal.throwIfAborted();
                    if (mode === 'parked') await this.#remember(plan, expert, lease.bytes, controller.signal);
                    else {
                      this.#uploadingBytes += plan.expertBytes;
                      this.#sampleStaging();
                      try {
                        let offset = 0;
                        for (const plane of plan.planes) {
                          const buffer = planes.get(plane.role);
                          if (!buffer) throw new Error('Tiered resident plane is missing.');
                          this.#options.device.queue.writeBuffer(buffer, expert * plane.expertBytes,
                            lease.bytes.subarray(offset, offset + plane.expertBytes));
                          this.#stats.uploadBytes += plane.expertBytes;
                          offset += plane.expertBytes;
                        }
                        // writeBuffer copies its source immediately, but its native
                        // upload allocation may survive until queue completion. Keep
                        // this lane charged and stop further reads until that upload
                        // settles; model-sized setup must not queue model-sized staging.
                        await this.#options.device.queue.onSubmittedWorkDone();
                        controller.signal.throwIfAborted();
                      } finally { this.#uploadingBytes -= plan.expertBytes; }
                    }
                  } finally { lease.release(); }
                }
              } catch (error) { controller.abort(error); throw error; }
            }
          } finally {
            if (directScratch) this.#uploadingBytes -= directScratch.byteLength;
          }
        }));
        const failure = results.find(result => result.status === 'rejected');
        if (failure) throw failure.reason;
        signal.throwIfAborted();
        if (mode === 'resident') {
          const weights = Object.fromEntries(plan.planes.map(plane => [plane.role,
            createWeightBuffer(/** @type {GPUBuffer} */ (planes.get(plane.role)), plane.dtype, 'row', plane.location.shape, plane.name)]));
          this.#resident.set(plan.layer, { expertIdx: 0, expertFormat: 'qwen3_5', numExperts: plan.numExperts,
            expertIntermediateSize: plan.intermediateSize, ...weights, down: weights.down });
        }
      } finally { signal.removeEventListener('abort', abort); }
    }
    await this.#options.device.queue.onSubmittedWorkDone();
    signal.throwIfAborted();
    this.#ready = true;
    this.#sampleStaging();
  }

  /** @type {import('./tiered-expert-cache.js').TieredExpertCache['getResidentWeights']} */
  getResidentWeights(layer) {
    return this.#ready && !this.#closed && !isDeviceLost(this.#options.device) ? this.#resident.get(layer) ?? null : null;
  }

  /** @type {import('./tiered-expert-cache.js').TieredExpertCache['capacity']} */
  capacity(layer) { return this.#placement.layers.get(layer)?.capacity ?? null; }

  /** @type {import('./tiered-expert-cache.js').TieredExpertCache['prepare']} */
  prepare(plan, ids, external, phase) {
    return this.#run(external, async signal => {
      if (!this.#ready || (phase !== 'prefill' && phase !== 'decode')) throw new Error('Tiered experts require completed setup and an explicit phase.');
      const row = this.#placement.layers.get(plan.layer);
      if (!row || row.plan !== plan || !ids.length || ids.some(id => !Number.isSafeInteger(id) || id < 0 || id >= plan.numExperts)) {
        throw new Error('Tiered selection must bind its admitted source plan and expert IDs.');
      }
      const resident = this.#resident.get(plan.layer);
      if (resident) return resident;
      if (!this.#compact) throw new Error('Tiered compact storage is absent.');
      this.#phase = phase;
      try { return await this.#compact.prepare(plan, [...new Set(ids)], signal); }
      finally { this.#phase = null; }
    });
  }

  getStats() {
    const compact = this.#compact?.getStats();
    return { ...this.#stats, evictionPolicy: /** @type {const} */ ('lru'), frequencyHistoryEntries: 0,
      frequencyResidentEntries: 0, phaseProfiles: {}, experts: (compact?.experts ?? 0)
        + [...this.#resident.values()].reduce((sum, row) => sum + row.numExperts, 0),
      cacheBytes: this.#stats.residentBytes, workingBytes: 0, compactLayers: compact?.compactLayers ?? {},
      hostCacheBytes: 0, hostReservedBytes: 0, hostExperts: 0,
      tiered: { residentLayers: [...this.#resident.keys()], ready: this.#ready, mapped: this.#mapped?.getStats() ?? null } };
  }

  async destroy() {
    this.#closed = true;
    this.#ready = false;
    this.#shutdown.abort(new DOMException('Tiered experts unloaded.', 'AbortError'));
    await this.#active?.catch(() => undefined);
    const errors = [];
    try { await this.#mapped?.close(); } catch (error) { errors.push(error); }
    try { await this.#compact?.destroy(); } catch (error) { errors.push(error); }
    try { await this.#options.device.queue.onSubmittedWorkDone(); }
    catch (error) { if (!isDeviceLost(this.#options.device)) errors.push(error); }
    // Mapped allocations retain their own retryable ownership if close failed.
    if (!errors.length) {
      for (const buffer of this.#owned) {
        try { this.#retire(buffer); } catch (error) { errors.push(error); }
      }
    }
    if (!this.#owned.size) this.#resident.clear();
    if (errors.length) throw new AggregateError(errors, 'Tiered expert cleanup failed; ownership retained for retry.');
  }
}
