import { ExpertFrequencyIndex } from './expert-frequency-index.js';
import { getRuntimeConfig } from '../../config/runtime.js';
import { CompactExpertCache, compactExpertCapacity } from './compact-expert-cache.js';
import { createWeightBuffer } from '../../gpu/weight-buffer.js';
import { validateQwen35RoutedConfig } from "../../config/schema/runtime/loading.schema.js";
import { createRoutedExpertCounters, readRoutedExpertRange, assertExpertReadShape, readRoutedExpertPayload } from './routed-expert-source.js';

import { writeRoutedPrefillExpert, selectPrefillSeedExperts } from './routed-expert-prefill.js';

// The cache owns individual expert payloads. One reusable logical bank preserves
// the existing kernels' expert IDs, source-format byte offsets and accumulation order.
export class RoutedExpertCache {
  /** @type {import('./routed-expert-cache.js').RoutedExpertCacheOptions} */
  #options;
  /** @type {Map<string, import('./routed-expert-cache.js').RoutedExpertEntry>} */
  #entries = new Map();
  /** @type {ExpertFrequencyIndex | null} */
  #frequency = null;
  #numLayers = 0;
  #numExperts = 0;
  /** @type {Set<GPUBuffer>} */
  #owned = new Set();
  /** @type {Map<string, GPUBuffer>} */
  #working = new Map();
  #loaded = new Set();
  #layer = -1;
  #signature = '';
  #cacheBytes = 0;
  #submittedSerial = 0;
  #completedSerial = 0;
  /** @type {Map<string, import('./routed-expert-cache.js').RoutedHostExpertEntry>} */
  #hostEntries = new Map();
  #hostBytes = 0;
  #hostReserved = 0;
  /** @type {Set<string>} */
  #hostSeen = new Set();
  /** @type {CompactExpertCache | null} */
  #compact = null;
  #workingBytes = 0;
  #staging = 0;
  /** @type {Uint8Array[]} */
  #readerStaging = [];
  #closed = false;
  #shutdown = new AbortController();
  /** @type {Promise<void>} */
  #mutation = Promise.resolve();
  /** @type {Promise<import('./routed-expert-cache.js').RoutedExpertWeights> | null} */
  #active = null;
  /** @type {Record<string, import('./routed-expert-cache.js').RoutedExpertPhaseProfile>} */
  #phaseProfiles = {};
  /** @type {import('./routed-expert-cache.js').RoutedExpertPhaseProfile | null} */
  #activeProfile = null;
  #stats = createRoutedExpertCounters();

  /** @param {import('./routed-expert-cache.js').RoutedExpertCacheOptions} options */
  constructor(options) {
    validateQwen35RoutedConfig(options.policy);
    this.#options = options;
    const lease = options.rangeLease;
    if (lease && (!Number.isSafeInteger(lease.concurrency ?? 1) || (lease.concurrency ?? 1) < 1
      || !Number.isSafeInteger(lease.maxPartBytes) || lease.maxPartBytes < 4
      || !Number.isSafeInteger(lease.memoryBytes) || lease.memoryBytes < lease.maxPartBytes * (lease.concurrency ?? 1)
      || lease.memoryBytes > options.policy.stagingBytes - 8 * options.policy.readConcurrency)) {
      throw new Error('Owned expert ranges must fit the aggregate staging budget with fallback scratch.');
    }
    this.#staging = lease?.memoryBytes ?? 0;
    if (options.policy.evictionPolicy === 'lfu') {
      if (typeof options.numLayers !== 'number' || typeof options.numExperts !== 'number'
        || !Number.isSafeInteger(options.numLayers) || options.numLayers < 1
        || !Number.isSafeInteger(options.numExperts) || options.numExperts < 1
        || !Number.isSafeInteger(options.numLayers * options.numExperts)) {
        throw new Error('LFU requires declared positive safe layer and expert counts.');
      }
      this.#numLayers = options.numLayers;
      this.#numExperts = options.numExperts;
      this.#frequency = new ExpertFrequencyIndex(options.numLayers * options.numExperts);
    }
  }

  #sampleProfileMemory() {
    const profile = this.#activeProfile;
    if (!profile) return;
    profile.peakGpuResidentBytes = Math.max(profile.peakGpuResidentBytes, this.#stats.residentBytes);
    profile.peakHostOwnedOrReservedBytes = Math.max(profile.peakHostOwnedOrReservedBytes, this.#hostBytes + this.#hostReserved);
    profile.peakStagingBytes = Math.max(profile.peakStagingBytes, this.#staging);
  }

  #clearReaderStaging() {
    for (const bytes of this.#readerStaging) this.#staging -= bytes?.byteLength ?? 0;
    this.#readerStaging = [];
  }

  /** @param {number} reader @param {number} length */
  #getReaderStaging(reader, length) {
    let bytes = this.#readerStaging[reader];
    if (!bytes || bytes.byteLength < length) {
      const previousBytes = bytes?.byteLength ?? 0;
      bytes = new Uint8Array(length);
      this.#staging -= previousBytes;
      this.#readerStaging[reader] = bytes;
      this.#staging += bytes.byteLength;
      this.#stats.stagingAllocations++;
    } else this.#stats.stagingReuses++;
    return bytes.subarray(0, length);
  }

  /** @param {string} key */
  #lookupHost(key) {
    if (!this.#activeProfile) return this.#hostEntries.get(key);
    const start = performance.now();
    const entry = this.#hostEntries.get(key);
    this.#activeProfile.hostLookupMs += performance.now() - start;
    return entry;
  }

  /** @template T @param {() => Promise<T>} action @returns {Promise<T>} */
  #locked(action) {
    const result = this.#mutation.then(action);
    this.#mutation = result.then(() => undefined, () => undefined);
    return result;
  }

  /** @param {number} bytes @param {string} label */
  #allocate(bytes, label) {
    const { device, policy } = this.#options;
    if (bytes > device.limits.maxBufferSize || bytes > device.limits.maxStorageBufferBindingSize) {
      throw new Error(`Routed expert buffer ${label} exceeds the owning device's binding limit.`);
    }
    if (this.#options.otherGpuBytes() + this.#stats.residentBytes + bytes > policy.gpuBudgetBytes - policy.reserveBytes) {
      throw new Error('Routed expert allocation would consume the configured activation/KV reserve.');
    }
    const buffer = device.createBuffer({ size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label });
    this.#owned.add(buffer);
    this.#stats.residentBytes += bytes;
    this.#sampleProfileMemory();
    this.#stats.peakResidentBytes = Math.max(this.#stats.peakResidentBytes, this.#stats.residentBytes);
    this.#stats.peakAccountedGpuBytes = Math.max(this.#stats.peakAccountedGpuBytes,
      this.#options.otherGpuBytes() + this.#stats.residentBytes);
    return buffer;
  }

  /** @param {GPUBuffer} buffer */
  #destroyBuffer(buffer) {
    buffer.destroy();
    if (this.#owned.delete(buffer)) this.#stats.residentBytes -= buffer.size;
  }

  // A fence covers only submissions issued before that call. Never mark later
  // copies complete when an earlier fence resolves; entries may still be in use.
  async #waitForSubmitted() {
    const serial = this.#submittedSerial;
    const profile = this.#activeProfile;
    const start = profile ? performance.now() : 0;
    try { await this.#options.device.queue.onSubmittedWorkDone(); }
    finally { if (profile) profile.synchronizationWaitMs += performance.now() - start; }
    this.#completedSerial = Math.max(this.#completedSerial, serial);
  }

  /** @param {number} bytes @param {number} cacheAddition @param {AbortSignal} signal
   * @param {boolean} reusePayload @returns {Promise<GPUBuffer | null>} */
  async #makeRoom(bytes, cacheAddition, signal, reusePayload = false) {
    const { policy } = this.#options;
    /** @type {GPUBuffer | null} */
    let replacement = null;
    try {
      while (this.#cacheBytes + cacheAddition > policy.cacheBytes
        || (policy.compactPrefillBytes > 0 && this.#stats.residentBytes
          - (this.#compact?.residentBytes ?? 0) + bytes > policy.compactPrefillBytes)
        || this.#options.otherGpuBytes() + this.#stats.residentBytes + bytes > policy.gpuBudgetBytes - policy.reserveBytes) {
        /** @type {[string, import('./routed-expert-cache.js').RoutedExpertEntry] | undefined} */
        let victim;
        // LRU order already lives in Map: do not allocate the entire cache on
        // every eviction just to inspect its first unpinned entry.
        if (this.#frequency) {
          const key = this.#frequency.oldestUnpinned(key => this.#entries.get(key)?.users !== 0);
          if (key !== null) {
            const entry = this.#entries.get(key);
            if (!entry) throw new Error('LFU resident index disagrees with cache ownership.');
            victim = [key, entry];
          }
        } else {
          for (const candidate of this.#entries) {
            if (candidate[1].users === 0) { victim = candidate; break; }
          }
        }
        if (!victim) throw new Error('Routed expert budget cannot accommodate the working set and concurrent readers.');
        // Pins are changed under the mutation lock. A victim cannot acquire a new
        // copy lease while this fence settles its previously submitted work.
        if (victim[1].lastUseSerial > this.#completedSerial) {
          this.#stats.evictionFenceWaits++;
          await this.#waitForSubmitted();
        } else this.#stats.evictionFenceSkips++;
        signal.throwIfAborted();
        // Transfer only an exact-size, unpinned, completed payload. It remains
        // owned and resident; no spare pool or extra budget is introduced.
        if (reusePayload && replacement === null && victim[1].bytes === bytes) {
          replacement = victim[1].buffer;
          bytes = 0;
        } else this.#destroyBuffer(victim[1].buffer);
        this.#cacheBytes -= victim[1].bytes;
        this.#entries.delete(victim[0]);
        this.#frequency?.remove(victim[0]);
        this.#stats.evictions++;
      }
      return replacement;
    } catch (error) {
      if (replacement) this.#destroyBuffer(replacement);
      throw error;
    }
  }

  /** @param {number} bytes */
  #makeHostRoom(bytes) {
    while (this.#hostBytes + this.#hostReserved + bytes > this.#options.policy.hostCacheBytes) {
      const oldest = this.#hostEntries.entries().next().value;
      if (!oldest) throw new Error('Routed host cache cannot hold the concurrent expert payloads.');
      this.#hostEntries.delete(oldest[0]); this.#hostBytes -= oldest[1].bytes;
      this.#stats.hostEvictions++;
    }
  }

  /** @param {import('./routed-expert-cache.js').RoutedExpertEntry} entry
   * @param {number} offset @param {Uint8Array} bytes @param {() => void} beforeSubmit */
  #upload(entry, offset, bytes, beforeSubmit) {
    beforeSubmit();
    const started = performance.now();
    entry.lastUseSerial = ++this.#submittedSerial;
    this.#options.device.queue.writeBuffer(entry.buffer, offset, bytes);
    this.#stats.uploadEnqueueMs += performance.now() - started;
    this.#stats.uploadBytes += bytes.byteLength;
  }

  /** @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan @param {AbortSignal} signal */
  async #prepareWorking(plan, signal) {
    const signature = JSON.stringify(plan.planes.map(p => [p.role, p.dtype, p.location.shape, p.bytes]));
    const { policy } = this.#options;
    const payloadBytes = policy.prefillUploadMode === 'direct-bank' ? 0 : plan.expertBytes * policy.readConcurrency;
    let spareCapacity = Math.max(0, policy.gpuBudgetBytes - policy.reserveBytes
      - this.#options.otherGpuBytes() - plan.bankBytes - payloadBytes);
    if (this.#working.size && (signature !== this.#signature
      || this.#workingBytes - plan.bankBytes > spareCapacity)) {
      await this.#waitForSubmitted();
      signal.throwIfAborted();
      // Capacity is owned storage, not tensor geometry. Preserve payload-reader
      // headroom; trim surplus if the current budget no longer accommodates it.
      for (const [role, buffer] of this.#working) {
        const next = plan.planes.find(plane => plane.role === role);
        if (next && buffer.size >= next.bytes && buffer.size - next.bytes <= spareCapacity) {
          spareCapacity -= buffer.size - next.bytes;
          this.#stats.workingBufferReuses++;
          this.#stats.workingBufferReusedBytes += buffer.size;
          continue;
        }
        this.#destroyBuffer(buffer);
        this.#working.delete(role);
        this.#workingBytes -= buffer.size;
      }
      this.#loaded.clear();
    }
    const missing = plan.planes.filter(plane => !this.#working.has(plane.role));
    if (missing.length) {
      await this.#makeRoom(missing.reduce((bytes, plane) => bytes + plane.bytes, 0), 0, signal);
      signal.throwIfAborted();
      for (const plane of missing) {
        const buffer = this.#allocate(plane.bytes, `routed_working_${plane.role}`);
        this.#working.set(plane.role, buffer);
        this.#workingBytes += plane.bytes;
        this.#stats.workingBufferAllocations++;
        this.#stats.workingBufferAllocatedBytes += plane.bytes;
      }
    }
    this.#signature = signature;
    if (this.#layer !== plan.layer) {
      this.#loaded.clear();
      this.#layer = plan.layer;
    }
    await this.#makeRoom(0, 0, signal);
    this.#makeHostRoom(0);
    if (this.#options.policy.hostCacheBytes === 0) this.#hostSeen.clear();
  }

  /** @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan
   * @param {number} expert @param {AbortSignal} signal @param {() => void} beforeSubmit
   * @param {'prefill' | 'decode' | undefined} phase @param {number} reader */
  async #get(plan, expert, signal, beforeSubmit, phase, reader) {
    const key = `${plan.layer}:${expert}`;
    const cached = await this.#locked(async () => {
      signal.throwIfAborted();
      const entry = this.#entries.get(key);
      if (entry) {
        this.#touchEntry(key, entry);
        entry.users++;
        this.#stats.cacheHits++;
      }
      return entry;
    });
    if (cached) return cached;
    this.#stats.cacheMisses++;
    this.#frequency?.observe(key);
    const entry = await this.#locked(async () => {
      const replacement = await this.#makeRoom(plan.expertBytes, plan.expertBytes, signal, true);
      if (signal.aborted) {
        if (replacement) this.#destroyBuffer(replacement);
        signal.throwIfAborted();
      }
      const buffer = replacement ?? this.#allocate(plan.expertBytes, `routed_expert_${key}`);
      if (replacement) this.#stats.bufferReuses++;
      this.#cacheBytes += plan.expertBytes;
      return { buffer, bytes: plan.expertBytes, users: 1, lastUseSerial: 0 };
    });
    // Keep read granularity unchanged; direct fill owns only the assembled output.
    const readRangeInto = this.#options.readRangeInto;
    const chunkBytes = Math.floor((this.#options.policy.stagingBytes - (this.#options.rangeLease?.memoryBytes ?? 0)) / (2 * this.#options.policy.readConcurrency * 4)) * 4;
    let targetOffset = 0;
    let hostReservation = 0;
    try {
      const host = this.#lookupHost(key);
      if (host) {
        this.#hostEntries.delete(key); this.#hostEntries.set(key, host);
        this.#stats.hostCacheHits++;
        // writeBuffer copies each immutable payload before the next await;
        // queued GPU work never borrows the host cache's backing bytes.
        for (const chunk of host.chunks) {
          signal.throwIfAborted();
          this.#upload(entry, chunk.offset, chunk.data, beforeSubmit);
        }
        await this.#locked(async () => { this.#entries.set(key, entry); this.#frequency?.admit(key); });
        return entry;
      }
      /** @type {Array<{offset: number, data: Uint8Array}> | null} */
      let chunks = null;
      if (this.#options.policy.hostCacheBytes > 0) {
        this.#stats.hostCacheMisses++;
        const repeatOnly = this.#options.policy.hostCacheAdmission === 'repeat-read';
        const phaseAllowed = this.#options.policy.hostCacheAdmissionPhase === 'all' || phase === 'decode';
        const admit = phaseAllowed && (!repeatOnly || this.#hostSeen.has(key));
        if (phaseAllowed && repeatOnly) this.#hostSeen.add(key);
        if (!phaseAllowed) this.#stats.hostPrefillAdmissionSkips++;
        if (admit) {
          this.#makeHostRoom(plan.expertBytes);
          this.#hostReserved += plan.expertBytes; hostReservation = plan.expertBytes;
          this.#sampleProfileMemory();
          this.#stats.peakHostOwnedOrReservedBytes = Math.max(this.#stats.peakHostOwnedOrReservedBytes,
            this.#hostBytes + this.#hostReserved);
          chunks = [];
        } else this.#stats.hostAdmissionSkips++;
      }
      if (this.#options.policy.sourceReadMode === 'expert-parallel') {
        const destination = this.#getReaderStaging(reader, plan.expertBytes);
        this.#sampleProfileMemory();
        this.#stats.peakStagingBytes = Math.max(this.#stats.peakStagingBytes, this.#staging);
        const started = performance.now();
        try {
          await readRoutedExpertPayload(this.#options, this.#stats, plan, expert, signal, destination, null);
          signal.throwIfAborted();
          this.#upload(entry, 0, destination, beforeSubmit);
        } finally { this.#stats.ioWaitMs += performance.now() - started; }
      } else {
        for (const plane of plan.planes) {
          for (let offset = 0; offset < plane.expertBytes;) {
            signal.throwIfAborted();
            const length = Math.min(chunkBytes, plane.expertBytes - offset);
            // Each reader owns its destination until writeBuffer has copied it.
            // Retained host chunks must keep independent immutable backing storage.
            const destination = readRangeInto && chunks === null && plane.location.sourceTransform == null
              ? this.#getReaderStaging(reader, length) : null;
            const ownedStagingBytes = destination ? 0 : (readRangeInto ? 1 : 2) * length;
            this.#staging += ownedStagingBytes;
            this.#sampleProfileMemory();
            this.#stats.peakStagingBytes = Math.max(this.#stats.peakStagingBytes, this.#staging);
            const started = performance.now();
            try {
              const bytes = await readRoutedExpertRange(this.#options, this.#stats, plane, expert, offset, length, signal, destination);
              signal.throwIfAborted();
              this.#upload(entry, targetOffset + offset, bytes, beforeSubmit);
              chunks?.push({ offset: targetOffset + offset, data: bytes });
            } finally {
              this.#stats.ioWaitMs += performance.now() - started;
              this.#staging -= ownedStagingBytes;
            }
            offset += length;
          }
          targetOffset += plane.expertBytes;
        }
      }
      if (chunks) {
        signal.throwIfAborted();
        this.#hostReserved -= hostReservation; hostReservation = 0;
        this.#hostEntries.set(key, { bytes: plan.expertBytes, chunks });
        this.#hostBytes += plan.expertBytes;
        if (phase === 'prefill') this.#stats.hostPrefillAdmissions++;
        if (phase === 'decode') this.#stats.hostDecodeAdmissions++;
      }
      await this.#locked(async () => { this.#entries.set(key, entry); this.#frequency?.admit(key); });
      return entry;
    } catch (error) {
      try {
        await this.#waitForSubmitted();
        await this.#locked(async () => { this.#destroyBuffer(entry.buffer); this.#cacheBytes -= entry.bytes; });
      } catch (cleanup) { throw new AggregateError([error, cleanup], 'Routed expert load and cleanup failed.'); }
      throw error;
    } finally {
      this.#hostReserved -= hostReservation;
    }
  }

  /** @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan
   * @param {number} expert @param {AbortSignal} signal
   * @param {() => void} beforeSubmit @param {number} reader */
  async #writeWorking(plan, expert, signal, beforeSubmit, reader) {
    const allLeased = this.#options.rangeLease && plan.planes.every(plane =>
      plane.location.sourceTransform == null && (!plane.location.spans || plane.location.spans.length <= 1));
    if (plan.planes.some(plane => plane.location.sourceTransform != null)
      || (!allLeased && !this.#options.readRangeInto)) {
      throw new Error('Direct prefill requires unchanged planes and an owned lease or direct range reader.');
    }
    const chunkBytes = Math.floor((this.#options.policy.stagingBytes - (this.#options.rangeLease?.memoryBytes ?? 0))
      / (2 * this.#options.policy.readConcurrency * 4)) * 4;
    const length = this.#options.policy.sourceReadMode === 'expert-parallel' ? plan.expertBytes : chunkBytes;
    const scratch = allLeased ? null : this.#getReaderStaging(reader, length);
    this.#stats.peakStagingBytes = Math.max(this.#stats.peakStagingBytes, this.#staging);
    this.#sampleProfileMemory();
    this.#stats.cacheMisses++;
    await writeRoutedPrefillExpert(this.#options, this.#stats, plan, expert, signal, this.#working, scratch,
      () => { beforeSubmit(); ++this.#submittedSerial; });
    this.#loaded.add(expert);
  }

  /** @param {string} key @param {import('./routed-expert-cache.js').RoutedExpertEntry} entry */
  #touchEntry(key, entry) {
    this.#frequency?.observe(key);
    this.#entries.delete(key); this.#entries.set(key, entry);
    const host = this.#lookupHost(key);
    if (host) { this.#hostEntries.delete(key); this.#hostEntries.set(key, host); }
  }

  /** @param {GPUCommandEncoder} encoder
   * @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan
   * @param {number} expert @param {import('./routed-expert-cache.js').RoutedExpertEntry} entry */
  #encodeExpertCopy(encoder, plan, expert, entry) {
    let offset = 0;
    for (const plane of plan.planes) {
      const destination = this.#working.get(plane.role);
      if (!destination) throw new Error('Routed working bank is incomplete.');
      encoder.copyBufferToBuffer(entry.buffer, offset, destination, expert * plane.expertBytes, plane.expertBytes);
      offset += plane.expertBytes;
    }
  }

  /** @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan
   * @param {number[]} ids @param {AbortSignal} signal @param {() => void} beforeSubmit */
  #copyCachedHits(plan, ids, signal, beforeSubmit) {
    return this.#locked(async () => {
      signal.throwIfAborted();
      const compact = this.#options.policy.compactPrefillReuse === 'resident-gpu' ? this.#compact : null;
      const residentSlots = compact?.getResidentSlots(plan.layer,
        ids.filter(expert => !this.#loaded.has(expert))) ?? new Map();
      const hits = ids.flatMap(expert => {
        const key = `${plan.layer}:${expert}`;
        const entry = this.#entries.get(key);
        return entry && !this.#loaded.has(expert) && !residentSlots.has(expert) ? [{ expert, key, entry }] : [];
      });
      const copied = new Set();
      if (!hits.length && !residentSlots.size) return copied;
      // No miss starts until this batch is submitted. Pins protect all encoded
      // sources, without requiring capacity for a whole layer's missing experts.
      for (const { entry } of hits) entry.users++;
      try {
        const encoder = this.#options.device.createCommandEncoder({ label: 'routed_expert_hit_batch' });
        if (compact && residentSlots.size) compact.encodeResidentCopies(plan, residentSlots, this.#working, encoder);
        for (const { expert, entry } of hits) this.#encodeExpertCopy(encoder, plan, expert, entry);
        const commands = encoder.finish();
        signal.throwIfAborted();
        beforeSubmit();
        const serial = ++this.#submittedSerial;
        for (const { entry } of hits) entry.lastUseSerial = serial;
        this.#options.device.queue.submit([commands]);
        this.#stats.expertCopySubmissions++;
        this.#stats.hitBatchSubmissions++;
        this.#stats.batchedHitExperts += hits.length;
        if (residentSlots.size) {
          this.#stats.prefillGpuReuseBytes += residentSlots.size * plan.expertBytes;
          this.#stats.prefillGpuReuseExperts += residentSlots.size;
          this.#stats.prefillGpuReuseSubmissions++;
          this.#stats.cacheHits += residentSlots.size;
          for (const expert of residentSlots.keys()) {
            this.#loaded.add(expert);
            copied.add(expert);
          }
        }
        for (const { expert, key, entry } of hits) {
          this.#touchEntry(key, entry);
          this.#stats.cacheHits++;
          this.#loaded.add(expert); copied.add(expert);
        }
        return copied;
      } finally {
        for (const { entry } of hits) entry.users--;
      }
    });
  }

  /** @type {import('./routed-expert-cache.js').RoutedExpertCache['prepare']} */
  prepare(plan, expertIds, externalSignal, phase) {
    if ((phase !== undefined && phase !== 'prefill' && phase !== 'decode')
        || (this.#options.policy.hostCacheAdmissionPhase === 'decode' && phase === undefined)) {
      throw new Error('Routed expert phase-aware host admission requires explicit prefill or decode phase.');
    }
    if (this.#closed || this.#active) throw new Error('Routed expert cache is closed or already preparing a layer.');
    const controller = new AbortController();
    const abortExternal = () => controller.abort(externalSignal?.reason);
    const abortShutdown = () => controller.abort(this.#shutdown.signal.reason);
    externalSignal?.addEventListener('abort', abortExternal, { once: true });
    this.#shutdown.signal.addEventListener('abort', abortShutdown, { once: true });
    if (externalSignal?.aborted) abortExternal();
    const profileEnabled = getRuntimeConfig().shared.debug.profiler.enabled;
    const before = profileEnabled ? { ...this.#stats } : null;
    const started = profileEnabled ? performance.now() : 0;
    if (profileEnabled) {
      this.#activeProfile = this.#phaseProfiles[phase ?? 'unspecified'] ??= {
        preparations: 0, failures: 0, prepareWallMs: 0, hostLookupMs: 0, synchronizationWaitMs: 0,
        peakGpuResidentBytes: 0, peakHostOwnedOrReservedBytes: 0, peakStagingBytes: 0, counters: {},
      };
      this.#activeProfile.preparations++;
      this.#sampleProfileMemory();
    }
    let succeeded = false;
    const task = this.#prepare(plan, expertIds, controller, phase).then(result => { succeeded = true; return result; });
    this.#active = task;
    return task.finally(() => {
      const profile = this.#activeProfile;
      if (profile && before) {
        profile.prepareWallMs += performance.now() - started;
        if (!succeeded) profile.failures++;
        const current = /** @type {Record<string, number>} */ (this.#stats);
        for (const [key, value] of Object.entries(before)) {
          // Peaks and gauges are not additive event counters.
          if (key.startsWith('peak') || key === 'residentBytes') continue;
          profile.counters[key] = (profile.counters[key] ?? 0) + current[key] - value;
        }
      }
      this.#activeProfile = null;
      externalSignal?.removeEventListener('abort', abortExternal);
      this.#shutdown.signal.removeEventListener('abort', abortShutdown);
      this.#active = null;
    });
  }

  /** @param {import('./routed-expert-plan.js').RoutedExpertBankPlan} plan
   * @param {number[]} expertIds @param {AbortController} controller
   * @param {'prefill' | 'decode' | undefined} phase
   * @returns {Promise<import('./routed-expert-cache.js').RoutedExpertWeights>} */
  async #prepare(plan, expertIds, controller, phase) {
    const signal = controller.signal;
    signal.throwIfAborted();
    if ((this.#options.policy.evictionPolicy === 'lfu') !== (this.#frequency !== null)) {
      throw new Error('Changing routed eviction policy requires recreating the cache.');
    }
    if (this.#frequency && (!Number.isSafeInteger(plan.layer) || plan.layer < 0
      || plan.layer >= this.#numLayers || plan.numExperts !== this.#numExperts)) {
      throw new Error('LFU plan exceeds the declared model layer/expert bounds.');
    }
    const ids = [...new Set(expertIds)];
    if (!ids.length || ids.some(id => !Number.isSafeInteger(id) || id < 0 || id >= plan.numExperts)) {
      throw new Error('Routed expert request contains invalid or absent selected expert IDs.');
    }
    if (plan.expertBytes * this.#options.policy.readConcurrency > this.#options.policy.cacheBytes) {
      throw new Error('Routed expert cache must hold one payload per concurrent reader.');
    }
    if (this.#options.policy.hostCacheBytes > 0
        && plan.expertBytes * this.#options.policy.readConcurrency > this.#options.policy.hostCacheBytes) {
      throw new Error('Routed host cache must hold one payload per concurrent reader.');
    }
    assertExpertReadShape(this.#options, plan);
    // Policy changes cannot leave old scratch charged alongside a different path.
    const scratchLimit = Math.floor((this.#options.policy.stagingBytes - (this.#options.rangeLease?.memoryBytes ?? 0)) / (2 * this.#options.policy.readConcurrency * 4)) * 4;
    if (!this.#options.readRangeInto
      || this.#readerStaging.length > this.#options.policy.readConcurrency
      || this.#readerStaging.some(bytes => bytes.byteLength > scratchLimit)) this.#clearReaderStaging();
    if (this.#options.policy.storageMode === 'compact-decode') {
      if (phase !== 'prefill' && phase !== 'decode') throw new Error('Compact experts require an explicit execution phase.');
      if (phase === 'decode') {
        this.#clearReaderStaging();
        await this.#waitForSubmitted();
        signal.throwIfAborted();
        if (this.#owned.size) {
          for (const buffer of this.#owned) this.#destroyBuffer(buffer);
          this.#entries.clear(); this.#working.clear(); this.#loaded.clear();
          this.#cacheBytes = 0; this.#workingBytes = 0; this.#signature = ''; this.#layer = -1;
        }
        if (!this.#compact) {
          const compactLayerCount = this.#options.compactLayerCount;
          if (!Number.isSafeInteger(compactLayerCount) || !compactLayerCount) throw new Error('Compact experts require declared layer count.');
          this.#compact = new CompactExpertCache({ ...this.#options, compactLayerCount }, this.#stats);
        }
        const weights = await this.#compact.prepare(plan, ids, signal);
        this.#sampleProfileMemory();
        return weights;
      }
      this.#compact?.clearStaging();
      if (this.#options.policy.compactPrefillBytes > 0) {
        const payloadBytes = this.#options.policy.prefillUploadMode === 'direct-bank' ? 0 : plan.expertBytes * this.#options.policy.readConcurrency;
        if (plan.bankBytes + payloadBytes > this.#options.policy.compactPrefillBytes) {
          throw new Error('Compact prefill reservation cannot hold the working bank and concurrent expert payloads.');
        }
      } else if (this.#compact) { await this.#compact.destroy(); this.#compact = null; }
    }
    await this.#locked(() => this.#prepareWorking(plan, signal));
    let cursor = 0;
    let readyAt = 0, submitted = false;
    // Only the first proven queue-idle gap is counted. Uploads and later gaps
    // are excluded; this is a lower bound, not global hardware utilization.
    const fence = this.#waitForSubmitted().then(() => { if (!submitted) readyAt = performance.now(); });
    fence.catch(() => undefined);
    const beforeSubmit = () => {
      if (!submitted && readyAt) this.#stats.initialQueueIdleForWeightsMs += performance.now() - readyAt;
      submitted = true;
    };
    const hitBatch = this.#copyCachedHits(plan, ids, signal, beforeSubmit);
    /** @param {number} reader */
    const work = async (reader) => {
      const copiedHits = await hitBatch;
      while (cursor < ids.length) {
        signal.throwIfAborted();
        const expert = ids[cursor++];
        if (copiedHits.has(expert)) continue;
        if (this.#loaded.has(expert)) {
          this.#frequency?.observe(`${plan.layer}:${expert}`);
          this.#stats.workingHits++;
          continue;
        }
        if (phase === 'prefill' && this.#options.policy.prefillUploadMode === 'direct-bank') {
          await this.#writeWorking(plan, expert, signal, beforeSubmit, reader);
          continue;
        }
        const entry = await this.#get(plan, expert, signal, beforeSubmit, phase, reader);
        try {
          signal.throwIfAborted();
          const encoder = this.#options.device.createCommandEncoder({ label: 'routed_expert_copy' });
          this.#encodeExpertCopy(encoder, plan, expert, entry);
          beforeSubmit();
          entry.lastUseSerial = ++this.#submittedSerial;
          this.#options.device.queue.submit([encoder.finish()]);
          this.#stats.expertCopySubmissions++;
          this.#loaded.add(expert);
        } finally { entry.users--; }
      }
    };
    const laneCount = this.#options.rangeLease && phase === 'prefill' && this.#options.policy.prefillUploadMode === 'direct-bank'
      ? Math.min(this.#options.policy.readConcurrency, this.#options.rangeLease.concurrency ?? 1) : this.#options.policy.readConcurrency;
    const outcomes = await Promise.allSettled(Array.from({ length: laneCount }, async (_, reader) => {
      try { await work(reader); } catch (error) { controller.abort(error); throw error; }
    }));
    const errors = [...new Set(outcomes.filter(r => r.status === 'rejected').map(r => r.reason))];
    try { await fence; } catch (error) { errors.push(error); }
    if (errors.length || signal.aborted) {
      this.#loaded.clear();
      try { await this.#waitForSubmitted(); } catch (error) { errors.push(error); }
      if (errors.length > 1) throw new AggregateError(errors, 'Routed expert preparation failed.');
      if (errors.length) throw errors[0];
      signal.throwIfAborted();
    }
    const weights = Object.fromEntries(plan.planes.map(plane => {
      const buffer = this.#working.get(plane.role);
      if (!buffer) throw new Error('Routed working bank is incomplete.');
      return [plane.role, createWeightBuffer(buffer, plane.dtype, 'row', plane.location.shape, plane.name)];
    }));
    if (!weights.down) throw new Error('Routed working bank has no down projection.');
    if (phase === 'prefill' && this.#options.policy.prefillCompactAdmission === 'selected-frequency-first'
      && !this.#compact?.hasLayer(plan.layer)) {
      const compactLayerCount = this.#options.compactLayerCount;
      if (!Number.isSafeInteger(compactLayerCount) || !compactLayerCount) throw new Error('Prefill seed requires assigned layer count.');
      const options = { ...this.#options, compactLayerCount };
      if (!this.#compact) this.#compact = new CompactExpertCache(options, this.#stats);
      const selected = selectPrefillSeedExperts(expertIds, compactExpertCapacity(plan, options));
      // GPU copies use already authenticated bytes and preserve original expert IDs.
      this.#clearReaderStaging();
      await this.#compact.prepare(plan, selected, signal, this.#working);
    }
    return { expertIdx: 0, expertFormat: 'qwen3_5', numExperts: plan.numExperts,
      expertIntermediateSize: plan.intermediateSize, ...weights, down: weights.down };
  }

  getStats() {
    this.#stats.peakAccountedGpuBytes = Math.max(this.#stats.peakAccountedGpuBytes,
      this.#options.otherGpuBytes() + this.#stats.residentBytes);
    const compact = this.#compact?.getStats();
    return { ...this.#stats,
      evictionPolicy: this.#options.policy.evictionPolicy,
      frequencyHistoryEntries: this.#frequency?.historySize ?? 0,
      frequencyResidentEntries: this.#frequency?.residentSize ?? 0,
      phaseProfiles: Object.fromEntries(Object.entries(this.#phaseProfiles).map(([phase, profile]) =>
        [phase, { ...profile, counters: { ...profile.counters } }])),
      experts: this.#entries.size,
      hostCacheBytes: this.#hostBytes, hostReservedBytes: this.#hostReserved, hostExperts: this.#hostEntries.size,
      ...compact,
      cacheBytes: this.#cacheBytes + (compact?.cacheBytes ?? 0),
      workingBytes: this.#workingBytes };
  }

  async destroy() {
    this.#closed = true;
    this.#shutdown.abort(new DOMException('Routed expert cache unloaded.', 'AbortError'));
    await this.#active?.catch(() => undefined);
    this.#clearReaderStaging();
    this.#hostEntries.clear(); this.#hostSeen.clear(); this.#hostBytes = 0; this.#hostReserved = 0;
    const errors = [];
    if (this.#compact) {
      try { await this.#compact.destroy(); this.#compact = null; } catch (error) { errors.push(error); }
    }
    if (this.#owned.size) {
      try { await this.#waitForSubmitted(); } catch (error) { errors.push(error); }
    }
    for (const buffer of this.#owned) {
      try { this.#destroyBuffer(buffer); } catch (error) { errors.push(error); }
    }
    if (!this.#owned.size) {
      this.#entries.clear(); this.#frequency?.clear(); this.#working.clear(); this.#loaded.clear();
      this.#cacheBytes = 0; this.#workingBytes = 0;
    }
    if (errors.length) throw new AggregateError(errors, 'Routed expert cache cleanup failed.');
  }
}
