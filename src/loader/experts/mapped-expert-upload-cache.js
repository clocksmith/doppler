import { isDeviceLost, observeDeviceLoss } from '../../memory/device-residency.js';

// Bytes supplied by the caller must already have passed its source-identity and
// authentication checks. This cache owns upload lifetime, never authentication.
export class MappedExpertUploadCache {
  /** @type {import('./mapped-expert-upload-cache.js').MappedExpertUploadCacheOptions} */
  #options;
  /** @type {Map<string, import('./mapped-expert-upload-cache.js').MappedExpertEntry>} */
  #entries = new Map();
  /** @type {Set<import('./mapped-expert-upload-cache.js').MappedExpertEntry>} */
  #owned = new Set();
  /** @type {Map<string, import('./mapped-expert-upload-cache.js').MappedExpertEntry>} */
  #ownedKeys = new Map();
  /** @type {Map<string, import('./mapped-expert-upload-cache.js').MappedExpertEntry>} */
  #probation = new Map();
  /** @type {Map<string, import('./mapped-expert-upload-cache.js').MappedExpertEntry>} */
  #protected = new Map();
  /** @type {Map<string, true>} */
  #ghost = new Map();
  #pinnedBytes = 0;
  #publishedBytes = 0;
  #publishedPinnedBytes = 0;
  #pendingMaps = 0;
  #protectedBytes = 0;
  #allocations = 0;
  #reuseAttempts = 0;
  #reuses = 0;
  #admissionSkips = 0;
  #reuseHits = 0;
  /** @type {Set<Promise<boolean>>} */
  #writers = new Set();
  /** @type {Set<import('./mapped-expert-upload-cache.js').MappedExpertUse>} */
  #uses = new Set();
  #shutdown = new AbortController();
  #closing = false;
  /** @type {Promise<void> | null} */
  #closeTask = null;
  #allocatedBytes = 0;
  #mappedBytes = 0;
  #peakBytes = 0;
  #peakMappedBytes = 0;
  #hits = 0;
  #misses = 0;
  #admissions = 0;
  #evictions = 0;
  #rejections = 0;
  #copySubmissions = 0;
  #copyBytes = 0;

  /** @param {import('./mapped-expert-upload-cache.js').MappedExpertUploadCacheOptions} options */
  constructor(options) {
    const { device, maxBytes, maxMappedBytes, maxEntries } = options;
    if (!device || ![maxBytes, maxMappedBytes].every(value => Number.isSafeInteger(value) && value > 0 && value % 4 === 0)
      || maxMappedBytes > maxBytes || !Number.isSafeInteger(maxEntries) || maxEntries < 1) {
      throw new Error('Mapped expert cache requires an owning device and explicit aligned allocation/mapped/entry bounds.');
    }
    const admission = options.admission ?? 'all';
    const protectedShare = options.protectedShare ?? 0;
    const maxGhostEntries = options.maxGhostEntries ?? maxEntries;
    if (!['all', 'repeat-read'].includes(admission) || !Number.isFinite(protectedShare) || protectedShare < 0 || protectedShare > 1
      || !Number.isSafeInteger(maxGhostEntries) || maxGhostEntries < 1) {
      throw new Error('Mapped expert admission/protection/ghost bounds are invalid.');
    }
    this.#options = { ...options, admission, protectedShare, maxGhostEntries };
    observeDeviceLoss(device);
  }

  #assertOpen() {
    if (this.#closing || isDeviceLost(this.#options.device)) throw new Error('Mapped expert cache is closing or its device was lost.');
  }

  /** @param {string} key */
  #assertKey(key) {
    if (typeof key !== 'string' || !key.length) throw new Error('Mapped expert cache requires a caller-bound identity key.');
  }

  /** @param {import('./mapped-expert-upload-cache.js').MappedExpertEntry} entry */
  #unmap(entry) {
    if (!entry.mapped) return;
    entry.buffer.unmap();
    entry.mapped = false;
    this.#mappedBytes -= entry.bytes;
  }

  /** @param {import('./mapped-expert-upload-cache.js').MappedExpertEntry} entry */
  #forget(entry) {
    if (this.#entries.get(entry.key) === entry) {
      this.#entries.delete(entry.key); this.#publishedBytes -= entry.bytes;
      if (entry.pinned) this.#publishedPinnedBytes -= entry.bytes;
    }
    this.#probation.delete(entry.key);
    if (this.#protected.delete(entry.key)) this.#protectedBytes -= entry.bytes;
    entry.protected = false;
  }

  /** @param {import('./mapped-expert-upload-cache.js').MappedExpertEntry} entry */
  #publish(entry) {
    this.#entries.set(entry.key, entry); this.#publishedBytes += entry.bytes;
    if (entry.pinned) this.#publishedPinnedBytes += entry.bytes;
    else this.#probation.set(entry.key, entry);
  }

  #protectedBudget() {
    return Math.floor((this.#options.maxBytes - this.#pinnedBytes) * (this.#options.protectedShare ?? 0) / 4) * 4;
  }

  #rebalance() {
    const budget = this.#protectedBudget();
    while (this.#protectedBytes > budget) {
      const entry = this.#protected.values().next().value;
      if (!entry) break;
      this.#protected.delete(entry.key); this.#protectedBytes -= entry.bytes;
      entry.protected = false; this.#probation.set(entry.key, entry);
    }
  }

  /** @param {import('./mapped-expert-upload-cache.js').MappedExpertEntry} entry @param {boolean} promote */
  #touch(entry, promote) {
    this.#entries.delete(entry.key); this.#entries.set(entry.key, entry);
    if (entry.pinned) return;
    if (promote && (this.#options.protectedShare ?? 0) > 0) {
      this.#probation.delete(entry.key);
      if (!entry.protected) { entry.protected = true; this.#protectedBytes += entry.bytes; }
      this.#protected.delete(entry.key); this.#protected.set(entry.key, entry);
      this.#rebalance();
    } else {
      const segment = entry.protected ? this.#protected : this.#probation;
      segment.delete(entry.key); segment.set(entry.key, entry);
    }
  }

  /** @param {string} key */
  #rememberMiss(key) {
    this.#ghost.delete(key); this.#ghost.set(key, true);
    while (this.#ghost.size > /** @type {number} */ (this.#options.maxGhostEntries)) {
      const oldest = this.#ghost.keys().next().value;
      if (oldest === undefined) break;
      this.#ghost.delete(oldest);
    }
  }

  /** Retain charges and identity if disposal throws. No submitted reader may remain.
   * @param {import('./mapped-expert-upload-cache.js').MappedExpertEntry} entry */
  #retire(entry) {
    if (entry.users || entry.mapping) throw new Error('Mapped expert cache cannot retire an active submitted copy or pending map.');
    const dispose = this.#options.dispose ?? (buffer => buffer.destroy());
    dispose(entry.buffer);
    if (entry.mapped) { entry.mapped = false; this.#mappedBytes -= entry.bytes; }
    if (this.#owned.delete(entry)) {
      this.#allocatedBytes -= entry.bytes;
      if (entry.pinned) this.#pinnedBytes -= entry.bytes;
    }
    if (this.#ownedKeys.get(entry.key) === entry) this.#ownedKeys.delete(entry.key);
    this.#forget(entry);
  }

  *#victims() {
    for (const segment of [this.#probation, this.#protected]) {
      for (const entry of segment.values()) if (!entry.users) yield entry;
    }
  }

  /** @param {number} bytes */
  #hasRoom(bytes) {
    return this.#allocatedBytes + bytes <= this.#options.maxBytes && this.#owned.size + 1 <= this.#options.maxEntries;
  }

  /** Null: allocate new storage; false: no bounded admission; entry: remap an idle victim.
   * @param {number} bytes @returns {import('./mapped-expert-upload-cache.js').MappedExpertEntry | null | false} */
  #makeRoom(bytes) {
    if (this.#hasRoom(bytes)) return null;
    let remainingBytes = this.#allocatedBytes, remainingEntries = this.#owned.size;
    const victims = [];
    for (const entry of this.#victims()) {
      if (!victims.length && entry.bytes === bytes) {
        this.#forget(entry); this.#ownedKeys.delete(entry.key); this.#evictions++;
        return entry;
      }
      victims.push(entry); remainingBytes -= entry.bytes; remainingEntries--;
      if (remainingBytes + bytes <= this.#options.maxBytes && remainingEntries + 1 <= this.#options.maxEntries) break;
    }
    if (remainingBytes + bytes > this.#options.maxBytes || remainingEntries + 1 > this.#options.maxEntries) return false;
    for (const victim of victims) { this.#retire(victim); this.#evictions++; }
    return null;
  }

  /** @type {import('./mapped-expert-upload-cache.js').MappedExpertUploadCache['admit']} */
  admit(key, bytes, fill, options = {}) {
    this.#assertOpen(); this.#assertKey(key);
    options.signal?.throwIfAborted();
    if (!Number.isSafeInteger(bytes) || bytes < 4 || bytes % 4 || typeof fill !== 'function') {
      return Promise.reject(new Error('Mapped expert admission requires an aligned complete expert and a fill callback.'));
    }
    const existing = this.#entries.get(key);
    if (existing) {
      if (existing.bytes !== bytes) return Promise.reject(new Error('Mapped expert identity changed its byte extent.'));
      if (options.pinned && !existing.pinned) {
        this.#forget(existing); existing.pinned = true; this.#pinnedBytes += existing.bytes;
        this.#publish(existing); this.#rebalance();
      }
      this.#touch(existing, false);
      return Promise.resolve(true);
    }
    if (this.#ownedKeys.has(key)) return Promise.reject(new Error('Mapped expert identity has an unfinished admission or retirement.'));
    if (bytes > this.#options.maxBytes || this.#mappedBytes + bytes > this.#options.maxMappedBytes
      || bytes > this.#options.device.limits.maxBufferSize) {
      this.#rejections++; return Promise.resolve(false);
    }
    if (!options.pinned && this.#options.admission === 'repeat-read' && !this.#hasRoom(bytes) && !this.#ghost.has(key)) {
      this.#rememberMiss(key); this.#admissionSkips++;
      return Promise.resolve(false);
    }
    const reusable = this.#makeRoom(bytes);
    if (reusable === false) { this.#rejections++; return Promise.resolve(false); }
    let entry;
    if (reusable) {
      entry = reusable;
      entry.key = key; entry.pinned = options.pinned === true; entry.mapped = true; entry.mapping = true;
      if (entry.pinned) this.#pinnedBytes += entry.bytes;
      this.#mappedBytes += entry.bytes; this.#reuseAttempts++; this.#pendingMaps++;
    } else {
      const allocate = this.#options.allocate ?? (descriptor => this.#options.device.createBuffer(descriptor));
      let buffer;
      try { buffer = allocate({ size: bytes, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true, label: 'mapped_expert_upload' }); }
      catch (error) { return Promise.reject(error); }
      entry = { key, bytes: buffer.size, buffer, mapped: true, mapping: false, pinned: options.pinned === true, protected: false, firstConsumption: true, users: 0 };
      this.#owned.add(entry); this.#allocatedBytes += entry.bytes; this.#mappedBytes += entry.bytes;
      if (entry.pinned) this.#pinnedBytes += entry.bytes;
      this.#allocations++;
    }
    this.#ownedKeys.set(key, entry);
    if (entry.pinned) this.#rebalance();
    this.#peakBytes = Math.max(this.#peakBytes, this.#allocatedBytes);
    this.#peakMappedBytes = Math.max(this.#peakMappedBytes, this.#mappedBytes);
    const task = Promise.resolve().then(() => this.#fill(entry, bytes, fill, options.signal));
    this.#writers.add(task);
    task.finally(() => this.#writers.delete(task)).catch(() => undefined);
    return task;
  }

  /** @param {import('./mapped-expert-upload-cache.js').MappedExpertEntry} entry
   * @param {number} requestedBytes
   * @param {import('./mapped-expert-upload-cache.js').MappedExpertFill} fill
   * @param {AbortSignal | null | undefined} externalSignal */
  async #fill(entry, requestedBytes, fill, externalSignal) {
    const controller = new AbortController();
    const externalAbort = () => controller.abort(externalSignal?.reason);
    const shutdownAbort = () => controller.abort(this.#shutdown.signal.reason);
    externalSignal?.addEventListener('abort', externalAbort, { once: true });
    this.#shutdown.signal.addEventListener('abort', shutdownAbort, { once: true });
    if (externalSignal?.aborted) externalAbort();
    if (this.#shutdown.signal.aborted) shutdownAbort();
    try {
      if (entry.bytes !== requestedBytes || this.#allocatedBytes > this.#options.maxBytes
        || this.#mappedBytes > this.#options.maxMappedBytes) throw new Error('Mapped expert allocator changed the admitted extent.');
      if (entry.mapping) {
        // Reserved before the first await, including pending native maps. Close
        // waits this operation; cancellation never claims a pending map is done.
        try { await entry.buffer.mapAsync(GPUMapMode.WRITE, 0, requestedBytes); this.#reuses++; }
        finally { entry.mapping = false; this.#pendingMaps--; }
      }
      controller.signal.throwIfAborted();
      const destination = new Uint8Array(entry.buffer.getMappedRange(0, requestedBytes));
      const written = await fill(destination, controller.signal);
      if (written !== requestedBytes) throw new Error('Mapped expert fill did not complete its declared byte extent.');
      controller.signal.throwIfAborted();
      this.#unmap(entry);
      entry.firstConsumption = true;
      this.#publish(entry);
      this.#ghost.delete(entry.key);
      this.#admissions++;
      return true;
    } catch (error) {
      // Any native map awaited above has settled; a pre-map failure has no native request.
      if (entry.mapping) { entry.mapping = false; this.#pendingMaps--; }
      try { this.#retire(entry); }
      catch (cleanup) { throw new AggregateError([error, cleanup], 'Mapped expert admission and retirement failed.'); }
      throw error;
    } finally {
      externalSignal?.removeEventListener('abort', externalAbort);
      this.#shutdown.signal.removeEventListener('abort', shutdownAbort);
    }
  }

  /** @param {import('./mapped-expert-upload-cache.js').MappedExpertUse} use */
  #startDrain(use) {
    const attempt = (async () => {
      try { if (!isDeviceLost(this.#options.device)) await this.#options.device.queue.onSubmittedWorkDone(); }
      catch (error) {
        if (!isDeviceLost(this.#options.device)) { use.failure = error; throw error; }
      }
      if (this.#uses.delete(use)) use.entry.users--;
      use.failure = null;
    })();
    use.attempt = attempt;
    attempt.catch(() => undefined);
    return attempt;
  }

  /** @type {import('./mapped-expert-upload-cache.js').MappedExpertUploadCache['copyInto']} */
  copyInto(key, copies, signal = null) {
    this.#assertOpen(); this.#assertKey(key); signal?.throwIfAborted();
    const entry = this.#entries.get(key);
    if (!entry) { this.#misses++; return null; }
    // Snapshot descriptors before encoding; callers cannot mutate an outstanding copy.
    const ranges = Array.isArray(copies) ? copies.map(copy => ({ ...copy })) : [];
    if (!ranges.length || ranges.length > 6) throw new Error('Mapped expert copies require one to six bounded ranges.');
    for (const range of ranges) {
      if (![range.sourceOffset, range.destinationOffset, range.bytes].every(value => Number.isSafeInteger(value) && value >= 0 && value % 4 === 0)
        || !range.bytes || range.sourceOffset + range.bytes > entry.bytes
        || range.destination === entry.buffer || !(range.destination?.usage & GPUBufferUsage.COPY_DST)
        || range.destinationOffset + range.bytes > range.destination.size) throw new Error('Mapped expert copy exceeds its owned source or destination.');
    }
    const encoder = this.#options.device.createCommandEncoder({ label: 'mapped_expert_copy' });
    for (const range of ranges) encoder.copyBufferToBuffer(entry.buffer, range.sourceOffset,
      range.destination, range.destinationOffset, range.bytes);
    signal?.throwIfAborted();
    const command = encoder.finish();
    const use = { entry, attempt: null, failure: null };
    const reusedPayload = !entry.firstConsumption;
    const promote = reusedPayload && entry.users === 0;
    entry.users++; this.#uses.add(use);
    try { this.#options.device.queue.submit([command]); }
    catch (error) { this.#startDrain(use); throw error; }
    this.#touch(entry, promote);
    entry.firstConsumption = false;
    this.#hits++; this.#copySubmissions++;
    if (reusedPayload) this.#reuseHits++;
    const bytes = ranges.reduce((sum, range) => sum + range.bytes, 0);
    this.#copyBytes += bytes;
    return { bytes, settled: this.#startDrain(use) };
  }

  /** Retry failed fences explicitly; a rejected attempt never releases the source pin. */
  async settle() {
    const results = await Promise.allSettled([...this.#uses].map(use => use.failure || !use.attempt
      ? this.#startDrain(use) : use.attempt));
    const errors = results.filter(result => result.status === 'rejected').map(result => result.reason);
    if (errors.length) throw new AggregateError(errors, 'Mapped expert copies did not settle.');
  }

  getStats() {
    const publishedBytes = this.#publishedBytes, pinnedBytes = this.#publishedPinnedBytes;
    return { admission: this.#options.admission ?? 'all', protectedShare: this.#options.protectedShare ?? 0,
      maxGhostEntries: this.#options.maxGhostEntries ?? this.#options.maxEntries,
      probationEntries: this.#probation.size, protectedEntries: this.#protected.size,
      protectedBytes: this.#protectedBytes, protectedBudgetBytes: this.#protectedBudget(), ghostEntries: this.#ghost.size,
      allocations: this.#allocations, reuseAttempts: this.#reuseAttempts, reuses: this.#reuses, admissionSkips: this.#admissionSkips, reuseHits: this.#reuseHits,
      maxBytes: this.#options.maxBytes, maxMappedBytes: this.#options.maxMappedBytes,
      maxEntries: this.#options.maxEntries, allocatedBytes: this.#allocatedBytes, mappedBytes: this.#mappedBytes,
      peakBytes: this.#peakBytes, peakMappedBytes: this.#peakMappedBytes,
      publishedBytes, pinnedBytes, evictableBytes: publishedBytes - pinnedBytes, unpublishedBytes: this.#allocatedBytes - publishedBytes,
      entries: this.#entries.size, pinnedEntries: this.#entries.size - this.#probation.size - this.#protected.size,
      evictableEntries: this.#probation.size + this.#protected.size, activeWriters: this.#writers.size,
      pendingMaps: this.#pendingMaps,
      pendingCopies: this.#uses.size, ownedAllocations: this.#owned.size,
      hits: this.#hits, misses: this.#misses, admissions: this.#admissions, evictions: this.#evictions,
      rejections: this.#rejections, copySubmissions: this.#copySubmissions, copyBytes: this.#copyBytes };
  }

  close() {
    if (this.#closeTask) return this.#closeTask;
    this.#closing = true;
    this.#shutdown.abort(new DOMException('Mapped expert cache closed.', 'AbortError'));
    const task = (async () => {
      // A writer that ignores cancellation still owns its mapping until it returns.
      await Promise.allSettled([...this.#writers]);
      const errors = [];
      try { await this.settle(); } catch (error) { errors.push(error); }
      for (const entry of this.#owned) {
        if (entry.users) continue;
        try { this.#retire(entry); } catch (error) { errors.push(error); }
      }
      if (errors.length) throw new AggregateError(errors, 'Mapped expert cache retirement failed; ownership remains retryable.');
      this.#ghost.clear();
    })();
    this.#closeTask = task.finally(() => { this.#closeTask = null; });
    this.#closeTask.catch(() => undefined);
    return this.#closeTask;
  }
}
