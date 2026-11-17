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
    this.#options = options;
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

  /** Retain charges and identity if disposal throws. No submitted reader may remain.
   * @param {import('./mapped-expert-upload-cache.js').MappedExpertEntry} entry */
  #retire(entry) {
    if (entry.users) throw new Error('Mapped expert cache cannot retire an active submitted copy.');
    const dispose = this.#options.dispose ?? (buffer => buffer.destroy());
    // destroy also invalidates a mapped range. A failed unmap must not release
    // mapped credit until unmap or destruction has actually succeeded.
    dispose(entry.buffer);
    if (entry.mapped) { entry.mapped = false; this.#mappedBytes -= entry.bytes; }
    if (this.#owned.delete(entry)) this.#allocatedBytes -= entry.bytes;
    if (this.#entries.get(entry.key) === entry) this.#entries.delete(entry.key);
  }

  /** @param {number} bytes */
  #makeRoom(bytes) {
    let remainingBytes = this.#allocatedBytes, remainingEntries = this.#owned.size;
    const victims = [];
    for (const entry of this.#entries.values()) {
      if (remainingBytes + bytes <= this.#options.maxBytes && remainingEntries + 1 <= this.#options.maxEntries) break;
      if (entry.pinned || entry.users) continue;
      victims.push(entry); remainingBytes -= entry.bytes; remainingEntries--;
    }
    if (remainingBytes + bytes > this.#options.maxBytes || remainingEntries + 1 > this.#options.maxEntries) return false;
    for (const victim of victims) { this.#retire(victim); this.#evictions++; }
    return true;
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
      if (options.pinned) existing.pinned = true;
      this.#entries.delete(key); this.#entries.set(key, existing);
      return Promise.resolve(true);
    }
    if ([...this.#owned].some(entry => entry.key === key)) return Promise.reject(new Error('Mapped expert identity has an unfinished admission or retirement.'));
    if (bytes > this.#options.maxBytes || this.#mappedBytes + bytes > this.#options.maxMappedBytes
      || bytes > this.#options.device.limits.maxBufferSize || !this.#makeRoom(bytes)) {
      this.#rejections++;
      return Promise.resolve(false);
    }
    const allocate = this.#options.allocate ?? (descriptor => this.#options.device.createBuffer(descriptor));
    let buffer;
    try { buffer = allocate({ size: bytes, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true, label: 'mapped_expert_upload' }); }
    catch (error) { return Promise.reject(error); }
    const entry = { key, bytes: buffer.size, buffer, mapped: true, pinned: options.pinned === true, users: 0 };
    this.#owned.add(entry); this.#allocatedBytes += entry.bytes; this.#mappedBytes += entry.bytes;
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
      controller.signal.throwIfAborted();
      const destination = new Uint8Array(entry.buffer.getMappedRange(0, requestedBytes));
      const written = await fill(destination, controller.signal);
      if (written !== requestedBytes) throw new Error('Mapped expert fill did not complete its declared byte extent.');
      controller.signal.throwIfAborted();
      this.#unmap(entry);
      this.#entries.set(entry.key, entry);
      this.#admissions++;
      return true;
    } catch (error) {
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
    entry.users++; this.#uses.add(use);
    this.#entries.delete(key); this.#entries.set(key, entry);
    try { this.#options.device.queue.submit([command]); }
    catch (error) { this.#startDrain(use); throw error; }
    this.#hits++; this.#copySubmissions++;
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
    const published = [...this.#entries.values()];
    const publishedBytes = published.reduce((sum, entry) => sum + entry.bytes, 0);
    const pinnedBytes = published.reduce((sum, entry) => sum + (entry.pinned ? entry.bytes : 0), 0);
    return { maxBytes: this.#options.maxBytes, maxMappedBytes: this.#options.maxMappedBytes,
      maxEntries: this.#options.maxEntries, allocatedBytes: this.#allocatedBytes, mappedBytes: this.#mappedBytes,
      peakBytes: this.#peakBytes, peakMappedBytes: this.#peakMappedBytes,
      publishedBytes, pinnedBytes, evictableBytes: publishedBytes - pinnedBytes, unpublishedBytes: this.#allocatedBytes - publishedBytes,
      entries: published.length, pinnedEntries: published.filter(entry => entry.pinned).length,
      evictableEntries: published.filter(entry => !entry.pinned).length, activeWriters: this.#writers.size,
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
    })();
    this.#closeTask = task.finally(() => { this.#closeTask = null; });
    this.#closeTask.catch(() => undefined);
    return this.#closeTask;
  }
}
