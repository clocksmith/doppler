import { recordRoutedExpertSourceTimings } from './routed-expert-source.js';
import { trace, isTraceEnabled } from '../../debug/index.js';

/*
 * Staged expert uploads for policy.uploadStaging === 'mapped'.
 *
 * queue.writeBuffer copies every expert on the JavaScript thread, which made that thread the limit
 * of expert loading. Here a small ring of mapped upload buffers receives the bytes instead: the
 * storage port reads already verified ranges on the I/O pool straight into a mapped slot, the slot
 * is unmapped, and one submitted command buffer copies each plane to its place. The submit happens
 * where the writeBuffer calls happened, so queue order relative to the caller's kernels is the same.
 *
 * A range whose pieces are not verified yet (first-time verification) is read into the slot at the
 * same time as a lane hashes those pieces. The slot is unmapped and its copy submitted only after the
 * lane has finished and the port confirms that every piece is verified for the file state the read
 * saw. Until then the bytes exist only in the mapped slot, which no model buffer and no kernel can
 * see. If the confirmation fails, or the port cannot read ahead, nothing is submitted and the caller
 * uses its lease path. The lane's own copy of the bytes is not used: copying it into the slot would
 * put the copy back on the JavaScript thread.
 *
 * policy.uploadVerification === 'before-release' moves that wait. Inside a release gate (a resident
 * partition step opens one and settles it before its output leaves the stage) first-time bytes are
 * submitted as soon as they are read, the kernels compute with them, and the lanes hash in the
 * background. Settling the gate waits for every hash and confirmation. If one fails, the step fails,
 * its output is never released, and the gate stays failed: no later step may begin until the stage is
 * reopened, because unverified bytes are resident in model buffers. Outside a gate nothing is deferred.
 */

const MAX_SLOTS = 16;
/** @type {WeakMap<object, MappedExpertStaging | null>} */
const stagings = new WeakMap();
/** One release gate per storage port.
 * @type {WeakMap<object, { depth: number, tasks: Set<Promise<void>>, abort: AbortController, failure: Error | null, stats: import('./routed-expert-source.js').RoutedExpertCounters | null }>} */
const gates = new WeakMap();

/** @type {import('./mapped-expert-staging.js').beginStagedRelease} */
export function beginStagedRelease(port) {
  let gate = gates.get(port);
  if (!gate) { gate = { depth: 0, tasks: new Set(), abort: new AbortController(), failure: null, stats: null }; gates.set(port, gate); }
  if (gate.failure) throw gate.failure;
  gate.depth++;
}

/** @type {import('./mapped-expert-staging.js').assertStagedVerification} */
export function assertStagedVerification(port) {
  const failure = gates.get(port)?.failure;
  if (failure) throw failure;
}

/** @type {import('./mapped-expert-staging.js').settleStagedRelease} */
export async function settleStagedRelease(port, { release = true } = {}) {
  const gate = gates.get(port);
  if (!gate || gate.depth < 1) throw new Error('Staged release gate is not open.');
  const started = performance.now();
  try {
    // A task may start another only through a new upload, which the caller has stopped making.
    while (gate.tasks.size) await Promise.allSettled([...gate.tasks]);
  } finally { gate.depth--; }
  if (gate.stats) gate.stats.stagedReleaseWaitMs += performance.now() - started;
  if (release && gate.failure) throw gate.failure;
}

class MappedExpertStaging {
  /** @type {GPUDevice} */ #device;
  /** @type {NonNullable<import('../../storage/verified-piece-gather.js').VerifiedRangeLeasePort['readVerifiedInto']>} */ #read;
  /** @type {((ranges: Array<{ shard: number, offset: number, length: number }>, signal: AbortSignal) => Promise<import('../../storage/verified-piece-gather.js').VerifiedRangeLease>) | null} */ #verify;
  /** @type {import('../../storage/verified-piece-gather.js').VerifiedRangeLeasePort['readAheadInto'] | null} */ #readAhead;
  /** @type {import('../../storage/verified-piece-gather.js').VerifiedRangeLeasePort['hasVerified'] | null} */ #hasVerified;
  /** @type {object} */ #port;
  /** @type {import('../../storage/verified-piece-gather.js').VerifiedRangeLeasePort['verifyMany'] | null} */ #verifyQueued;
  /** @type {import('../../storage/verified-piece-gather.js').VerifiedRangeLeasePort['verifyMany'] | null} */ #verifyWaiting;
  #lanes = 1; #active = 0;
  #maxRanges; #maxPartBytes;
  /** @type {Array<{ buffer: GPUBuffer, view: Uint8Array | null }>} */ #slots = [];
  /** @type {Array<{ buffer: GPUBuffer, view: Uint8Array | null }>} */ #free = [];
  /** @type {Array<(slot: { buffer: GPUBuffer, view: Uint8Array | null } | null) => void>} */ #waiters = [];
  /** @type {Set<Promise<void>>} */ #remapping = new Set();
  #closed = false;

  /** @param {{ device: GPUDevice, port: import('../../storage/verified-piece-gather.js').VerifiedRangeLeasePort, slotBytes: number, slotCount: number, deferVerification: boolean }} options */
  constructor({ device, port, slotBytes, slotCount, deferVerification }) {
    if (typeof port.readVerifiedInto !== 'function') throw new Error('Mapped expert staging requires direct verified reads.');
    this.#device = device; this.#read = port.readVerifiedInto.bind(port);
    this.#verify = typeof port.readMany === 'function' ? port.readMany.bind(port) : null;
    const ahead = typeof port.readAheadInto === 'function' && typeof port.hasVerified === 'function';
    this.#readAhead = ahead ? /** @type {NonNullable<typeof port.readAheadInto>} */ (port.readAheadInto).bind(port) : null;
    this.#hasVerified = ahead ? /** @type {NonNullable<typeof port.hasVerified>} */ (port.hasVerified).bind(port) : null;
    this.#port = port;
    this.#verifyWaiting = typeof port.verifyMany === 'function' ? port.verifyMany.bind(port) : null;
    this.#verifyQueued = deferVerification ? this.#verifyWaiting : null;
    this.#lanes = Number.isSafeInteger(port.concurrency) && Number(port.concurrency) > 0 ? Number(port.concurrency) : 1;
    this.#maxRanges = port.maxRanges ?? 3; this.#maxPartBytes = port.maxPartBytes;
    this.slotBytes = slotBytes; this.slotCount = slotCount;
  }

  get bytes() { return this.#slots.length * this.slotBytes; }

  /** @returns {Promise<{ buffer: GPUBuffer, view: Uint8Array | null } | null>} */
  #take() {
    if (this.#closed) return Promise.resolve(null);
    const ready = this.#free.pop();
    if (ready) return Promise.resolve(ready);
    if (this.#slots.length < this.slotCount) {
      const buffer = this.#device.createBuffer({ label: 'expert_staged_upload', size: this.slotBytes,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
      const slot = { buffer, view: new Uint8Array(buffer.getMappedRange(0, this.slotBytes)) };
      this.#slots.push(slot);
      return Promise.resolve(slot);
    }
    return new Promise(resolve => this.#waiters.push(resolve));
  }

  /** @param {{ buffer: GPUBuffer, view: Uint8Array | null }} slot */
  #give(slot) {
    const waiter = this.#waiters.shift();
    if (waiter) waiter(slot); else this.#free.push(slot);
  }

  /** The GPU may still be copying from the slot; it is free again when the map resolves.
   * @param {{ buffer: GPUBuffer, view: Uint8Array | null }} slot */
  #remap(slot) {
    const operation = slot.buffer.mapAsync(GPUMapMode.WRITE).then(() => {
      if (this.#closed) return;
      slot.view = new Uint8Array(slot.buffer.getMappedRange(0, this.slotBytes));
      this.#give(slot);
    }, () => {
      // A slot that cannot be mapped again is retired; waiting callers fall back to their lease path.
      this.#slots = this.#slots.filter(candidate => candidate !== slot);
      slot.buffer.destroy();
      if (this.#slots.length === 0) for (const waiter of this.#waiters.splice(0)) waiter(null);
    });
    this.#remapping.add(operation);
    operation.finally(() => this.#remapping.delete(operation));
  }

  /**
   * @param {ReadonlyArray<{ shard: number, offset: number, length: number, at: number }>} parts
   * @param {ReadonlyArray<{ at: number, length: number, target: GPUBuffer, targetOffset: number }>} copies
   * @param {AbortSignal} signal @param {(() => void) | null} beforeSubmit
   * @param {import('./routed-expert-source.js').RoutedExpertCounters | null} [stats] receives the timings of background verification
   * @param {number | null} [layer] names the upload in the loader trace
   * @returns {Promise<false | import('./mapped-expert-staging.js').StagedUpload>} false when nothing was staged and nothing was submitted
   */
  async upload(parts, copies, signal, beforeSubmit, stats = null, layer = null) {
    const gate = gates.get(this.#port) ?? null;
    if (gate?.failure) throw gate.failure;
    const slot = await this.#take();
    if (!slot) return false;
    let unmapped = false;
    this.#active++;
    try {
      const view = slot.view;
      if (!view) throw new Error('Staged upload slot is not mapped.');
      const started = performance.now();
      const hasVerified = this.#hasVerified, readAhead = this.#readAhead;
      const pending = hasVerified ? parts.filter(part => !hasVerified(part.shard, part.offset, part.length)) : [];
      if (pending.length && (!this.#verify || !readAhead || pending.some(part => part.length > this.#maxPartBytes))) return false;
      const ahead = new Set(pending);
      const verifyQueued = this.#verifyQueued;
      if (pending.length && verifyQueued && gate && gate.depth > 0) {
        // Verification before release: submit once the bytes are read, and let the gate hold the output.
        const read = await Promise.allSettled(parts.map(part => {
          const target = view.subarray(part.at, part.at + part.length);
          return ahead.has(part) && readAhead ? readAhead(part.shard, part.offset, target, signal)
            : this.#read(part.shard, part.offset, target, signal);
        }));
        const broken = read.find(outcome => outcome.status === 'rejected');
        if (broken) throw /** @type {PromiseRejectedResult} */ (broken).reason;
        const values = read.map(outcome => /** @type {PromiseFulfilledResult<boolean | (() => boolean)>} */ (outcome).value);
        // Only an already verified range (true) or a read with a confirmation to run later may be submitted.
        if (values.some(value => value !== true && typeof value !== 'function')) return false;
        const readWait = performance.now() - started;
        signal.throwIfAborted();
        // The gate must still be open: an upload that outlived its step has nothing left to hold its bytes back.
        if (this.#closed || gate.failure || gate.depth < 1) return false;
        const confirms = values.filter(value => typeof value === 'function');
        const groups = this.#groups(pending);
        const bytes = pending.reduce((sum, part) => sum + part.length, 0);
        slot.view = null; slot.buffer.unmap(); unmapped = true;
        beforeSubmit?.();
        const encoder = this.#device.createCommandEncoder({ label: 'expert_staged_upload' });
        for (const copy of copies) encoder.copyBufferToBuffer(slot.buffer, copy.at, copy.target, copy.targetOffset, copy.length);
        this.#device.queue.submit([encoder.finish()]);
        // From here unverified bytes are in model buffers. The verification task is registered before anything
        // else can throw. Every lane job settles before the outcome is decided, and only a failed hash or a
        // failed confirmation fails the gate; counters and trace lines cannot.
        const task = (async () => {
          const outcomes = await Promise.allSettled(groups.map(group => verifyQueued(group, gate.abort.signal)));
          const failed = outcomes.find(outcome => outcome.status === 'rejected');
          let failure = failed ? /** @type {PromiseRejectedResult} */ (failed).reason : null;
          if (!failure) {
            try {
              for (const confirm of confirms) {
                if (!(/** @type {() => boolean} */ (confirm))()) throw new Error('Expert bytes already in use could not be confirmed verified.');
              }
            } catch (error) { failure = error; }
          }
          if (failure) {
            gate.failure ??= new Error(`Staged expert verification failed after upload: ${failure?.message ?? failure}`, { cause: failure });
            return;
          }
          try {
            const timings = outcomes.map(outcome => /** @type {PromiseFulfilledResult<import('./mapped-expert-staging.js').StagedUpload['laneTimings'][number]>} */ (outcome).value ?? null);
            if (stats) recordRoutedExpertSourceTimings(stats, timings.map(timing => timing ? { timings: timing } : null));
            if (isTraceEnabled('loader')) traceUpload({ layer, bytes, verifiedAfterSubmit: true, startedAt: started, verifyDoneAt: performance.now(), lanes: timings });
          } catch { /* Observation only: the bytes are verified. */ }
        })();
        gate.tasks.add(task);
        task.finally(() => gate.tasks.delete(task));
        if (stats) gate.stats = stats;
        if (isTraceEnabled('loader')) traceUpload({ layer, bytes, firstTime: bytes, deferred: true, startedAt: started, readDoneAt: started + readWait, submittedAt: performance.now() });
        return { laneTimings: [], readAheadBytes: bytes, readWaitMs: readWait, verifyWaitMs: 0, deferred: true };
      }
      // The reads and the lanes run together. Every read settles before the slot is unmapped or handed
      // on, including after a failure, because a native read may still be writing into the slot.
      let readWaitMs = 0, verifyWaitMs = 0;
      const reads = Promise.allSettled(parts.map(part => {
        const target = view.subarray(part.at, part.at + part.length);
        return ahead.has(part) && readAhead ? readAhead(part.shard, part.offset, target, signal)
          : this.#read(part.shard, part.offset, target, signal);
      })).then(outcomes => { readWaitMs = performance.now() - started; return outcomes; });
      /** @type {Promise<{ timings: import('./mapped-expert-staging.js').StagedUpload['laneTimings'], error?: unknown }>} */
      // Lanes that no other upload is using hash this upload's ranges side by side. During decode a layer
      // asks for one or two experts at a time, so most lanes would otherwise sit idle behind one job.
      const verification = pending.length
        ? this.#verifyRanges(pending, signal, Math.floor(this.#lanes / this.#active)).then(timings => { verifyWaitMs = performance.now() - started; return { timings }; },
          error => ({ timings: [], error }))
        : Promise.resolve({ timings: [] });
      const [outcomes, verified] = await Promise.all([reads, verification]);
      const failed = outcomes.find(outcome => outcome.status === 'rejected');
      if (failed) throw /** @type {PromiseRejectedResult} */ (failed).reason;
      if ('error' in verified) throw verified.error;
      // A range read ahead counts only once the port confirms its pieces are now verified.
      for (const outcome of outcomes) {
        const value = /** @type {PromiseFulfilledResult<boolean | (() => boolean)>} */ (outcome).value;
        if (value !== true && (typeof value !== 'function' || !value())) return false;
      }
      signal.throwIfAborted();
      if (this.#closed) return false;
      slot.view = null; slot.buffer.unmap(); unmapped = true;
      beforeSubmit?.();
      const encoder = this.#device.createCommandEncoder({ label: 'expert_staged_upload' });
      for (const copy of copies) encoder.copyBufferToBuffer(slot.buffer, copy.at, copy.target, copy.targetOffset, copy.length);
      this.#device.queue.submit([encoder.finish()]);
      const firstTime = pending.reduce((sum, part) => sum + part.length, 0);
      if (isTraceEnabled('loader')) {
        traceUpload({ layer, bytes: parts.reduce((sum, part) => sum + part.length, 0), firstTime, deferred: false, startedAt: started,
          readDoneAt: started + readWaitMs, ...(firstTime ? { verifyDoneAt: started + verifyWaitMs, lanes: verified.timings } : {}), submittedAt: performance.now() });
      }
      return { laneTimings: verified.timings, readAheadBytes: firstTime, readWaitMs, verifyWaitMs, deferred: false };
    } finally {
      this.#active--;
      if (unmapped) this.#remap(slot);
      else this.#give(slot);
    }
  }

  /** Have lanes authenticate these ranges; their copies of the bytes are not used. Resolves each job's timings.
   * With `spread` above one and a port that queues verification, the ranges go to up to that many lanes at once.
   * @param {ReadonlyArray<{ shard: number, offset: number, length: number }>} parts @param {AbortSignal} signal
   * @param {number} [spread]
   * @returns {Promise<import('./mapped-expert-staging.js').StagedUpload['laneTimings']>} */
  async #verifyRanges(parts, signal, spread = 1) {
    if (!this.#verify) throw new Error('Staged upload has no lane to verify with.');
    const waiting = this.#verifyWaiting;
    if (waiting && spread > 1 && parts.length > 1) {
      const jobs = Math.min(spread, parts.length), size = Math.min(this.#maxRanges, Math.ceil(parts.length / jobs));
      /** @type {Array<Array<{ shard: number, offset: number, length: number }>>} */ const chunks = [];
      for (let at = 0; at < parts.length; at += size) chunks.push(parts.slice(at, at + size).map(({ shard, offset, length }) => ({ shard, offset, length })));
      // Every job settles before a failure is reported, so no lane is still reading when the caller moves on.
      const outcomes = await Promise.allSettled(chunks.flatMap(chunk => this.#groups(chunk)).map(group => waiting(group, signal)));
      const failed = outcomes.find(outcome => outcome.status === 'rejected');
      if (failed) throw /** @type {PromiseRejectedResult} */ (failed).reason;
      return outcomes.map(outcome => /** @type {PromiseFulfilledResult<import('./mapped-expert-staging.js').StagedUpload['laneTimings'][number]>} */ (outcome).value ?? null);
    }
    /** @type {import('./mapped-expert-staging.js').StagedUpload['laneTimings']} */ const timings = [];
    for (const group of this.#groups(parts)) {
      signal.throwIfAborted();
      const lease = await this.#verify(group, signal);
      timings.push(lease.timings ?? null);
      lease.release();
    }
    return timings;
  }

  /** Split ranges into lane jobs within the port's range and byte bounds.
   * @param {ReadonlyArray<{ shard: number, offset: number, length: number }>} parts */
  #groups(parts) {
    /** @type {Array<Array<{ shard: number, offset: number, length: number }>>} */ const groups = [];
    let bytes = 0;
    for (const { shard, offset, length } of parts) {
      const group = groups.at(-1);
      if (!group || group.length >= this.#maxRanges || bytes + length > this.#maxPartBytes) { groups.push([{ shard, offset, length }]); bytes = length; }
      else { group.push({ shard, offset, length }); bytes += length; }
    }
    return groups;
  }

  async destroy() {
    this.#closed = true;
    const gate = gates.get(this.#port);
    if (gate) {
      if (gate.tasks.size > 0) {
        gate.failure ??= new Error('Staged expert verification aborted during staging teardown.');
        gate.abort.abort(gate.failure);
      }
      await Promise.allSettled([...gate.tasks]);
    }
    for (const waiter of this.#waiters.splice(0)) waiter(null);
    await Promise.allSettled([...this.#remapping]);
    await this.#device.queue.onSubmittedWorkDone();
    for (const slot of this.#slots.splice(0)) slot.buffer.destroy();
    this.#free.length = 0;
  }
}

/**
 * One loader-trace line per staged upload, in milliseconds on the process clock: when it started, when its
 * reads into the slot had settled, when its lanes had hashed any first-time pieces (with each lane job's own
 * read, hash and wall time), and when its copy was submitted.
 * @param {{ layer: number | null, bytes: number, firstTime?: number, deferred?: boolean, verifiedAfterSubmit?: boolean, startedAt: number,
 *   readDoneAt?: number, verifyDoneAt?: number, submittedAt?: number, lanes?: ReadonlyArray<unknown> }} event */
function traceUpload(event) {
  const round = /** @param {number | undefined} value */ value => value === undefined ? undefined : Number(value.toFixed(2));
  const lanes = (event.lanes ?? []).filter(Boolean).map(job => {
    const timing = /** @type {{ workerWallMs: number, readWaitMs: number, verificationMs: number, roundTripMs: number }} */ (job);
    return [round(timing.workerWallMs), round(timing.readWaitMs), round(timing.verificationMs), round(timing.roundTripMs)];
  });
  trace.loader(`staged-upload ${JSON.stringify({ layer: event.layer, bytes: event.bytes, firstTime: event.firstTime, deferred: event.deferred,
    verifiedAfterSubmit: event.verifiedAfterSubmit, start: round(event.startedAt), readDone: round(event.readDoneAt),
    verifyDone: round(event.verifyDoneAt), submitted: round(event.submittedAt), ...(lanes.length ? { laneWallReadHashRoundTrip: lanes } : {}) })}`);
}

/** One ring per storage port: prefill and decode share it.
 * @type {import('./mapped-expert-staging.js').getMappedExpertStaging} */
export function getMappedExpertStaging(options, plan) {
  const port = options.rangeLease;
  if (options.policy.uploadStaging !== 'mapped' || options.policy.sourceReadMode !== 'expert-lease-batch'
    || !port || typeof port.readVerifiedInto !== 'function') return null;
  let staging = stagings.get(port);
  if (staging === undefined) {
    const slotBytes = Math.ceil(plan.expertBytes / 4) * 4;
    const slotCount = Math.min(MAX_SLOTS, Math.floor(options.policy.stagingBytes / slotBytes));
    staging = slotCount >= 1 ? new MappedExpertStaging({ device: options.device, port, slotBytes, slotCount,
      deferVerification: options.policy.uploadVerification === 'before-release' }) : null;
    stagings.set(port, staging);
  }
  return staging;
}

/** @type {import('./mapped-expert-staging.js').destroyMappedExpertStaging} */
export async function destroyMappedExpertStaging(options) {
  const port = options.rangeLease;
  const staging = port ? stagings.get(port) : null;
  if (!port || !staging) return;
  stagings.delete(port);
  await staging.destroy();
}

/**
 * Stage whole experts (every plane) and copy each plane to the buffer `targetOf` names.
 * @type {import('./mapped-expert-staging.js').stageRoutedExperts}
 */
export async function stageRoutedExperts(staging, stats, plan, experts, targetOf, signal, beforeSubmit, layerStats) {
  if (!experts.length || experts.length * plan.expertBytes > staging.slotBytes
    || plan.planes.some(plane => plane.location.sourceTransform != null
      || (plane.location.spans && plane.location.spans.length !== 1) || plane.expertBytes % 4)) return false;
  /** @type {Array<{ shard: number, offset: number, length: number, at: number }>} */ const parts = [];
  /** @type {Array<{ at: number, length: number, target: GPUBuffer, targetOffset: number }>} */ const copies = [];
  let at = 0;
  for (const expert of experts) {
    for (const plane of plan.planes) {
      const physical = plane.location.spans?.[0] ?? plane.location;
      const offset = Number(physical.offset) + expert * plane.expertBytes;
      if (!Number.isSafeInteger(physical.shardIndex) || Number(physical.shardIndex) < 0 || !Number.isSafeInteger(offset)
        || offset % 4 || offset + plane.expertBytes > Number(physical.offset) + Number(physical.size)) return false;
      const { buffer, offset: targetOffset } = targetOf(expert, plane);
      parts.push({ shard: Number(physical.shardIndex), offset, length: plane.expertBytes, at });
      copies.push({ at, length: plane.expertBytes, target: buffer, targetOffset });
      at += plane.expertBytes;
    }
  }
  const started = performance.now();
  const staged = await staging.upload(parts, copies, signal, beforeSubmit, stats, plan.layer);
  if (!staged) return false;
  const elapsed = performance.now() - started;
  // Lanes ran only for ranges verified for the first time. With none, the job is counted as untimed. A
  // deferred upload's lane jobs are recorded when they finish.
  if (staged.deferred) stats.stagedDeferredBytes += staged.readAheadBytes;
  else recordRoutedExpertSourceTimings(stats, staged.laneTimings.length ? staged.laneTimings.map(timings => timings ? { timings } : null) : [null]);
  stats.stagedReadAheadBytes += staged.readAheadBytes; stats.stagedReadWaitMs += staged.readWaitMs; stats.stagedVerifyWaitMs += staged.verifyWaitMs;
  stats.sourceReads += parts.length; stats.sourceReadBytes += at; stats.sourceReadIntoBytes += at;
  stats.uploadBytes += at; stats.sourceReadWaitMs += elapsed; stats.ioWaitMs += elapsed;
  stats.peakStagingBytes = Math.max(stats.peakStagingBytes, staging.bytes);
  if (layerStats) {
    layerStats.sourceReadBytes += at; layerStats.uploadBytes += at; layerStats.sourceReadWaitMs += elapsed;
  }
  return true;
}
