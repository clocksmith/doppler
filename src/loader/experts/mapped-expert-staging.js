import { recordRoutedExpertSourceTimings } from './routed-expert-source.js';

/*
 * Staged expert uploads for policy.uploadStaging === 'mapped'.
 *
 * queue.writeBuffer copies every expert on the JavaScript thread, which made that thread the limit
 * of expert loading. Here a small ring of mapped upload buffers receives the bytes instead: the
 * storage port reads already verified ranges on the I/O pool straight into a mapped slot, the slot
 * is unmapped, and one submitted command buffer copies each plane to its place. The submit happens
 * where the writeBuffer calls happened, so queue order relative to the caller's kernels is the same.
 *
 * The port declines a range it cannot serve from verified state (first-time verification, or a
 * changed file). The ring then asks a lane to read the same ranges once: the lane hashes their
 * pieces and records them verified, its copy of the bytes is dropped, and the direct read is tried
 * again. If it is still declined nothing has reached a GPU-visible buffer, and the caller uses its
 * lease path.
 */

const MAX_SLOTS = 16;
/** @type {WeakMap<object, MappedExpertStaging | null>} */
const stagings = new WeakMap();

class MappedExpertStaging {
  /** @type {GPUDevice} */ #device;
  /** @type {NonNullable<import('../../storage/verified-piece-gather.js').VerifiedRangeLeasePort['readVerifiedInto']>} */ #read;
  /** @type {((ranges: Array<{ shard: number, offset: number, length: number }>, signal: AbortSignal) => Promise<import('../../storage/verified-piece-gather.js').VerifiedRangeLease>) | null} */ #verify;
  #maxRanges; #maxPartBytes;
  /** @type {Array<{ buffer: GPUBuffer, view: Uint8Array | null }>} */ #slots = [];
  /** @type {Array<{ buffer: GPUBuffer, view: Uint8Array | null }>} */ #free = [];
  /** @type {Array<(slot: { buffer: GPUBuffer, view: Uint8Array | null } | null) => void>} */ #waiters = [];
  /** @type {Set<Promise<void>>} */ #remapping = new Set();
  #closed = false;

  /** @param {{ device: GPUDevice, port: import('../../storage/verified-piece-gather.js').VerifiedRangeLeasePort, slotBytes: number, slotCount: number }} options */
  constructor({ device, port, slotBytes, slotCount }) {
    if (typeof port.readVerifiedInto !== 'function') throw new Error('Mapped expert staging requires direct verified reads.');
    this.#device = device; this.#read = port.readVerifiedInto.bind(port);
    this.#verify = typeof port.readMany === 'function' ? port.readMany.bind(port) : null;
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
   * @returns {Promise<boolean>} false when nothing was staged and nothing was submitted
   */
  async upload(parts, copies, signal, beforeSubmit) {
    const slot = await this.#take();
    if (!slot) return false;
    let unmapped = false;
    try {
      const view = slot.view;
      if (!view) throw new Error('Staged upload slot is not mapped.');
      // Every read settles before the slot is unmapped or handed on, including after a failure.
      const fill = async () => {
        const outcomes = await Promise.allSettled(parts.map(part =>
          this.#read(part.shard, part.offset, view.subarray(part.at, part.at + part.length), signal)));
        const failed = outcomes.find(outcome => outcome.status === 'rejected');
        if (failed) throw /** @type {PromiseRejectedResult} */ (failed).reason;
        return outcomes.every(outcome => outcome.status === 'fulfilled' && outcome.value === true);
      };
      if (!await fill()) {
        if (!this.#verify || !await this.#verifyRanges(parts, signal) || !await fill()) return false;
      }
      signal.throwIfAborted();
      if (this.#closed) return false;
      slot.view = null; slot.buffer.unmap(); unmapped = true;
      beforeSubmit?.();
      const encoder = this.#device.createCommandEncoder({ label: 'expert_staged_upload' });
      for (const copy of copies) encoder.copyBufferToBuffer(slot.buffer, copy.at, copy.target, copy.targetOffset, copy.length);
      this.#device.queue.submit([encoder.finish()]);
      return true;
    } finally {
      if (unmapped) this.#remap(slot);
      else this.#give(slot);
    }
  }

  /** Have the lanes authenticate these ranges; their copy of the bytes is not used.
   * @param {ReadonlyArray<{ shard: number, offset: number, length: number }>} parts @param {AbortSignal} signal */
  async #verifyRanges(parts, signal) {
    if (!this.#verify) return false;
    /** @type {Array<Array<{ shard: number, offset: number, length: number }>>} */ const groups = [];
    let bytes = 0;
    for (const { shard, offset, length } of parts) {
      if (length > this.#maxPartBytes) return false;
      const group = groups.at(-1);
      if (!group || group.length >= this.#maxRanges || bytes + length > this.#maxPartBytes) { groups.push([{ shard, offset, length }]); bytes = length; }
      else { group.push({ shard, offset, length }); bytes += length; }
    }
    for (const group of groups) {
      signal.throwIfAborted();
      (await this.#verify(group, signal)).release();
    }
    return true;
  }

  async destroy() {
    this.#closed = true;
    for (const waiter of this.#waiters.splice(0)) waiter(null);
    await Promise.allSettled([...this.#remapping]);
    await this.#device.queue.onSubmittedWorkDone();
    for (const slot of this.#slots.splice(0)) slot.buffer.destroy();
    this.#free.length = 0;
  }
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
    staging = slotCount >= 1 ? new MappedExpertStaging({ device: options.device, port, slotBytes, slotCount }) : null;
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
  if (!await staging.upload(parts, copies, signal, beforeSubmit)) return false;
  const elapsed = performance.now() - started;
  // No lane ran, so there is no lane timing to report; the job is counted as untimed.
  recordRoutedExpertSourceTimings(stats, [null]);
  stats.sourceReads += parts.length; stats.sourceReadBytes += at; stats.sourceReadIntoBytes += at;
  stats.uploadBytes += at; stats.sourceReadWaitMs += elapsed; stats.ioWaitMs += elapsed;
  stats.peakStagingBytes = Math.max(stats.peakStagingBytes, staging.bytes);
  if (layerStats) {
    layerStats.sourceReadBytes += at; layerStats.uploadBytes += at; layerStats.sourceReadWaitMs += elapsed;
  }
  return true;
}
