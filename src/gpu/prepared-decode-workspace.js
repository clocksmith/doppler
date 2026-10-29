import { isDeviceLost, observeDeviceLoss } from '../memory/device-residency.js';

// Session-owned decode storage. Queue order serializes writers and all consumers
// before the next token reuses a slot. The session, never a recorder, owns it.
export class PreparedDecodeWorkspace {
  /** @type {GPUDevice} */
  #device;
  /** @type {Map<string, GPUBuffer>} */
  #buffers = new Map();
  /** @type {Set<GPUBuffer>} */
  #owned = new Set();
  /** @type {Map<string, { layout: GPUBindGroupLayout; bindGroup: GPUBindGroup; entries: Array<{ binding: number; buffer: GPUBuffer; offset?: number; size?: number }> }>} */
  #bindings = new Map();
  #hits = 0;
  #misses = 0;
  #released = false;

  /** @param {GPUDevice} device */
  constructor(device) { this.#device = device; observeDeviceLoss(device); }

  /** @param {string} slot @param {number} bytes */
  getBuffer(slot, bytes) {
    if (this.#released || isDeviceLost(this.#device)) throw new Error('Prepared decode workspace was released or its GPUDevice was lost.');
    if (!Number.isSafeInteger(bytes) || bytes <= 0 || bytes % 4 !== 0) {
      throw new Error('Prepared decode workspace requires a positive four-byte-aligned size.');
    }
    const existing = this.#buffers.get(slot);
    if (existing) {
      if (existing.size !== bytes) throw new Error(`Prepared decode slot ${slot} changed size; create a new session.`);
      return existing;
    }
    const buffer = this.#device.createBuffer({ label: `prepared_decode_${slot}`, size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.#buffers.set(slot, buffer);
    this.#owned.add(buffer);
    return buffer;
  }

  /** @param {GPUBuffer} buffer */
  ownsBuffer(buffer) { return this.#owned.has(buffer); }

  /** @param {string} slot @param {GPUBindGroupDescriptor} descriptor */
  getBindGroup(slot, descriptor) {
    if (this.#released || isDeviceLost(this.#device)) throw new Error('Prepared decode workspace was released or its GPUDevice was lost.');
    const entries = Array.from(descriptor.entries);
    const old = this.#bindings.get(slot);
    if (old && old.layout === descriptor.layout && old.entries.length === entries.length
        && entries.every((entry, i) => {
          const previous = old.entries[i];
          const resource = entry.resource;
          return resource && 'buffer' in resource && previous.binding === entry.binding
            && previous.buffer === resource.buffer && previous.offset === resource.offset
            && previous.size === resource.size;
        })) {
      this.#hits++;
      return old.bindGroup;
    }
    const bindGroup = this.#device.createBindGroup(descriptor);
    this.#misses++;
    const snapshots = [];
    for (const entry of entries) {
      if (!entry.resource || !('buffer' in entry.resource)) {
        this.#bindings.delete(slot);
        return bindGroup;
      }
      snapshots.push({ binding: entry.binding, buffer: entry.resource.buffer,
        offset: entry.resource.offset, size: entry.resource.size });
    }
    this.#bindings.set(slot, { layout: descriptor.layout, bindGroup, entries: snapshots });
    return bindGroup;
  }

  getStats() {
    return { buffers: this.#buffers.size, bytes: [...this.#owned].reduce((sum, buffer) => sum + buffer.size, 0),
      bindings: this.#bindings.size, hits: this.#hits, misses: this.#misses };
  }

  release() {
    if (this.#released) return;
    this.#released = true;
    this.#bindings.clear();
    for (const buffer of this.#owned) buffer.destroy();
    this.#buffers.clear();
    this.#owned.clear();
  }
}
