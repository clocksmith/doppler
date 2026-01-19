
import { log } from '../debug/index.js';
import { MODULE, generateBufferId } from './virtual-utils.js';

export class VirtualCPU {
  constructor(index, spec, opfsRootPath) {
    this.index = index;
    this.spec = spec;
    this.opfsPath = `${opfsRootPath}/cpu${index}`;

    this._buffers = new Map();

    this._ramBuffers = new Map();

    this._allocated = 0;

    this._initialized = false;
  }

  async initialize() {
    if (this._initialized) return;
    this._initialized = true;
    log.verbose(MODULE, `VirtualCPU ${this.index} initialized`);
  }

  async allocate(sizeBytes, label) {
    await this.initialize();

    const id = generateBufferId();
    const buffer = new ArrayBuffer(sizeBytes);

    this._buffers.set(id, { sizeBytes, label });
    this._ramBuffers.set(id, buffer);
    this._allocated += sizeBytes;

    log.verbose(MODULE, `CPU ${this.index}: Allocated ${label || id} (${sizeBytes} bytes)`);

    const cpu = this;
    return {
      metadata: {
        id,
        sizeBytes,
        location: 'ram',
        gpuIndex: -1,
        label,
        lastAccessTime: Date.now(),
        accessCount: 0,
        pinned: false,
      },
      async getData() {
        return cpu.read(id);
      },
      async getGPUBuffer() {
        throw new Error('CPU buffers cannot be promoted to GPU');
      },
      release() {
        cpu.free(id);
      },
    };
  }

  async write(bufferId, data) {
    const buffer = this._ramBuffers.get(bufferId);
    if (!buffer) {
      throw new Error(`Buffer ${bufferId} not found on CPU ${this.index}`);
    }
    new Uint8Array(buffer).set(new Uint8Array(data));
  }

  async read(bufferId) {
    const buffer = this._ramBuffers.get(bufferId);
    if (!buffer) {
      throw new Error(`Buffer ${bufferId} not found on CPU ${this.index}`);
    }
    return buffer.slice(0);
  }

  async free(bufferId) {
    const meta = this._buffers.get(bufferId);
    if (!meta) return;

    this._ramBuffers.delete(bufferId);
    this._buffers.delete(bufferId);
    this._allocated -= meta.sizeBytes;

    log.verbose(MODULE, `CPU ${this.index}: Freed ${meta.label || bufferId}`);
  }

  getMemoryStats() {
    return {
      cpuIndex: this.index,
      totalMemoryBytes: this.spec.memoryBytes,
      allocatedBytes: this._allocated,
      bufferCount: this._buffers.size,
    };
  }

  async destroy() {
    this._ramBuffers.clear();
    this._buffers.clear();
    this._allocated = 0;
    this._initialized = false;

    log.verbose(MODULE, `VirtualCPU ${this.index} destroyed`);
  }
}
