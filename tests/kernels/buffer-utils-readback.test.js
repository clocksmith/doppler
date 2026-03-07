import assert from 'node:assert/strict';

globalThis.GPUBufferUsage = {
  MAP_READ: 0x0001,
  COPY_DST: 0x0008,
};

globalThis.GPUMapMode = {
  READ: 1 << 0,
};

const { readGPUBuffer } = await import('./harness/buffer-utils.js');

class FakeBuffer {
  constructor(size, mapReject = false) {
    this.size = size;
    this.mapReject = mapReject;
    this.destroyed = false;
    this.unmapped = false;
    this.data = new Uint8Array(size);
  }

  async mapAsync() {
    if (this.mapReject) {
      throw new Error('map failed');
    }
  }

  getMappedRange() {
    return this.data.slice(0).buffer;
  }

  unmap() {
    this.unmapped = true;
  }

  destroy() {
    this.destroyed = true;
  }
}

function createDevice({ rejectMapRead = false } = {}) {
  return {
    queue: {
      submit() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    createBuffer({ size, usage }) {
      return new FakeBuffer(size, rejectMapRead && (usage & GPUBufferUsage.MAP_READ) !== 0);
    },
    createCommandEncoder() {
      return {
        copyBufferToBuffer() {},
        finish() {
          return {};
        },
      };
    },
  };
}

{
  const device = createDevice({ rejectMapRead: true });
  const source = new FakeBuffer(16);

  await assert.rejects(
    () => readGPUBuffer(device, source, 16),
    /map failed/
  );
}

console.log('buffer-utils-readback.test: ok');
