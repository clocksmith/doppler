import assert from 'node:assert/strict';

globalThis.GPUBufferUsage = {
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
};

class FakeBuffer {
  constructor({ size, usage }) {
    this.size = size;
    this.usage = usage;
  }

  destroy() {}
}

const ORIGINAL_GPU_BUFFER = globalThis.GPUBuffer;

const { setDevice } = await import('../../src/gpu/device.js');

const fakeDevice = {
  features: new Set(),
  limits: {
    maxStorageBufferBindingSize: 1 << 20,
    maxBufferSize: 1 << 20,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupStorageSize: 16384,
  },
  queue: {
    submit() {},
    writeBuffer() {},
  },
  createBuffer({ size, usage }) {
    return new FakeBuffer({ size, usage });
  },
  createBindGroup() {
    return {};
  },
  createCommandEncoder() {
    return {
      finish() {
        return {};
      },
    };
  },
  createShaderModule() {
    return {};
  },
  lost: new Promise(() => {}),
};

try {
  delete globalThis.GPUBuffer;
  setDevice(fakeDevice, { platformConfig: null });
  assert.equal(globalThis.GPUBuffer, FakeBuffer);
} finally {
  setDevice(null, { platformConfig: null });
  if (ORIGINAL_GPU_BUFFER === undefined) {
    delete globalThis.GPUBuffer;
  } else {
    globalThis.GPUBuffer = ORIGINAL_GPU_BUFFER;
  }
}

console.log('device-gpubuffer-constructor.test: ok');
