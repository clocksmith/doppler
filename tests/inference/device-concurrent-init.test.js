import assert from 'node:assert/strict';

globalThis.GPUBufferUsage = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  STORAGE: 0x0080,
};
globalThis.GPUMapMode = {
  READ: 0x0001,
};

class FakeBuffer {
  constructor({ size, usage }) {
    this.size = size;
    this.usage = usage;
  }

  destroy() {}

  mapAsync() {
    return Promise.resolve();
  }

  unmap() {}
}

const limits = {
  maxStorageBufferBindingSize: 1 << 20,
  maxBufferSize: 1 << 20,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 1,
  maxComputeWorkgroupSizeZ: 1,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupStorageSize: 16384,
  maxStorageBuffersPerShaderStage: 8,
  maxUniformBufferBindingSize: 65536,
};

function createFakeDevice(id) {
  return {
    id,
    features: new Set(),
    limits,
    queue: {
      submit() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    createBuffer(descriptor) {
      return new FakeBuffer(descriptor);
    },
    createBindGroup(descriptor) {
      return descriptor;
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        copyBufferToBuffer() {},
        finish() {
          return {};
        },
      };
    },
    createComputePipeline() {
      return {
        getBindGroupLayout() {
          return {};
        },
      };
    },
    createShaderModule() {
      return {};
    },
    destroy() {},
  };
}

const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
let adapterRequests = 0;
let deviceRequests = 0;
const adapter = {
  features: new Set(),
  limits,
  info: {
    vendor: 'unit',
    architecture: 'concurrent-init',
    device: 'fake',
    description: 'Concurrent initialization test adapter',
  },
  async requestDevice() {
    deviceRequests += 1;
    await Promise.resolve();
    return createFakeDevice(deviceRequests);
  },
};

Object.defineProperty(globalThis, 'navigator', {
  configurable: true,
  value: {
    gpu: {
      async requestAdapter() {
        adapterRequests += 1;
        await Promise.resolve();
        return adapter;
      },
    },
  },
});

const moduleA = await import('../../src/gpu/device.js?concurrent-init-a');
const moduleB = await import('../../src/gpu/device.js?concurrent-init-b');

try {
  moduleA.resetDeviceState();
  const [deviceA, deviceB] = await Promise.all([
    moduleA.initDevice(),
    moduleB.initDevice(),
  ]);

  assert.equal(adapterRequests, 1, 'concurrent callers must share one adapter request');
  assert.equal(deviceRequests, 1, 'concurrent callers must share one device request');
  assert.equal(deviceA, deviceB, 'concurrent callers must receive the same GPUDevice');
  assert.equal(moduleA.getDevice(), deviceA);
  assert.equal(moduleB.getDevice(), deviceA);
  assert.equal(moduleA.getDeviceEpoch(), moduleB.getDeviceEpoch());
} finally {
  moduleA.resetDeviceState();
  if (originalNavigator) {
    Object.defineProperty(globalThis, 'navigator', originalNavigator);
  } else {
    delete globalThis.navigator;
  }
}

console.log('device-concurrent-init.test: ok');
