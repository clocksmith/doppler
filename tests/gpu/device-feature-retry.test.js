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

const {
  FEATURES,
  getKernelCapabilities,
  initDevice,
  resetDeviceState,
} = await import('../../src/gpu/device.js');

class FakeBuffer {
  constructor({ size, usage }) {
    this.size = size;
    this.usage = usage;
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }

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

function createFakeDevice(features) {
  return {
    features: new Set(features),
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

function installNavigator(adapter) {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      gpu: {
        requestAdapter() {
          return Promise.resolve(adapter);
        },
      },
    },
  });
  return () => {
    if (original) {
      Object.defineProperty(globalThis, 'navigator', original);
    } else {
      delete globalThis.navigator;
    }
  };
}

resetDeviceState();

const requestDeviceCalls = [];
const adapter = {
  features: new Set([FEATURES.SHADER_F16, FEATURES.SUBGROUPS, FEATURES.TIMESTAMP_QUERY]),
  limits,
  info: {
    vendor: 'unit',
    architecture: 'feature-retry',
    device: 'fake',
    description: 'Fake adapter',
  },
  async requestDevice(descriptor = {}) {
    const features = [...(descriptor.requiredFeatures || [])];
    requestDeviceCalls.push(features);
    if (features.includes(FEATURES.TIMESTAMP_QUERY)) {
      throw new Error('timestamp-query denied in this context');
    }
    return createFakeDevice(features);
  },
};

const restoreNavigator = installNavigator(adapter);
try {
  await initDevice();
  const caps = getKernelCapabilities();
  assert.equal(caps.hasF16, true);
  assert.equal(caps.hasSubgroups, true);
  assert.equal(caps.hasTimestampQuery, false);
  assert.deepEqual(requestDeviceCalls[0], [
    FEATURES.SHADER_F16,
    FEATURES.SUBGROUPS,
    FEATURES.TIMESTAMP_QUERY,
  ]);
  assert.deepEqual(requestDeviceCalls[1], [
    FEATURES.SHADER_F16,
    FEATURES.SUBGROUPS,
  ]);
} finally {
  resetDeviceState();
  restoreNavigator();
}

console.log('device-feature-retry.test: ok');
