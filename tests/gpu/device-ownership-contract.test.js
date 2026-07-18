import assert from 'node:assert/strict';

function deferred() {
  let resolve;
  const promise = new Promise((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

class FakeBuffer {
  constructor(owner, size, usage) {
    this.owner = owner;
    this.size = size;
    this.usage = usage;
  }

  destroy() {}
}

function createFakeDevice(label, lost = null) {
  let bindGroupCalls = 0;
  return {
    label,
    features: new Set(),
    limits: {
      maxStorageBufferBindingSize: 1 << 20,
      maxBufferSize: 1 << 20,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
      maxStorageBuffersPerShaderStage: 8,
      maxUniformBufferBindingSize: 65536,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    queue: {
      submit() {},
      writeBuffer() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    lost: lost?.promise ?? null,
    createBuffer({ size, usage }) {
      return new FakeBuffer(label, size, usage);
    },
    createBindGroup(descriptor) {
      bindGroupCalls += 1;
      return descriptor;
    },
    createCommandEncoder() {
      return { finish() { return {}; } };
    },
    createShaderModule() {
      return {};
    },
    destroy() {},
    bindGroupCalls() {
      return bindGroupCalls;
    },
  };
}

const first = await import('../../src/gpu/device.js?ownership=first');
const second = await import('../../src/gpu/device.js?ownership=second');
const staleLoss = deferred();
const deviceA = createFakeDevice('A', staleLoss);
const deviceB = createFakeDevice('B');

first.setDevice(deviceA, { platformConfig: null });
assert.equal(second.getDevice(), deviceA);
const bufferA = deviceA.createBuffer({ size: 16, usage: 0x80 });

second.setDevice(null);
assert.equal(first.getDevice(), null);

second.setDevice(deviceB, { platformConfig: null });
assert.equal(first.getDevice(), deviceB);

staleLoss.resolve({ message: 'stale device loss', reason: 'destroyed' });
await Promise.resolve();
await Promise.resolve();
assert.equal(first.getDevice(), deviceB);
assert.equal(second.getDevice(), deviceB);

assert.throws(
  () => deviceB.createBindGroup({
    label: 'cross-device-contract',
    entries: [{ binding: 0, resource: { buffer: bufferA } }],
  }),
  /GPUBuffer created by a different GPUDevice/
);
assert.equal(deviceB.bindGroupCalls(), 0);

const bufferB = deviceB.createBuffer({ size: 16, usage: 0x80 });
deviceB.createBindGroup({
  label: 'same-device-contract',
  entries: [{ binding: 0, resource: { buffer: bufferB } }],
});
assert.equal(deviceB.bindGroupCalls(), 1);

first.setDevice(null);
assert.equal(second.getDevice(), null);

const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
const initializedDevice = createFakeDevice('initialized');
const reinitializedDevice = createFakeDevice('reinitialized');
let requestDeviceCalls = 0;
const adapter = {
  features: new Set(),
  limits: initializedDevice.limits,
  info: {
    vendor: 'unit',
    architecture: 'device-ownership',
    device: 'fake',
    description: 'Fake adapter',
  },
  async requestDevice() {
    requestDeviceCalls += 1;
    await Promise.resolve();
    return requestDeviceCalls === 1 ? initializedDevice : reinitializedDevice;
  },
};
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

try {
  const [initializedA, initializedB] = await Promise.all([
    first.initDevice(),
    second.initDevice(),
  ]);
  assert.equal(requestDeviceCalls, 1);
  assert.equal(initializedA, initializedDevice);
  assert.equal(initializedB, initializedDevice);
  assert.equal(first.getDevice(), initializedDevice);
  assert.equal(second.getDevice(), initializedDevice);

  const cachedDevice = await second.initDevice();
  assert.equal(cachedDevice, initializedDevice);
  assert.equal(requestDeviceCalls, 1);

  first.resetDeviceState();
  const [reinitializedA, reinitializedB] = await Promise.all([
    first.initDevice(),
    second.initDevice(),
  ]);
  assert.equal(requestDeviceCalls, 2);
  assert.equal(reinitializedA, reinitializedDevice);
  assert.equal(reinitializedB, reinitializedDevice);
  assert.equal(first.getDevice(), reinitializedDevice);
  assert.equal(second.getDevice(), reinitializedDevice);
} finally {
  first.resetDeviceState();
  if (originalNavigator) {
    Object.defineProperty(globalThis, 'navigator', originalNavigator);
  } else {
    delete globalThis.navigator;
  }
}

console.log('device-ownership-contract.test: ok');
