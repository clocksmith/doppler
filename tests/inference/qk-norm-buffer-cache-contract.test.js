import assert from 'node:assert/strict';

globalThis.GPUBufferUsage = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
};

class FakeBuffer {
  constructor({ size, usage, label = '', owner }) {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.owner = owner;
  }

  destroy() {}
}

function createFakeDevice(owner) {
  const createdBuffers = [];
  return {
    owner,
    createdBuffers,
    queue: {
      writeBuffer() {},
      submit() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
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
    createBuffer({ size, usage, label }) {
      const buffer = new FakeBuffer({ size, usage, label, owner });
      createdBuffers.push(buffer);
      return buffer;
    },
    createBindGroup() {
      return {};
    },
    createShaderModule() {
      return {};
    },
    createCommandEncoder() {
      return {
        finish() {
          return {};
        },
      };
    },
  };
}

const { setDevice } = await import('../../src/gpu/device.js');
const { getQKNormOnesBuffer } = await import('../../src/inference/pipelines/text/attention/types.js');

const deviceA = createFakeDevice('A');
setDevice(deviceA, { platformConfig: null });
const createdAAfterSetDevice = deviceA.createdBuffers.length;
const aFirst = getQKNormOnesBuffer(256);
const aSecond = getQKNormOnesBuffer(256);
assert.equal(aSecond, aFirst);
assert.equal(deviceA.createdBuffers.length - createdAAfterSetDevice, 1);
assert.equal(aFirst.owner, 'A');

const deviceB = createFakeDevice('B');
setDevice(deviceB, { platformConfig: null });
const createdBAfterSetDevice = deviceB.createdBuffers.length;
const bFirst = getQKNormOnesBuffer(256);
const bSecond = getQKNormOnesBuffer(256);
assert.notEqual(bFirst, aFirst);
assert.equal(bSecond, bFirst);
assert.equal(deviceB.createdBuffers.length - createdBAfterSetDevice, 1);
assert.equal(bFirst.owner, 'B');

setDevice(null, { platformConfig: null });

console.log('qk-norm-buffer-cache-contract.test: ok');
