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
  constructor({ size, usage }) {
    this.size = size;
    this.usage = usage;
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }
}

globalThis.GPUBuffer = FakeBuffer;

const { setDevice } = await import('../../src/gpu/device.js');
const { destroyBufferPool, getBufferPool } = await import('../../src/memory/buffer-pool.js');
const { initRoPEFrequencies } = await import('../../src/inference/pipelines/text/init.js');

function createFakeDevice() {
  const createdBuffers = [];
  let writeBufferCount = 0;
  return {
    createdBuffers,
    get writeBufferCount() {
      return writeBufferCount;
    },
    queue: {
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
      submit() {},
      writeBuffer() {
        writeBufferCount += 1;
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
    createBuffer({ size, usage }) {
      const buffer = new FakeBuffer({ size, usage });
      createdBuffers.push(buffer);
      return buffer;
    },
    createBindGroup() {
      return {};
    },
  };
}

function resetRuntimeState(device = null) {
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
  if (device) {
    getBufferPool().configure({ enablePooling: false });
  }
}

const ropeConfig = {
  headDim: 256,
  localHeadDim: 64,
  rotaryDim: 64,
  ropeLocalRotaryDim: 64,
  ropeFrequencyBaseDim: 64,
  ropeLocalFrequencyBaseDim: 64,
  maxSeqLen: 8,
  ropeTheta: 1000000,
  ropeLocalTheta: null,
  mropeInterleaved: false,
  mropeSection: null,
  partialRotaryFactor: 0.25,
  ropeLocalPartialRotaryFactor: null,
  ropeScale: 1,
  ropeLocalScale: 1,
  ropeScalingType: null,
  ropeLocalScalingType: null,
  ropeScaling: null,
  ropeLocalScaling: null,
};

{
  resetRuntimeState();
  const first = await initRoPEFrequencies(ropeConfig, false);
  const second = await initRoPEFrequencies(ropeConfig, false);
  assert.equal(second.cos, first.cos);
  assert.equal(second.sin, first.sin);
  assert.equal(second.localCos, first.localCos);
  assert.equal(second.localSin, first.localSin);
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);
  const first = await initRoPEFrequencies(ropeConfig, true);
  const createdBuffersAfterFirst = device.createdBuffers.length;
  const writeBufferCountAfterFirst = device.writeBufferCount;
  const second = await initRoPEFrequencies(ropeConfig, true);
  assert.equal(second.cos, first.cos);
  assert.equal(second.sin, first.sin);
  assert.equal(second.localCos, first.localCos);
  assert.equal(second.localSin, first.localSin);
  assert.equal(device.createdBuffers.length, createdBuffersAfterFirst);
  assert.equal(device.writeBufferCount, writeBufferCountAfterFirst);
  resetRuntimeState();
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);
  const first = await initRoPEFrequencies(ropeConfig, true);
  const createdBuffersAfterFirst = device.createdBuffers.length;
  const writeBufferCountAfterFirst = device.writeBufferCount;

  destroyBufferPool();
  getBufferPool().configure({ enablePooling: false });

  const second = await initRoPEFrequencies(ropeConfig, true);
  assert.equal(first.cos.destroyed, true);
  assert.equal(first.sin.destroyed, true);
  assert.notEqual(second.cos, first.cos);
  assert.notEqual(second.sin, first.sin);
  assert.equal(second.cos.destroyed, false);
  assert.equal(second.sin.destroyed, false);
  assert.equal(second.localCos, null);
  assert.equal(second.localSin, null);
  assert.ok(device.createdBuffers.length > createdBuffersAfterFirst);
  assert.ok(device.writeBufferCount > writeBufferCountAfterFirst);
  resetRuntimeState();
}

console.log('rope-frequency-cache-contract.test: ok');
