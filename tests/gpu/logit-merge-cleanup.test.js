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

globalThis.GPUShaderStage = {
  COMPUTE: 0x2,
};

const { LogitMergeKernel } = await import('../../src/gpu/kernels/logit-merge.js');
const { setDevice } = await import('../../src/gpu/device.js');

class FakeBuffer {
  constructor({ size, usage, label }) {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }
}

function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function createFakeDevice(workDoneDeferred) {
  const createdBuffers = [];
  return {
    createdBuffers,
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
        return workDoneDeferred.promise;
      },
    },
    createBindGroupLayout(descriptor) {
      return descriptor;
    },
    createPipelineLayout(descriptor) {
      return descriptor;
    },
    createShaderModule(descriptor) {
      return descriptor;
    },
    async createComputePipelineAsync(descriptor) {
      return descriptor;
    },
    createBindGroup(descriptor) {
      return descriptor;
    },
    createBuffer({ size, usage, label }) {
      const buffer = new FakeBuffer({ size, usage, label });
      createdBuffers.push(buffer);
      return buffer;
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
        finish() {
          return {};
        },
      };
    },
  };
}

async function flushMicrotasks() {
  await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

{
  const deferred = createDeferred();
  const device = createFakeDevice(deferred);
  setDevice(device, { platformConfig: null });
  const kernel = new LogitMergeKernel();
  const logitsA = new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE, label: 'a' });
  const logitsB = new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE, label: 'b' });

  await kernel.merge(logitsA, logitsB, 4, {
    strategy: 'weighted',
    weights: [0.5, 0.5],
    temperature: 1.0,
  });

  assert.equal(device.createdBuffers.length, 2);
  assert.equal(device.createdBuffers[1].label, 'logit-merge-params');
  assert.equal(device.createdBuffers[1].destroyed, false);

  deferred.resolve();
  await flushMicrotasks();

  assert.equal(device.createdBuffers[1].destroyed, true);
}

{
  const deferred = createDeferred();
  const device = createFakeDevice(deferred);
  setDevice(device, { platformConfig: null });
  const kernel = new LogitMergeKernel();
  const logitsA = new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE, label: 'a' });
  const logitsB = new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE, label: 'b' });

  await kernel.merge(logitsA, logitsB, 4, {
    strategy: 'max',
    weights: [0.5, 0.5],
    temperature: 1.0,
  });

  deferred.reject(new Error('device lost'));
  await flushMicrotasks();

  assert.equal(device.createdBuffers[1].destroyed, true);
}

setDevice(null);
console.log('logit-merge-cleanup.test: ok');
