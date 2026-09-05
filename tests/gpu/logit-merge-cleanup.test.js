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
const { createShaderSourceScope, runWithShaderSourceScope } = await import('../../src/gpu/kernels/shader-source-scope.js');

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

const ORIGINAL_GPU_BUFFER = globalThis.GPUBuffer;
globalThis.GPUBuffer = FakeBuffer;

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
  const compiledSources = [];
  return {
    createdBuffers,
    compiledSources,
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
      compiledSources.push(descriptor.code);
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

{
  const deferred = createDeferred();
  const device = createFakeDevice(deferred);
  setDevice(device, { platformConfig: null });
  const kernel = new LogitMergeKernel();
  await kernel.init();
  assert.equal(device.compiledSources.length, 3);
  await assert.rejects(
    () => runWithShaderSourceScope(createShaderSourceScope(new Map()), () => kernel.init()),
    /logit_merge_weighted.wgsl is outside the verified Pack source closure/
  );
  const sources = new Map(['weighted', 'max', 'geometric'].map((strategy) =>
    [`logit_merge_${strategy}.wgsl`, `// scoped fixture ${strategy}`]));
  const scope = createShaderSourceScope(sources);
  await runWithShaderSourceScope(scope, async () => {
    await kernel.init();
    await kernel.init();
    assert.equal(device.compiledSources.length, 6);
    assert.deepEqual(device.compiledSources.slice(3), [...sources.values()]);
  });
  // A failed replacement must not leave the previous scope marked initialized
  // after clearing its pipelines. Exercise actual use, not just init's return.
  await assert.rejects(
    () => runWithShaderSourceScope(createShaderSourceScope(new Map()), () => kernel.init()),
    /outside the verified Pack source closure/
  );
  const a = new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  const b = new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  const output = await runWithShaderSourceScope(scope, () =>
    kernel.merge(a, b, 4, { strategy: 'weighted', weights: [0.5, 0.5], temperature: 1 }));
  output.destroy();
  deferred.resolve();
  await flushMicrotasks();
}

setDevice(null);
console.log('logit-merge-cleanup.test: ok');
if (ORIGINAL_GPU_BUFFER === undefined) {
  delete globalThis.GPUBuffer;
} else {
  globalThis.GPUBuffer = ORIGINAL_GPU_BUFFER;
}
