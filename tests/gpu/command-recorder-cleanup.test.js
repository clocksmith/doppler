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

globalThis.GPUMapMode = {
  READ: 1 << 0,
  WRITE: 1 << 1,
};

const { CommandRecorder } = await import('../../src/gpu/command-recorder.js');
const { FEATURES, setDevice } = await import('../../src/gpu/device.js');
const { configurePerfGuards } = await import('../../src/gpu/perf-guards.js');
const { resetUniformCache } = await import('../../src/gpu/uniform-cache.js');
const { setRuntimeConfig, resetRuntimeConfig } = await import('../../src/config/runtime.js');
const {
  acquireBuffer,
  destroyBufferPool,
  getBufferPool,
  markPersistentBuffer,
  releaseBuffer,
} = await import('../../src/memory/buffer-pool.js');

class FakeBuffer {
  constructor({ size, usage, mapReject = false }) {
    this.size = size;
    this.usage = usage;
    this.mapReject = mapReject;
    this.destroyed = false;
    this.unmapped = false;
  }

  async mapAsync() {
    if (this.mapReject) {
      throw new Error('map failed');
    }
  }

  getMappedRange(_offset = 0, size = this.size) {
    return new ArrayBuffer(size);
  }

  unmap() {
    this.unmapped = true;
  }

  destroy() {
    this.destroyed = true;
  }
}

class FakeQuerySet {
  destroyed = false;

  destroy() {
    this.destroyed = true;
  }
}

function createFakeDevice(options = {}) {
  const createdBuffers = [];
  const createdQuerySets = [];
  const rejectSubmitted = options.rejectSubmitted === true;
  const rejectMapRead = options.rejectMapRead === true;
  const submitThrows = options.submitThrows === true;
  const throwCreateBufferAt = options.throwCreateBufferAt ?? null;
  const features = new Set(options.features ?? []);
  let createBufferCount = 0;
  let submittedWorkDoneCount = 0;

  const queue = {
    submit() {
      if (submitThrows) {
        throw new Error('submit failed');
      }
    },
    writeBuffer() {},
    onSubmittedWorkDone() {
      submittedWorkDoneCount += 1;
      if (rejectSubmitted) {
        return Promise.reject(new Error('device lost'));
      }
      return Promise.resolve();
    },
  };

  return {
    createdBuffers,
    createdQuerySets,
    queue,
    features,
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
      maxQuerySetSize: 64,
    },
    createBindGroup() {
      return {};
    },
    createQuerySet() {
      const querySet = new FakeQuerySet();
      createdQuerySets.push(querySet);
      return querySet;
    },
    createBuffer({ size, usage }) {
      createBufferCount += 1;
      if (throwCreateBufferAt === createBufferCount) {
        throw new Error(`createBuffer failed at ${createBufferCount}`);
      }
      const mapReject = rejectMapRead && (usage & GPUBufferUsage.MAP_READ) !== 0;
      const buffer = new FakeBuffer({ size, usage, mapReject });
      createdBuffers.push(buffer);
      return buffer;
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            end() {},
          };
        },
        copyBufferToBuffer() {},
        resolveQuerySet() {},
        finish() {
          return {};
        },
      };
    },
    getSubmittedWorkDoneCount() {
      return submittedWorkDoneCount;
    },
  };
}

async function flushMicrotasks() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

configurePerfGuards({
  allowGPUReadback: true,
  trackSubmitCount: false,
  trackAllocations: false,
  logExpensiveOps: false,
  strictMode: false,
});

{
  const device = createFakeDevice({ rejectSubmitted: true });
  const recorder = new CommandRecorder(device, 'cleanup_failure');
  const temp = recorder.createTempBuffer(64, GPUBufferUsage.STORAGE, 'temp');

  recorder.submit();
  await flushMicrotasks();

  assert.equal(temp.destroyed, true);
}

{
  const device = createFakeDevice();
  const recorder = new CommandRecorder(device, 'deferred_cleanup');
  const temp = recorder.createTempBuffer(64, GPUBufferUsage.STORAGE, 'temp');

  recorder.submit({ cleanup: 'deferred' });

  assert.equal(temp.destroyed, false);
  assert.equal(device.getSubmittedWorkDoneCount(), 0);

  await recorder.completeDeferredCleanup();

  assert.equal(temp.destroyed, true);
  assert.equal(device.getSubmittedWorkDoneCount(), 0);
  assert.equal(typeof recorder.getSubmitLatencyMs(), 'number');
}

{
  const device = createFakeDevice();
  const recorder = new CommandRecorder(device, 'deferred_completion_task');
  let completed = false;

  recorder.enqueueCompletionTask(async () => {
    completed = true;
  });
  recorder.submit({ cleanup: 'deferred' });

  assert.equal(completed, false);
  await recorder.completeDeferredCleanup();
  assert.equal(completed, true);
  assert.equal(device.getSubmittedWorkDoneCount(), 1);
}

{
  const device = createFakeDevice();
  const recorder = new CommandRecorder(device, 'external_buffer_preserved');
  const external = new FakeBuffer({ size: 64, usage: GPUBufferUsage.STORAGE });

  recorder.trackTemporaryBuffer(external);
  recorder.submit();
  await flushMicrotasks();

  assert.equal(external.destroyed, false);
}

{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });
  getBufferPool().configure({ enablePooling: false });

  const recorder = new CommandRecorder(device, 'persistent_pooled_buffer_preserved');
  const persistent = acquireBuffer(64, GPUBufferUsage.STORAGE, 'persistent_weight');
  markPersistentBuffer(persistent);

  recorder.trackTemporaryBuffer(persistent);
  recorder.submit();
  await flushMicrotasks();

  assert.equal(persistent.destroyed, false);
  releaseBuffer(persistent);
  destroyBufferPool();
  setDevice(null);
}

{
  const device = createFakeDevice({ submitThrows: true });
  const recorder = new CommandRecorder(device, 'submit_throw_cleanup');
  const temp = recorder.createTempBuffer(64, GPUBufferUsage.STORAGE, 'temp');

  assert.throws(
    () => recorder.submit(),
    /submit failed/
  );

  assert.equal(temp.destroyed, true);
}

{
  setRuntimeConfig({
    shared: {
      debug: {
        profiler: {
          maxQueries: 8,
          defaultQueryLimit: 8,
        },
      },
    },
  });

  const device = createFakeDevice({
    rejectMapRead: true,
    features: [FEATURES.TIMESTAMP_QUERY],
  });
  setDevice(device, { platformConfig: null });

  const recorder = new CommandRecorder(device, 'profile_cleanup', { profile: true });
  assert.equal(recorder.isProfilingEnabled(), true);

  recorder.beginComputePass('timed_pass').end();
  recorder.submit();

  await assert.rejects(
    () => recorder.resolveProfileTimings(),
    /map failed/
  );

  assert.equal(device.createdQuerySets[0].destroyed, true);
  assert.equal(device.createdBuffers[0].destroyed, true);
  assert.equal(device.createdBuffers[1].destroyed, true);
}

{
  setRuntimeConfig({
    shared: {
      debug: {
        profiler: {
          maxQueries: 8,
          defaultQueryLimit: 8,
        },
      },
    },
  });

  const device = createFakeDevice({
    throwCreateBufferAt: 2,
    features: [FEATURES.TIMESTAMP_QUERY],
  });
  setDevice(device, { platformConfig: null });

  const recorder = new CommandRecorder(device, 'profile_init_cleanup', { profile: true });
  assert.equal(recorder.isProfilingEnabled(), false);
  assert.equal(device.createdQuerySets[0].destroyed, true);
  assert.equal(device.createdBuffers[0].destroyed, true);
}

resetUniformCache();
destroyBufferPool();
resetRuntimeConfig();
setDevice(null);
console.log('command-recorder-cleanup.test: ok');
