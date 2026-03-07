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
  const features = new Set(options.features ?? []);

  const queue = {
    submit() {},
    writeBuffer() {},
    onSubmittedWorkDone() {
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

resetUniformCache();
resetRuntimeConfig();
setDevice(null);
console.log('command-recorder-cleanup.test: ok');
