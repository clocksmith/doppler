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

const { GPUProfiler } = await import('../../src/gpu/profiler.js');
const { FEATURES } = await import('../../src/gpu/device.js');
const { configurePerfGuards } = await import('../../src/gpu/perf-guards.js');
const { setRuntimeConfig, resetRuntimeConfig } = await import('../../src/config/runtime.js');

class FakeBuffer {
  constructor({ size, usage, mapReject = false, rangeBuffer = null }) {
    this.size = size;
    this.usage = usage;
    this.mapReject = mapReject;
    this.rangeBuffer = rangeBuffer ?? new ArrayBuffer(size);
    this.destroyed = false;
    this.unmapped = false;
  }

  async mapAsync() {
    if (this.mapReject) {
      throw new Error('map failed');
    }
  }

  getMappedRange() {
    return this.rangeBuffer;
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

function createTimestampRange() {
  return new BigUint64Array([0n, 1_000_000n, 0n, 1_000_000n]).buffer;
}

function createFakeDevice(options = {}) {
  const createdBuffers = [];
  const createdQuerySets = [];

  const device = {
    createdBuffers,
    createdQuerySets,
    features: new Set([FEATURES.TIMESTAMP_QUERY]),
    limits: {
      maxQuerySetSize: 64,
    },
    queue: {
      submit() {},
    },
    createQuerySet() {
      const querySet = new FakeQuerySet();
      createdQuerySets.push(querySet);
      return querySet;
    },
    createBuffer({ size, usage }) {
      const isReadback = (usage & GPUBufferUsage.MAP_READ) !== 0;
      const buffer = new FakeBuffer({
        size,
        usage,
        mapReject: isReadback && options.mapReject === true,
        rangeBuffer: isReadback ? createTimestampRange() : null,
      });
      createdBuffers.push(buffer);
      if (isReadback) {
        device.readbackBuffer = buffer;
      }
      return buffer;
    },
    createCommandEncoder() {
      return {
        resolveQuerySet() {},
        copyBufferToBuffer() {},
        finish() {
          return {};
        },
      };
    },
  };

  return device;
}

function writeSample(profiler, label = 'sample') {
  const pass = {
    writeTimestamp() {},
  };
  profiler.writeTimestamp(pass, label, false);
  profiler.writeTimestamp(pass, label, true);
}

setRuntimeConfig({
  shared: {
    debug: {
      profiler: {
        queryCapacity: 4,
        maxSamples: 8,
        maxDurationMs: 1000,
      },
    },
  },
});

{
  const device = createFakeDevice();
  const profiler = new GPUProfiler(device);

  configurePerfGuards({
    allowGPUReadback: false,
    trackSubmitCount: false,
    trackAllocations: false,
    logExpensiveOps: false,
    strictMode: false,
  });

  writeSample(profiler);
  await profiler.resolve();
  assert.deepEqual(profiler.getResults(), {});

  configurePerfGuards({
    allowGPUReadback: true,
    trackSubmitCount: false,
    trackAllocations: false,
    logExpensiveOps: false,
    strictMode: false,
  });

  writeSample(profiler);
  await profiler.resolve();

  assert.equal(profiler.getResult('sample')?.count, 1);
  profiler.destroy();
}

{
  const device = createFakeDevice({ mapReject: true });
  const profiler = new GPUProfiler(device);

  configurePerfGuards({
    allowGPUReadback: true,
    trackSubmitCount: false,
    trackAllocations: false,
    logExpensiveOps: false,
    strictMode: false,
  });

  writeSample(profiler);
  await assert.rejects(() => profiler.resolve(), /map failed/);

  device.readbackBuffer.mapReject = false;
  writeSample(profiler);
  await profiler.resolve();

  assert.equal(profiler.getResult('sample')?.count, 1);
  profiler.destroy();
}

resetRuntimeConfig();
console.log('profiler-cleanup.test: ok');
