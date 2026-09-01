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

const { BufferPool, BufferUsage } = await import('../../src/memory/buffer-pool.js');
const { DEFAULT_BUFFER_POOL_CONFIG } = await import('../../src/config/schema/buffer-pool.schema.js');
const { setDevice } = await import('../../src/gpu/device.js');

function createSchemaConfig() {
  return structuredClone(DEFAULT_BUFFER_POOL_CONFIG);
}

class FakeBuffer {
  constructor({ size, usage, mapReject = false, mapPromise = null, unmapReject = false }) {
    this.size = size;
    this.usage = usage;
    this.mapReject = mapReject;
    this.mapPromise = mapPromise;
    this.unmapReject = unmapReject;
    this.destroyed = false;
    this.unmapped = false;
  }

  async mapAsync() {
    if (this.mapReject) {
      throw new Error('map failed');
    }
    await this.mapPromise;
  }

  getMappedRange(_offset = 0, size = this.size) {
    return new Uint8Array(size).buffer;
  }

  unmap() {
    if (this.unmapReject) {
      throw new Error('unmap failed');
    }
    if (this.destroyed) {
      throw new Error('unmap called after destroy');
    }
    this.unmapped = true;
  }

  destroy() {
    this.destroyed = true;
  }
}

function createFakeDevice(options = {}) {
  const createdBuffers = [];
  const rejectSubmitted = options.rejectSubmitted === true;
  const rejectMapRead = options.rejectMapRead === true;
  const mapPromise = options.mapPromise ?? null;
  const rejectUnmap = options.rejectUnmap === true;

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
    queue,
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
    createBindGroup() {
      return {};
    },
    createBuffer({ size, usage }) {
      const mapReject = rejectMapRead && (usage & GPUBufferUsage.MAP_READ) !== 0;
      const unmapReject = rejectUnmap && (usage & GPUBufferUsage.MAP_READ) !== 0;
      const buffer = new FakeBuffer({ size, usage, mapReject, mapPromise, unmapReject });
      createdBuffers.push(buffer);
      return buffer;
    },
    createCommandEncoder() {
      return {
        copyBufferToBuffer() {},
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

{
  const device = createFakeDevice({ rejectSubmitted: true });
  setDevice(device, { platformConfig: null });

  const pool = new BufferPool(false, createSchemaConfig());
  pool.configure({ enablePooling: false });

  const buffer = pool.acquire(64, BufferUsage.STORAGE, 'deferred_destroy_failure');
  pool.release(buffer);
  await flushMicrotasks();

  assert.equal(buffer.destroyed, true);
  assert.equal(pool.getStats().activeBuffers, 0);
  assert.equal(pool.getStats().currentBytesAllocated, 0);
}

{
  const device = createFakeDevice({ rejectMapRead: true });
  setDevice(device, { platformConfig: null });

  const pool = new BufferPool(false, createSchemaConfig());
  const source = { size: 64, usage: BufferUsage.STORAGE };

  await assert.rejects(
    () => pool.readBufferSlice(source, 0, 32),
    /map failed/
  );
  await flushMicrotasks();

  const staging = device.createdBuffers[0];
  assert.ok(staging);
  assert.equal(staging.destroyed, true);
  assert.equal(pool.getStats().activeBuffers, 0);
  assert.equal(pool.getStats().currentBytesAllocated, 0);
}

{
  let resolveMap;
  const mapPromise = new Promise((resolve) => {
    resolveMap = resolve;
  });
  const device = createFakeDevice({ mapPromise });
  setDevice(device, { platformConfig: null });

  const pool = new BufferPool(false, createSchemaConfig());
  const source = { size: 64, usage: BufferUsage.STORAGE };
  const readPromise = pool.readBufferSlice(source, 0, 32);
  await flushMicrotasks();

  const staging = device.createdBuffers[0];
  assert.ok(staging);
  pool.destroy();
  await flushMicrotasks();
  assert.equal(staging.destroyed, true);

  resolveMap();
  const result = await readPromise;
  assert.equal(result.byteLength, 32);
  assert.equal(staging.unmapped, false);
  assert.equal(pool.getStats().activeBuffers, 0);
  assert.equal(pool.getStats().currentBytesAllocated, 0);
}

{
  const device = createFakeDevice({ rejectUnmap: true });
  setDevice(device, { platformConfig: null });

  const pool = new BufferPool(false, createSchemaConfig());
  const source = { size: 64, usage: BufferUsage.STORAGE };

  await assert.rejects(
    () => pool.readBufferSlice(source, 0, 32),
    /unmap failed/
  );
  await flushMicrotasks();

  const staging = device.createdBuffers[0];
  assert.ok(staging);
  assert.equal(staging.destroyed, true);
  assert.equal(pool.getStats().activeBuffers, 0);
  assert.equal(pool.getStats().currentBytesAllocated, 0);
}

{
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });

  const pool = new BufferPool(false, createSchemaConfig());
  const buffer = pool.acquire(64, BufferUsage.STORAGE, 'discard_buffer');
  pool.discard(buffer);
  await flushMicrotasks();

  assert.equal(buffer.destroyed, true);
  assert.equal(pool.getStats().activeBuffers, 0);
  assert.equal(pool.getStats().currentBytesAllocated, 0);
}

setDevice(null);
console.log('buffer-pool-cleanup.test: ok');
