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
const {
  acquireBuffer,
  destroyBufferPool,
  getBufferPool,
} = await import('../../src/memory/buffer-pool.js');
const {
  clearDequantCache,
  getCachedDequant,
  setCachedDequant,
  setDequantCacheMaxEntries,
} = await import('../../src/inference/pipelines/text/moe-cache.js');

function createFakeDevice() {
  return {
    lost: new Promise(() => {}),
    queue: {
      submit() {},
      writeBuffer() {},
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
    createBuffer({ size, usage }) {
      return new FakeBuffer({ size, usage });
    },
    createBindGroup() {
      return {};
    },
    destroy() {},
  };
}

function resetRuntime(device = null) {
  clearDequantCache();
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
  if (device) {
    getBufferPool().configure({ enablePooling: false });
  }
}

async function flushDeferredCleanup() {
  await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

try {
  resetRuntime(createFakeDevice());
  setDequantCacheMaxEntries(1);

  const gateUpA = acquireBuffer(16, GPUBufferUsage.STORAGE, 'gate_up_a');
  const downA = acquireBuffer(16, GPUBufferUsage.STORAGE, 'down_a');
  setCachedDequant(0, 0, 'f16', gateUpA, downA);

  const gateUpB = acquireBuffer(16, GPUBufferUsage.STORAGE, 'gate_up_b');
  const downB = acquireBuffer(16, GPUBufferUsage.STORAGE, 'down_b');
  setCachedDequant(0, 1, 'f16', gateUpB, downB);
  await flushDeferredCleanup();

  assert.equal(gateUpA.destroyed, true);
  assert.equal(downA.destroyed, true);
  assert.equal(getCachedDequant(0, 0, 'f16'), undefined);

  clearDequantCache();
  await flushDeferredCleanup();
  assert.equal(gateUpB.destroyed, true);
  assert.equal(downB.destroyed, true);
} finally {
  setDequantCacheMaxEntries(null);
  resetRuntime();
}

console.log('moe-cache-cleanup.test: ok');
