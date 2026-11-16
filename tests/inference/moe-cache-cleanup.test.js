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
const { ExpertDequantCache } = await import('../../src/loader/experts/expert-dequant-cache.js');
const { DopplerLoader } = await import('../../src/loader/doppler-loader.js');

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
  const cache = new ExpertDequantCache();
  cache.setMaxEntries(1);

  const gateUpA = acquireBuffer(16, GPUBufferUsage.STORAGE, 'gate_up_a');
  const downA = acquireBuffer(16, GPUBufferUsage.STORAGE, 'down_a');
  cache.set(0, 0, 'f16', gateUpA, downA);

  const gateUpB = acquireBuffer(16, GPUBufferUsage.STORAGE, 'gate_up_b');
  const downB = acquireBuffer(16, GPUBufferUsage.STORAGE, 'down_b');
  cache.set(0, 1, 'f16', gateUpB, downB);
  await flushDeferredCleanup();

  assert.equal(gateUpA.destroyed, true);
  assert.equal(downA.destroyed, true);
  assert.equal(cache.get(0, 0, 'f16'), undefined);

  cache.clear();
  await flushDeferredCleanup();
  assert.equal(gateUpB.destroyed, true);
  assert.equal(downB.destroyed, true);

  // Each loaded model owns its cache: identical layer/expert/dtype keys never
  // cross models, and unloading one model releases only its entries.
  const modelA = new DopplerLoader();
  const modelB = new DopplerLoader();
  const ownedA = [acquireBuffer(16, GPUBufferUsage.STORAGE, 'a_gate_up'), acquireBuffer(16, GPUBufferUsage.STORAGE, 'a_down')];
  const ownedB = [acquireBuffer(16, GPUBufferUsage.STORAGE, 'b_gate_up'), acquireBuffer(16, GPUBufferUsage.STORAGE, 'b_down')];
  modelA.expertDequantCache.set(0, 3, 'f32', ...ownedA);
  assert.equal(modelB.expertDequantCache.get(0, 3, 'f32'), undefined, 'model B never sees model A dequantized experts');
  modelB.expertDequantCache.set(0, 3, 'f32', ...ownedB);
  await modelA.unload();
  await flushDeferredCleanup();
  assert.equal(ownedA.every(buffer => buffer.destroyed), true, 'unload releases the model\'s dequantized experts');
  assert.equal(modelA.expertDequantCache.get(0, 3, 'f32'), undefined);
  assert.equal(ownedB.some(buffer => buffer.destroyed), false, 'the other model keeps its entries');
  assert.equal(modelB.expertDequantCache.get(0, 3, 'f32').gateUp, ownedB[0]);
  await modelB.unload();
} finally {
  resetRuntime();
}

console.log('moe-cache-cleanup.test: ok');
