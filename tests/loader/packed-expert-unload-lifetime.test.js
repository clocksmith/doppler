import assert from 'node:assert/strict';

globalThis.GPUBufferUsage = {
  MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128,
};

class FakeBuffer {
  constructor({ size, usage }) {
    this.size = size;
    this.usage = usage;
    this.destroyCalls = 0;
  }

  destroy() {
    this.destroyCalls++;
  }
}

globalThis.GPUBuffer = FakeBuffer;

const { setDevice } = await import('../../src/gpu/device.js');
const { acquireBuffer, destroyBufferPool, getBufferPool, isBufferActive } = await import('../../src/memory/buffer-pool.js');
const { DopplerLoader } = await import('../../src/loader/doppler-loader.js');

let complete;
const fence = new Promise(resolve => { complete = resolve; });
setDevice({
  lost: new Promise(() => {}),
  features: new Set(),
  limits: {
    maxStorageBufferBindingSize: 1 << 20,
    maxBufferSize: 1 << 20,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupStorageSize: 16384,
  },
  queue: { submit() {}, onSubmittedWorkDone: () => fence },
  createBuffer: descriptor => new FakeBuffer(descriptor),
  createBindGroup: descriptor => descriptor,
}, { platformConfig: null });
getBufferPool().configure({ enablePooling: false });

const modelA = new DopplerLoader();
const modelB = new DopplerLoader();
try {
  const gateA = acquireBuffer(16, GPUBufferUsage.STORAGE, 'packed_gate_a');
  const downA = acquireBuffer(16, GPUBufferUsage.STORAGE, 'packed_down_a');
  const gateB = acquireBuffer(16, GPUBufferUsage.STORAGE, 'packed_gate_b');
  const downB = acquireBuffer(16, GPUBufferUsage.STORAGE, 'packed_down_b');
  const packedA = { gateUpBlocks: gateA, downBlocks: downA };
  modelA.gpuBuffers.add(gateA);
  modelA.gpuBuffers.add(downA);
  modelA.experts.set('layer_0_expert_0', packedA);
  modelA.experts.set('layer_0_expert_1', packedA);
  modelB.experts.set('layer_0_expert_0', { gateUpBlocks: gateB, downBlocks: downB });

  await modelA.unload();
  assert.equal(isBufferActive(gateA), false);
  assert.equal(isBufferActive(gateB), true);
  assert(getBufferPool().getStats().resources.deferredCleanup.bytes > 0);
  assert.deepEqual([gateA.destroyCalls, downA.destroyCalls], [0, 0],
    'alias cleanup must not bypass the owning pool retirement fence');

  complete();
  await getBufferPool().settleDeferredDestruction();
  assert.deepEqual([gateA.destroyCalls, downA.destroyCalls], [1, 1]);
  assert.deepEqual([gateB.destroyCalls, downB.destroyCalls], [0, 0],
    'unloading another model preserves this model\'s buffers');

  await modelB.unload();
  await getBufferPool().settleDeferredDestruction();
  assert.deepEqual([gateB.destroyCalls, downB.destroyCalls], [1, 1]);
  assert.equal(getBufferPool().getStats().currentBytesAllocated, 0);
  assert.equal(getBufferPool().getStats().resources.deferredCleanup.bytes, 0);

  const unpooled = new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  const modelC = new DopplerLoader();
  const packedC = { gateUpBlocks: unpooled, downBlocks: unpooled };
  modelC.experts.set('layer_0_expert_0', packedC);
  modelC.experts.set('layer_0_expert_1', packedC);
  await modelC.unload();
  assert.equal(unpooled.destroyCalls, 1, 'unpooled aliases also release exactly once');
} finally {
  complete();
  await modelA.unload();
  await modelB.unload();
  await getBufferPool().settleDeferredDestruction();
  destroyBufferPool();
  setDevice(null);
}

console.log('packed-expert-unload-lifetime.test: ok');
