import assert from 'node:assert/strict';

// Contract: the QSA indexer cache never publishes state across a reset,
// truncate, restore or destroy that happens while an append or select is in
// flight; destruction waits for pending work before releasing buffers; and an
// allocation failure releases every buffer acquired before it. Fake device:
// lifecycle and ownership only, no numerical execution.
globalThis.GPUBufferUsage = { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
globalThis.GPUShaderStage = { COMPUTE: 2 };

class FakeBuffer {
  constructor({ size, usage, label = null }) {
    Object.assign(this, { size, usage, label, destroyed: false, bytes: null });
  }
  ensureBytes(minSize = this.size) {
    if (!this.bytes || this.bytes.length < minSize) this.bytes = new Uint8Array(minSize);
    return this.bytes;
  }
  destroy() { this.destroyed = true; }
}
globalThis.GPUBuffer = FakeBuffer;

const { setDevice } = await import('../../src/gpu/device.js');
const { createTensor } = await import('../../src/gpu/tensor.js');
const { destroyBufferPool, getBufferPool } = await import('../../src/memory/buffer-pool.js');
const { QsaIndexerCache } = await import('../../src/inference/pipelines/text/indexed-attention/qsa-cache.js');
const { runQsaSelect } = await import('../../src/gpu/kernels/qsa-indexer.js');

function createFakeDevice({ failBufferAt = null } = {}) {
  let buffers = 0;
  const gates = [];
  const device = {
    gates,
    created: [],
    pausePipelines: false,
    queue: {
      submit() {},
      writeBuffer(buffer, offset, data) {
        const bytes = new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength);
        buffer.ensureBytes(offset + bytes.byteLength).set(bytes, offset);
      },
      onSubmittedWorkDone: () => Promise.resolve(),
    },
    features: new Set(['shader-f16', 'subgroups']),
    limits: {
      maxStorageBufferBindingSize: 1 << 30, maxBufferSize: 1 << 30, maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupStorageSize: 32768,
      maxStorageBuffersPerShaderStage: 10, maxUniformBufferBindingSize: 65536, maxComputeWorkgroupsPerDimension: 65535,
    },
    createBindGroup: () => ({}),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    async createComputePipelineAsync() {
      if (device.pausePipelines) await new Promise((resolve) => gates.push(resolve));
      return { getBindGroupLayout: () => ({}) };
    },
    createBuffer(descriptor) {
      buffers += 1;
      if (buffers === failBufferAt) throw new Error(`createBuffer failed at ${buffers}`);
      const buffer = new FakeBuffer(descriptor);
      device.created.push(buffer);
      return buffer;
    },
    createCommandEncoder: () => ({
      beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }),
      copyBufferToBuffer() {},
      finish: () => ({}),
    }),
  };
  return device;
}

const contract = Object.freeze({
  semantics: 'qwen4_exp_qsa', layers: Object.freeze([3, 7]), queryHeads: 4, keyHeads: 1, headDim: 128, rotaryDim: 64,
  tokenBudget: 2048, compressRatio: 4, blockTopK: 512, denseVisibleLimit: 2051, normEps: 1e-6,
  tieOrder: 'lower_block_index', forceSelfToken: false,
});
const fakeTensor = (shape) => createTensor(new FakeBuffer({ size: shape.reduce((a, b) => a * b, 1) * 4, usage: 128 }), 'f32', shape);
const appendInput = (startPos, numTokens) => ({
  projection: fakeTensor([numTokens, 5 * 128]), normWeightOffset: true, qNormWeight: fakeTensor([128]), kNormWeight: fakeTensor([128]),
  rope: { cos: new FakeBuffer({ size: 1 << 20, usage: 128 }), sin: new FakeBuffer({ size: 1 << 20, usage: 128 }) },
  startPos, numTokens,
});
const reset = (device) => {
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
  return device;
};
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));
async function untilPaused(device) {
  for (let i = 0; i < 500 && device.gates.length === 0; i++) await new Promise((resolve) => setTimeout(resolve, 2));
  assert.ok(device.gates.length > 0, 'operation is paused inside GPU pipeline preparation');
}
const openGates = (device) => { device.pausePipelines = false; for (const resolve of device.gates.splice(0)) resolve(); };

// Reset during append: the append fails and the empty cache stays empty.
{
  const device = reset(createFakeDevice());
  const cache = new QsaIndexerCache({ contract, capacity: 64 });
  device.pausePipelines = true;
  const pending = cache.append(null, 3, appendInput(0, 4));
  await untilPaused(device);
  cache.reset();
  openGates(device);
  await assert.rejects(pending, /truncated, reset or restored during append/);
  assert.equal(cache.length(3), 0);
  // The layer accepts a fresh append afterwards.
  const query = await cache.append(null, 3, appendInput(0, 4));
  assert.equal(cache.length(3), 4);
  assert.deepEqual(query.shape, [4, 512]);
  await cache.destroy();
}

// Concurrent appends to one layer are rejected; a restore during append also fails it.
{
  const device = reset(createFakeDevice());
  const cache = new QsaIndexerCache({ contract, capacity: 64 });
  device.pausePipelines = true;
  const first = cache.append(null, 3, appendInput(0, 4));
  await assert.rejects(cache.append(null, 3, appendInput(0, 4)), /another append pending/);
  assert.throws(() => cache.checkpoint(), /append is pending/);
  cache.truncate(0);
  openGates(device);
  await assert.rejects(first, /during append/);
  assert.equal(cache.length(3), 0);
  await cache.destroy();
}

// Destroy during append: buffers are released only after the append settles.
{
  const device = reset(createFakeDevice());
  const cache = new QsaIndexerCache({ contract, capacity: 64 });
  device.pausePipelines = true;
  const pending = cache.append(null, 7, appendInput(0, 8));
  await untilPaused(device);
  const destroyed = cache.destroy();
  await flush();
  assert.equal(getBufferPool().getStats().activeBuffers > 0, true, 'cache buffers stay owned while work is pending');
  openGates(device);
  await assert.rejects(pending, /destroyed during append/);
  await destroyed;
  assert.equal(getBufferPool().getStats().activeBuffers, 0);
  assert.throws(() => cache.length(7), /destroyed/);
  await assert.rejects(cache.append(null, 7, appendInput(0, 4)), /destroyed/);
}

// Constructor allocation failure releases every earlier allocation.
for (const failBufferAt of [1, 2, 3, 4]) {
  reset(createFakeDevice({ failBufferAt }));
  assert.throws(() => new QsaIndexerCache({ contract, capacity: 64 }), /createBuffer failed/);
  assert.equal(getBufferPool().getStats().activeBuffers, 0, `constructor failure at allocation ${failBufferAt}`);
}

// Selection allocation failure releases the buffers acquired before it.
for (const failBufferAt of [1, 2, 3]) {
  reset(createFakeDevice({ failBufferAt }));
  await assert.rejects(runQsaSelect(null, { numTokens: 2, startPos: 0, compressRatio: 4, blockTopK: 512, denseVisibleLimit: 2051 }),
    /createBuffer failed|uniform/);
  assert.equal(getBufferPool().getStats().activeBuffers, 0, `selection failure at allocation ${failBufferAt}`);
}

reset(null);
console.log('qsa-cache-lifecycle.test: ok');
