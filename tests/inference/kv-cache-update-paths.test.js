import assert from 'node:assert/strict';

globalThis.GPUBufferUsage = {
  MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008, INDEX: 0x0010,
  VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080, INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
};
globalThis.GPUShaderStage = { COMPUTE: 0x2 };

const { CommandRecorder } = await import('../../src/gpu/command-recorder.js');
const { FEATURES, setDevice } = await import('../../src/gpu/device.js');
const { KVCache } = await import('../../src/inference/kv-cache/base.js');
const { SlidingWindowKVCache } = await import('../../src/inference/kv-cache/sliding-window.js');
const { destroyBufferPool, getBufferPool } = await import('../../src/memory/buffer-pool.js');

class FakeBuffer {
  constructor({ label = null, size, usage }) { this.label = label; this.size = size; this.usage = usage; this.destroyed = false; }
  destroy() { this.destroyed = true; }
}
const ORIGINAL_GPU_BUFFER = globalThis.GPUBuffer;
globalThis.GPUBuffer = FakeBuffer;

// Immediate and recorded writes must encode the same copies; the fake device
// records every copy and every submit so both paths can be compared.
function createFakeDevice() {
  const copyOps = [], submits = [];
  const encoder = {
    beginComputePass() { return { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }; },
    copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
      copyOps.push({ source: source.label, sourceOffset, destination: destination.label, destinationOffset, size });
    },
    finish() { return { copies: copyOps.length }; },
  };
  return {
    copyOps, submits,
    lost: new Promise(() => {}),
    queue: { submit(buffers) { submits.push(buffers.length); }, writeBuffer() {}, onSubmittedWorkDone() { return Promise.resolve(); } },
    features: new Set([FEATURES.SHADER_F16]),
    limits: { maxStorageBufferBindingSize: 1 << 20, maxBufferSize: 1 << 20, maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupSizeY: 1, maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupStorageSize: 16384, maxStorageBuffersPerShaderStage: 8, maxUniformBufferBindingSize: 65536,
      maxComputeWorkgroupsPerDimension: 65535, maxQuerySetSize: 64 },
    createBuffer({ label, size, usage }) { return new FakeBuffer({ label, size, usage }); },
    createBindGroup(descriptor) { return descriptor; },
    createBindGroupLayout() { return {}; },
    createPipelineLayout() { return {}; },
    createShaderModule() { return { async getCompilationInfo() { return { messages: [] }; } }; },
    async createComputePipelineAsync() { return { getBindGroupLayout() { return {}; } }; },
    createCommandEncoder() { return encoder; },
    destroy() {},
  };
}
function resetRuntime(device) {
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
  getBufferPool().configure({ enablePooling: false });
}
const source = (label, elements) => new FakeBuffer({ label, size: elements * 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const config = { numLayers: 2, numHeads: 2, headDim: 4, maxSeqLen: 8, useGPU: true, layout: 'contiguous', pageSize: 8, kvDtype: 'f16' };
const kvSize = config.numHeads * config.headDim;
const metadata = (cache, layerIdx) => ({ seqLen: cache.layers[layerIdx].seqLen, currentSeqLen: cache.currentSeqLen, totalTokensSeen: cache.totalTokensSeen, tokensWritten: cache.counters?.tokensWritten ?? null });
const labelled = (cache, prefix) => { cache.layers.forEach((layer, index) => { layer.keysGPU.label = `${prefix}-keys-${index}`; layer.valuesGPU.label = `${prefix}-values-${index}`; }); return cache; };

try {
  const device = createFakeDevice();
  resetRuntime(device);

  // Base cache: immediate and recorded writes produce the same copies and bookkeeping.
  const immediate = labelled(new KVCache(config), 'cache');
  const recorded = labelled(new KVCache(config), 'cache');
  const keys = source('keys', 3 * kvSize), values = source('values', 3 * kvSize);
  immediate.updateFromGPU(1, keys, values, 2, 3);
  const immediateOps = device.copyOps.splice(0), immediateSubmits = device.submits.splice(0);
  const recorder = new CommandRecorder(device, 'kv_paths');
  recorded.recordUpdateFromGPU(recorder, 1, keys, values, 2, 3);
  const recordedOps = device.copyOps.splice(0);
  assert.deepEqual(recordedOps, immediateOps, 'recorded copies match immediate copies');
  assert.deepEqual(immediateOps.map((op) => [op.destination, op.destinationOffset, op.size]), [['cache-keys-1', 2 * kvSize * 2, 3 * kvSize * 2], ['cache-values-1', 2 * kvSize * 2, 3 * kvSize * 2]]);
  assert.deepEqual(immediateSubmits, [1], 'the immediate path submits once');
  assert.deepEqual(device.submits, [], 'the recorded path leaves submission to the recorder');
  assert.deepEqual(metadata(recorded, 1), metadata(immediate, 1));
  assert.deepEqual(metadata(immediate, 1), { seqLen: 5, currentSeqLen: 5, totalTokensSeen: 5, tokensWritten: 3 });
  assert.equal(immediate.counters.gpuUpdateCalls, 1);
  assert.equal(recorded.counters.recordedGpuUpdateCalls, 1);
  assert.equal(immediate.layers[0].seqLen, 0, 'other layers are untouched');
  immediate.updateFromGPU(1, keys, values, 0, 0);
  assert.deepEqual(metadata(immediate, 1), { seqLen: 5, currentSeqLen: 5, totalTokensSeen: 5, tokensWritten: 3 }, 'zero tokens is a no-op');
  assert.throws(() => immediate.updateFromGPU(1, keys, values, 6, 3), /Cache overflow: 9 > 8/);
  assert.throws(() => recorded.recordUpdateFromGPU(recorder, 1, keys, values, 0, 4), /KVCache recordUpdateFromGPU buffer size is smaller than requested write/);
  assert.throws(() => immediate.updateFromGPU(1, keys, values, 0, 4), /KVCache updateFromGPU buffer size is smaller than requested write/);
  assert.throws(() => immediate.updateFromGPU(1, keys, values, 0, -1), /non-negative integer token count/);
  device.copyOps.length = 0;

  // Sliding window: the ring plan is shared, so a wrapping write encodes the same two chunks either way.
  const ring = { ...config, windowSize: 4, maxSeqLen: 4 };
  const immediateRing = labelled(new SlidingWindowKVCache(ring), 'ring');
  const recordedRing = labelled(new SlidingWindowKVCache(ring), 'ring');
  const ringKeys = source('ring-keys', 3 * kvSize), ringValues = source('ring-values', 3 * kvSize);
  immediateRing.updateFromGPU(1, ringKeys, ringValues, 2, 3);
  const ringImmediate = device.copyOps.splice(0);
  recordedRing.recordUpdateFromGPU(recorder, 1, ringKeys, ringValues, 2, 3);
  const ringRecorded = device.copyOps.splice(0);
  assert.deepEqual(ringRecorded, ringImmediate);
  const bytesPerToken = kvSize * 2;
  assert.deepEqual(ringImmediate.map((op) => [op.destination, op.sourceOffset, op.destinationOffset, op.size]), [
    ['ring-keys-1', 0, 2 * bytesPerToken, 2 * bytesPerToken], ['ring-values-1', 0, 2 * bytesPerToken, 2 * bytesPerToken],
    ['ring-keys-1', 2 * bytesPerToken, 0, bytesPerToken], ['ring-values-1', 2 * bytesPerToken, 0, bytesPerToken],
  ], 'two tokens fill the window end, the third wraps to slot zero');
  assert.deepEqual(metadata(recordedRing, 1), metadata(immediateRing, 1));
  assert.deepEqual(metadata(immediateRing, 1), { seqLen: 4, currentSeqLen: 4, totalTokensSeen: 5, tokensWritten: 0 });

  // More tokens than the window: only the newest windowSize tokens are copied, from a source offset.
  const wide = labelled(new SlidingWindowKVCache(ring), 'wide');
  const wideKeys = source('wide-keys', 6 * kvSize), wideValues = source('wide-values', 6 * kvSize);
  wide.updateFromGPU(0, wideKeys, wideValues, 0, 6);
  const wideOps = device.copyOps.splice(0);
  assert.deepEqual(wideOps.map((op) => [op.destination, op.sourceOffset, op.destinationOffset, op.size]), [
    ['wide-keys-0', 2 * bytesPerToken, 2 * bytesPerToken, 2 * bytesPerToken], ['wide-values-0', 2 * bytesPerToken, 2 * bytesPerToken, 2 * bytesPerToken],
    ['wide-keys-0', 4 * bytesPerToken, 0, 2 * bytesPerToken], ['wide-values-0', 4 * bytesPerToken, 0, 2 * bytesPerToken],
  ]);
  assert.equal(wide.totalTokensSeen, 6);
  assert.throws(() => wide.updateFromGPU(0, source('short', kvSize), source('short', kvSize), 0, 2), /SlidingWindowKVCache updateFromGPU buffer size is smaller than requested write/);
  assert.throws(() => wide.recordUpdateFromGPU(recorder, 0, ringKeys, ringValues, 0, 1.5), /SlidingWindowKVCache recordUpdateFromGPU requires a non-negative integer token count/);
  console.log('kv-cache-update-paths: immediate and recorded writes share encoding and bookkeeping for base and sliding caches');
} finally {
  destroyBufferPool();
  setDevice(null, { platformConfig: null });
  globalThis.GPUBuffer = ORIGINAL_GPU_BUFFER;
}
