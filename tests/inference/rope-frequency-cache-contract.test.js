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
const { destroyBufferPool, getBufferPool } = await import('../../src/memory/buffer-pool.js');
const { _initRoPE } = await import('../../src/inference/pipelines/text/lifecycle.js');
const { initRoPEFrequencies } = await import('../../src/inference/pipelines/text/init.js');
const { releaseRoPEFrequencies } = await import('../../src/inference/pipelines/text/init-rope.js');
const { createShaderSourceScope, runWithShaderSourceScope } = await import('../../src/gpu/kernels/shader-source-scope.js');

function createFakeDevice() {
  const createdBuffers = [];
  let writeBufferCount = 0;
  return {
    createdBuffers,
    get writeBufferCount() {
      return writeBufferCount;
    },
    queue: {
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
      submit() {},
      writeBuffer(buffer, offset, data) {
        writeBufferCount += 1;
        buffer.written = new Uint8Array(ArrayBuffer.isView(data) ? data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) : data.slice(0));
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
      const buffer = new FakeBuffer({ size, usage });
      createdBuffers.push(buffer);
      return buffer;
    },
    createBindGroup() {
      return {};
    },
    createShaderModule() {
      return {};
    },
    async createComputePipelineAsync() {
      return {
        getBindGroupLayout() {
          return {};
        },
      };
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

function resetRuntimeState(device = null) {
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
  if (device) {
    getBufferPool().configure({ enablePooling: false });
  }
}

const ropeConfig = {
  headDim: 256,
  localHeadDim: 64,
  rotaryDim: 64,
  ropeLocalRotaryDim: 64,
  ropeFrequencyBaseDim: 64,
  ropeLocalFrequencyBaseDim: 64,
  maxSeqLen: 8,
  ropeTheta: 1000000,
  ropeLocalTheta: null,
  mropeInterleaved: false,
  mropeSection: null,
  partialRotaryFactor: 0.25,
  ropeLocalPartialRotaryFactor: null,
  ropeScale: 1,
  ropeLocalScale: 1,
  ropeScalingType: null,
  ropeLocalScalingType: null,
  ropeScaling: null,
  ropeLocalScaling: null,
};

{
  resetRuntimeState();
  await assert.rejects(
    () => initRoPEFrequencies(ropeConfig, false),
    /requires the declared WebGPU execution path/
  );
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);
  const first = await initRoPEFrequencies(ropeConfig, true);
  const createdBuffersAfterFirst = device.createdBuffers.length;
  const writeBufferCountAfterFirst = device.writeBufferCount;
  const second = await initRoPEFrequencies(ropeConfig, true);
  assert.equal(second.cos, first.cos);
  assert.equal(second.sin, first.sin);
  assert.equal(second.localCos, first.localCos);
  assert.equal(second.localSin, first.localSin);
  assert.equal(device.createdBuffers.length, createdBuffersAfterFirst);
  assert.equal(device.writeBufferCount, writeBufferCountAfterFirst);
  // Warm frequency buffers must not conceal a missing declared Capsule kernel.
  await assert.rejects(
    () => runWithShaderSourceScope(createShaderSourceScope(new Map()),
      () => initRoPEFrequencies(ropeConfig, true)),
    /rope_precompute.wgsl is outside the verified Capsule source closure/
  );
  const restored = await initRoPEFrequencies(ropeConfig, true);
  assert.equal(restored.cos, first.cos);
  assert.equal(restored.sin, first.sin);
  const sourceFrequencies = Array(32).fill(0.5);
  const sourceConfig = { ...ropeConfig, ropeInverseFrequencies: sourceFrequencies };
  const source = await initRoPEFrequencies(sourceConfig, true);
  assert.notEqual(source.cos, first.cos, 'Different source frequencies must not reuse a generated table.');
  const state = { useGPU: true,
    runtimeConfig: { inference: { session: { kvcache: { maxSeqLen: 8 } } } },
    modelConfig: { ...sourceConfig,
    globalHeadDim: ropeConfig.headDim, headDim: ropeConfig.localHeadDim,
    ropeRotaryDim: ropeConfig.rotaryDim } };
  await _initRoPE.call(state);
  assert.equal(state.ropeFreqsCos, source.cos, 'Pipeline load must forward source frequencies.');
  const changed = await initRoPEFrequencies({ ...sourceConfig, ropeInverseFrequencies: Array(32).fill(0.25) }, true);
  assert.notEqual(changed.cos, source.cos, 'Source-frequency bytes belong to the cache identity.');
  resetRuntimeState();
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);
  const config = { ...ropeConfig, maxSeqLen: 8192, positionCount: 8,
    ropeScalingType: 'longrope', ropeLocalScalingType: 'longrope',
    ropeScaling: { original_max_position_embeddings: 64,
      short_factor: Array(32).fill(1), long_factor: Array(32).fill(2) } };
  config.ropeLocalScaling = config.ropeScaling;
  const frequencies = await initRoPEFrequencies(config, true);
  const uniform = device.createdBuffers.find(buffer => (buffer.usage & GPUBufferUsage.UNIFORM) && buffer.written);
  const words = new Uint32Array(uniform.written.buffer);
  assert.equal(words[0], 8192, 'LongRoPE magnitude keeps the declared model context.');
  assert.equal(words[15], 8, 'Only session positions are allocated and dispatched.');
  const factors = device.createdBuffers.find(buffer => buffer.written?.byteLength === 32 * 4);
  assert.ok(new Float32Array(factors.written.buffer).every(value => value === 2),
    'Short allocation must not select short-context factors.');
  for (const positionCount of [0, -1, 0.5, 8193]) {
    await assert.rejects(() => initRoPEFrequencies({ ...config, positionCount }, true), /positionCount/);
  }
  const wider = await initRoPEFrequencies({ ...config, positionCount: 16 }, true);
  assert.notEqual(wider.cos, frequencies.cos, 'Allocation extent belongs to lease identity.');
  const otherContext = await initRoPEFrequencies({ ...config, maxSeqLen: 64 }, true);
  assert.notEqual(otherContext.cos, frequencies.cos, 'Scaling context remains in lease identity.');
  releaseRoPEFrequencies(frequencies);
  releaseRoPEFrequencies(wider);
  releaseRoPEFrequencies(otherContext);
  resetRuntimeState();
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);
  const first = await initRoPEFrequencies(ropeConfig, true);
  const createdBuffersAfterFirst = device.createdBuffers.length;
  const writeBufferCountAfterFirst = device.writeBufferCount;

  destroyBufferPool();
  getBufferPool().configure({ enablePooling: false });

  const second = await initRoPEFrequencies(ropeConfig, true);
  assert.equal(first.cos.destroyed, true);
  assert.equal(first.sin.destroyed, true);
  assert.notEqual(second.cos, first.cos);
  assert.notEqual(second.sin, first.sin);
  assert.equal(second.cos.destroyed, false);
  assert.equal(second.sin.destroyed, false);
  assert.equal(second.localCos, null);
  assert.equal(second.localSin, null);
  assert.ok(device.createdBuffers.length > createdBuffersAfterFirst);
  assert.ok(device.writeBufferCount > writeBufferCountAfterFirst);
  resetRuntimeState();
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);
  const first = await initRoPEFrequencies(ropeConfig, true);
  const second = await initRoPEFrequencies(ropeConfig, true);
  assert.notEqual(first, second, 'Each acquisition owns a distinct lease.');
  releaseRoPEFrequencies(first);
  releaseRoPEFrequencies(first);
  assert.equal(second.cos.destroyed, false, 'Another session still owns these tables.');
  // Release must use the originating pool even after a different device is bound.
  const otherDevice = createFakeDevice();
  setDevice(otherDevice, { platformConfig: null });
  releaseRoPEFrequencies(second);
  await device.queue.onSubmittedWorkDone();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(second.cos.destroyed, true);
  assert.equal(second.sin.destroyed, true);
  setDevice(device, { platformConfig: null });
  const reopened = await initRoPEFrequencies(ropeConfig, true);
  assert.notEqual(reopened.cos, first.cos);
  releaseRoPEFrequencies(reopened);
  resetRuntimeState();
}

for (const modelPartition of [null, { index: 0 }]) {
  const device = createFakeDevice();
  resetRuntimeState(device);
  const state = { useGPU: true, modelPartition,
    runtimeConfig: { inference: { session: { kvcache: { maxSeqLen: 8 } } } },
    modelConfig: { ...ropeConfig, maxSeqLen: 8192, globalHeadDim: ropeConfig.headDim,
      headDim: ropeConfig.localHeadDim, ropeRotaryDim: ropeConfig.rotaryDim } };
  await _initRoPE.call(state);
  const bounded = await initRoPEFrequencies({ ...ropeConfig, maxSeqLen: 8192, positionCount: 8 }, true);
  assert.equal(state.ropeFreqsCos, bounded.cos,
    'Standalone and partition tables cover the session while retaining the model scaling context.');
  assert.ok(state.ropeFreqsCos.size <= 8 * (ropeConfig.rotaryDim / 2) * 4);
  assert.equal(state.modelConfig.maxSeqLen, 8192, 'Model contract remains unchanged.');
  const existing = state.ropeFrequencyLease;
  for (const maxSeqLen of [undefined, null, 0, -1, 1.5]) {
    state.runtimeConfig.inference.session.kvcache.maxSeqLen = maxSeqLen;
    await assert.rejects(() => _initRoPE.call(state), /positive session maxSeqLen/);
    assert.equal(state.ropeFrequencyLease, existing, 'Invalid replacement preserves the live lease.');
  }
  releaseRoPEFrequencies(existing);
  releaseRoPEFrequencies(bounded);
  resetRuntimeState();
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);
  await assert.rejects(() => initRoPEFrequencies({
    ...ropeConfig, ropeLocalTheta: 10000, ropeLocalScalingType: 'unsupported',
  }, true));
  assert.equal(getBufferPool().getStats().activeBuffers, 0,
    'Failed local-table preparation releases completed global tables.');
  resetRuntimeState();
}

console.log('rope-frequency-cache-contract.test: ok');

// Explicit non-local layer schedules must not allocate unreachable local tables.
{
  const device = createFakeDevice();
  resetRuntimeState(device);
  const config = { ...ropeConfig, localHeadDim: 256, ropeLocalRotaryDim: 256,
    ropeLocalFrequencyBaseDim: 256, layerTypes: ['linear_attention', 'full_attention'] };
  const globalOnly = await initRoPEFrequencies(config, true);
  assert.equal(globalOnly.localCos, null);
  assert.equal(globalOnly.localSin, null);
  const state = { useGPU: true,
    runtimeConfig: { inference: { session: { kvcache: { maxSeqLen: config.maxSeqLen } } } },
    modelConfig: { ...config, globalHeadDim: config.headDim,
      ropeRotaryDim: config.rotaryDim } };
  await _initRoPE.call(state);
  assert.equal(state.ropeLocalCos, null, 'Pipeline forwards the declared schedule.');
  assert.equal(state.ropeFreqsCos, globalOnly.cos);
  releaseRoPEFrequencies(state.ropeFrequencyLease);
  for (const layerType of ['sliding_attention', 'local_attention', 'local', 'sliding']) {
    const withLocal = await initRoPEFrequencies({ ...config, layerTypes: ['full_attention', layerType] }, true);
    assert.ok(withLocal.localCos);
    assert.ok(withLocal.localSin);
    assert.notEqual(withLocal.cos, globalOnly.cos, 'Cache identity includes local-table demand.');
    releaseRoPEFrequencies(withLocal);
  }
  releaseRoPEFrequencies(globalOnly);
  resetRuntimeState();
}
