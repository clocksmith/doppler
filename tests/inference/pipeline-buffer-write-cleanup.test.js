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

const { setDevice } = await import('../../src/gpu/device.js');
const { destroyBufferPool, getBufferPool } = await import('../../src/memory/buffer-pool.js');
const { EnergyRowHeadPipeline } = await import('../../src/inference/pipelines/energy-head/row-head-pipeline.js');
const { EnergyPipeline } = await import('../../src/inference/pipelines/energy/pipeline.js');
const { createDiffusionIndexBuffer } = await import('../../src/inference/pipelines/diffusion/helpers.js');
const { getWeightBuffer, getNormWeightBuffer } = await import('../../src/inference/pipelines/text/weights.js');
const { initRoPEFrequencies } = await import('../../src/inference/pipelines/text/init.js');

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

function createFakeDevice({ writeBufferThrowAt = null } = {}) {
  let writeBufferCount = 0;
  const createdBuffers = [];

  return {
    createdBuffers,
    queue: {
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
      submit() {},
      writeBuffer() {
        writeBufferCount += 1;
        if (writeBufferCount === writeBufferThrowAt) {
          throw new Error(`writeBuffer failed at ${writeBufferCount}`);
        }
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
  };
}

function resetRuntimeState(device = null) {
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
  if (device) {
    getBufferPool().configure({ enablePooling: false });
  }
}

function assertPoolIsClean() {
  const stats = getBufferPool().getStats();
  assert.equal(stats.activeBuffers, 0);
  assert.equal(stats.currentBytesAllocated, 0);
}

{
  const device = createFakeDevice({ writeBufferThrowAt: 1 });
  resetRuntimeState(device);

  const pipeline = new EnergyRowHeadPipeline();
  await pipeline.initialize({ runtimeConfig: {} });
  await pipeline.loadModel({
    modelType: 'dream_energy_head',
    modelId: 'dream-energy-head-test',
    featureIds: ['f0', 'f1'],
    weights: [0.5, -0.25],
    bias: 0,
  });

  await assert.rejects(
    () => pipeline.scoreRows({
      backend: 'gpu',
      rows: [{ rowId: 'r0', features: [1, 2] }],
    }),
    /writeBuffer failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ writeBufferThrowAt: 1 });
  resetRuntimeState(device);

  const pipeline = new EnergyPipeline();
  await pipeline.initialize({ runtimeConfig: { inference: { energy: {} } } });
  await pipeline.loadModel({
    modelType: 'energy',
    modelId: 'energy-test',
    energy: {
      shape: [1, 1, 1],
    },
  });

  const result = await pipeline.generate({
    problem: 'quintel',
    quintel: {
      backend: 'gpu',
      size: 1,
    },
  });
  assert.equal(result.backend, 'CPU');
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ writeBufferThrowAt: 1 });
  resetRuntimeState(device);

  assert.throws(
    () => createDiffusionIndexBuffer(device, new Uint32Array([1, 2, 3]), 'diffusion_idx'),
    /writeBuffer failed at 1/
  );
  assert.equal(device.createdBuffers.length, 1);
  assert.equal(device.createdBuffers[0].destroyed, true);
  resetRuntimeState();
}

{
  const device = createFakeDevice({ writeBufferThrowAt: 1 });
  resetRuntimeState(device);

  assert.throws(
    () => getWeightBuffer(new Float32Array([1, 2, 3, 4]), 'weight_upload'),
    /writeBuffer failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ writeBufferThrowAt: 1 });
  resetRuntimeState(device);

  assert.throws(
    () => getNormWeightBuffer(new Float32Array([1, 2, 3, 4]), 'norm_upload'),
    /writeBuffer failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ writeBufferThrowAt: 2 });
  resetRuntimeState(device);

  await assert.rejects(
    () => initRoPEFrequencies({
      headDim: 256,
      rotaryDim: 64,
      ropeFrequencyBaseDim: 64,
      ropeLocalFrequencyBaseDim: 256,
      maxSeqLen: 8,
      ropeTheta: 10000000,
      ropeLocalTheta: null,
      mropeInterleaved: true,
      mropeSection: [11, 11, 10],
      partialRotaryFactor: 0.25,
      ropeScale: 1,
      ropeLocalScale: 1,
      ropeScalingType: null,
      ropeLocalScalingType: null,
      ropeScaling: null,
      ropeLocalScaling: null,
    }, true),
    /writeBuffer failed at 2/
  );
  assertPoolIsClean();
  for (const buffer of device.createdBuffers) {
    assert.equal(buffer.destroyed, true);
  }
  resetRuntimeState();
}

console.log('pipeline-buffer-write-cleanup.test: ok');
