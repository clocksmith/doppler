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

const { MoERouter } = await import('../../src/inference/moe-router.js');
const { InferencePipeline } = await import('../../src/inference/pipelines/text.js');
const { setDevice } = await import('../../src/gpu/device.js');

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
    },
    createBuffer({ size, usage }) {
      const buffer = new FakeBuffer({ size, usage });
      createdBuffers.push(buffer);
      return buffer;
    },
    createBindGroup() {
      return {};
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

function resetDevice(device = null) {
  setDevice(device, { platformConfig: null });
}

{
  const router = new MoERouter({
    numExperts: 2,
    topK: 1,
    hiddenSize: 2,
    normalizeWeights: true,
  });
  const oldWeight = new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  const oldBias = new FakeBuffer({ size: 8, usage: GPUBufferUsage.STORAGE });
  router._gateWeightGPU = oldWeight;
  router._gateBiasGPU = oldBias;

  router.loadWeights(new Float32Array([1, 2, 3, 4]), new Float32Array([0.1, 0.2]));

  assert.equal(oldWeight.destroyed, true);
  assert.equal(oldBias.destroyed, true);
  assert.equal(router._gateWeightGPU, null);
  assert.equal(router._gateBiasGPU, null);
}

{
  const router = new MoERouter({
    numExperts: 2,
    topK: 1,
    hiddenSize: 2,
    normalizeWeights: true,
  });
  const cachedWeight = new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  const cachedBias = new FakeBuffer({ size: 8, usage: GPUBufferUsage.STORAGE });
  router._gateWeightGPU = cachedWeight;
  router._gateBiasGPU = cachedBias;

  router.destroy();

  assert.equal(cachedWeight.destroyed, true);
  assert.equal(cachedBias.destroyed, true);
  assert.equal(router._gateWeightGPU, null);
  assert.equal(router._gateBiasGPU, null);
  assert.equal(router.gateWeight, null);
  assert.equal(router.gateBias, null);
}

{
  const device = createFakeDevice({ writeBufferThrowAt: 1 });
  resetDevice(device);

  const router = new MoERouter({
    numExperts: 2,
    topK: 1,
    hiddenSize: 2,
    normalizeWeights: true,
  });
  router.loadWeights(new Float32Array([1, 2, 3, 4]));

  await assert.rejects(
    () => router.computeRouterLogitsGPU(new FakeBuffer({ size: 8, usage: GPUBufferUsage.STORAGE }), 1),
    /writeBuffer failed at 1/
  );

  assert.equal(device.createdBuffers.length, 1);
  assert.equal(device.createdBuffers[0].destroyed, true);
  assert.equal(router._gateWeightGPU, null);
  resetDevice();
}

{
  const device = createFakeDevice({ writeBufferThrowAt: 1 });
  resetDevice(device);

  const router = new MoERouter({
    numExperts: 2,
    topK: 1,
    hiddenSize: 2,
    normalizeWeights: true,
  });
  router.gateBias = new Float32Array([0.1, 0.2]);

  await assert.rejects(
    () => router._getGateBiasBuffer(device),
    /writeBuffer failed at 1/
  );

  assert.equal(device.createdBuffers.length, 1);
  assert.equal(device.createdBuffers[0].destroyed, true);
  assert.equal(router._gateBiasGPU, null);
  resetDevice();
}

{
  const device = createFakeDevice({ writeBufferThrowAt: 1 });
  resetDevice(device);

  const router = new MoERouter({
    numExperts: 2,
    topK: 1,
    hiddenSize: 2,
    normalizeWeights: true,
  });
  router._getBiasAddPipeline = () => ({
    getBindGroupLayout() {
      return {};
    },
  });

  await assert.rejects(
    () => router._addBiasInPlace(
      new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE }),
      new FakeBuffer({ size: 8, usage: GPUBufferUsage.STORAGE }),
      1,
      device,
      'f32',
      'f32'
    ),
    /writeBuffer failed at 1/
  );

  assert.equal(device.createdBuffers.length, 1);
  assert.equal(device.createdBuffers[0].destroyed, true);
  resetDevice();
}

{
  const pipeline = new InferencePipeline();
  let destroyCalls = 0;
  pipeline.moeRouter = {
    destroy() {
      destroyCalls += 1;
    },
  };

  await pipeline.unload();
  assert.equal(destroyCalls, 1);
  assert.equal(pipeline.moeRouter, null);
}

console.log('moe-router-cleanup.test: ok');
