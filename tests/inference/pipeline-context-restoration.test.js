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

globalThis.GPUShaderStage = {
  COMPUTE: 0x2,
};

const {
  applyPipelineContexts,
  restorePipelineContexts,
} = await import('../../src/inference/pipelines/context.js');
const { getRuntimeConfig, setRuntimeConfig } = await import('../../src/config/runtime.js');
const {
  getDevice,
  getKernelCapabilities,
  getPlatformConfig,
  setDevice,
} = await import('../../src/gpu/device.js');
const { EnergyPipeline } = await import('../../src/inference/pipelines/energy/pipeline.js');

class FakeBuffer {}
globalThis.GPUBuffer = FakeBuffer;

function createFakeDevice(label) {
  return {
    label,
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
    queue: {
      submit() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    createBindGroup(descriptor) {
      return descriptor;
    },
  };
}

{
  const originalRuntime = getRuntimeConfig();
  const originalDevice = createFakeDevice('original');
  setDevice(originalDevice, {
    platformConfig: { platform: { id: 'original-platform', name: 'Original' } },
    adapterInfo: { vendor: 'orig', architecture: 'orig-arch', device: 'orig-device', description: '' },
  });

  const target = {
    gpuContext: { device: originalDevice },
    useGPU: false,
    memoryContext: { id: 'mem-original' },
    storageContext: { id: 'storage-original' },
    baseUrl: 'https://original.example',
    _onProgress: 'original-progress',
  };
  const overrideRuntime = {
    inference: {
      batching: {
        batchSize: 3,
      },
    },
  };
  const scopedDevice = createFakeDevice('scoped');
  const progressHandler = () => {};

  const applied = applyPipelineContexts(target, {
    runtimeConfig: overrideRuntime,
    gpu: { device: scopedDevice },
    memory: { id: 'mem-scoped' },
    storage: { id: 'storage-scoped' },
    baseUrl: 'https://scoped.example',
    onProgress: progressHandler,
  }, {
    assignGpuContext: true,
    assignUseGPU: true,
    assignMemoryContext: true,
    assignStorageContext: true,
  });

  assert.equal(applied.runtimeConfig.inference.batching.batchSize, 3);
  assert.equal(getRuntimeConfig().inference.batching.batchSize, 3);
  assert.equal(getDevice(), scopedDevice);
  assert.deepEqual(target.gpuContext, { device: scopedDevice });
  assert.equal(target.useGPU, true);
  assert.deepEqual(target.memoryContext, { id: 'mem-scoped' });
  assert.deepEqual(target.storageContext, { id: 'storage-scoped' });
  assert.equal(target.baseUrl, 'https://scoped.example');
  assert.equal(target._onProgress, progressHandler);

  const replacementRuntime = {
    inference: {
      batching: {
        batchSize: 5,
      },
    },
  };
  const replacementDevice = createFakeDevice('replacement');
  const replacement = applyPipelineContexts(target, {
    runtimeConfig: replacementRuntime,
    gpu: { device: replacementDevice },
  }, {
    assignGpuContext: true,
    assignUseGPU: true,
  });

  assert.equal(replacement.runtimeConfig.inference.batching.batchSize, 5);
  assert.equal(getRuntimeConfig().inference.batching.batchSize, 5);
  assert.equal(getDevice(), replacementDevice);

  replacement.restore();
  assert.equal(getRuntimeConfig().inference.batching.batchSize, originalRuntime.inference.batching.batchSize);
  assert.equal(getDevice(), originalDevice);
  assert.deepEqual(target.gpuContext, { device: originalDevice });
  assert.equal(target.useGPU, false);
  assert.deepEqual(target.memoryContext, { id: 'mem-original' });
  assert.deepEqual(target.storageContext, { id: 'storage-original' });
  assert.equal(target.baseUrl, 'https://original.example');
  assert.equal(target._onProgress, 'original-progress');

  assert.equal(restorePipelineContexts(target), false);
  setRuntimeConfig(originalRuntime);
  setDevice(null);
}

{
  const originalRuntime = getRuntimeConfig();
  const originalDevice = createFakeDevice('original');
  setDevice(originalDevice, {
    platformConfig: { platform: { id: 'original-platform', name: 'Original' } },
    adapterInfo: { vendor: 'orig', architecture: 'orig-arch', device: 'orig-device', description: '' },
  });

  const pipeline = new EnergyPipeline();
  const scopedDevice = createFakeDevice('energy-scoped');
  await pipeline.initialize({
    runtimeConfig: {
      inference: {
        batching: {
          batchSize: 7,
        },
      },
    },
    gpu: { device: scopedDevice },
  });

  assert.equal(getRuntimeConfig().inference.batching.batchSize, 7);
  assert.equal(getDevice(), scopedDevice);
  assert.equal(pipeline.useGPU, undefined);

  assert.equal(restorePipelineContexts(pipeline), true);
  assert.equal(getRuntimeConfig().inference.batching.batchSize, originalRuntime.inference.batching.batchSize);
  assert.equal(getDevice(), originalDevice);
  assert.equal(restorePipelineContexts(pipeline), false);
  setRuntimeConfig(originalRuntime);
  setDevice(null);
}

{
  const originalRuntime = getRuntimeConfig();
  const originalDevice = createFakeDevice('original');
  const originalPlatformConfig = {
    platform: { id: 'original-platform', name: 'Original' },
  };
  const originalAdapterInfo = {
    vendor: 'orig',
    architecture: 'orig-arch',
    device: 'orig-device',
    description: '',
  };
  setDevice(originalDevice, {
    platformConfig: originalPlatformConfig,
    adapterInfo: originalAdapterInfo,
  });

  const target = {};
  const applied = applyPipelineContexts(target, {
    gpu: { device: originalDevice },
  }, {
    assignGpuContext: true,
  });

  assert.equal(getDevice(), originalDevice);
  assert.deepEqual(getKernelCapabilities().adapterInfo, originalAdapterInfo);
  assert.deepEqual(getPlatformConfig(), originalPlatformConfig);
  assert.deepEqual(target.gpuContext, { device: originalDevice });

  applied.restore();
  assert.equal(getDevice(), originalDevice);
  assert.deepEqual(getKernelCapabilities().adapterInfo, originalAdapterInfo);
  assert.deepEqual(getPlatformConfig(), originalPlatformConfig);
  setRuntimeConfig(originalRuntime);
  setDevice(null);
}

{
  const target = {};
  const storageContextAlias = { id: 'storage-alias' };
  const applied = applyPipelineContexts(target, {
    storageContext: storageContextAlias,
  }, {
    assignStorageContext: true,
  });

  assert.deepEqual(target.storageContext, storageContextAlias);
  applied.restore();
  assert.equal(target.storageContext, undefined);
}

console.log('pipeline-context-restoration.test: ok');
