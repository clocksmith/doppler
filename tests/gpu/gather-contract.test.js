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
const { destroyBufferPool } = await import('../../src/memory/buffer-pool.js');
const { runGather } = await import('../../src/gpu/kernels/gather.js');

function createFakeDevice({ hasF16 = false } = {}) {
  return {
    queue: {
      submit() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    features: new Set(hasF16 ? ['shader-f16'] : []),
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
      throw new Error('createBindGroup should not be reached');
    },
  };
}

function resetRuntimeState(device = null) {
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
}

const fakeBuffer = { size: 64 };

{
  resetRuntimeState(createFakeDevice({ hasF16: false }));
  await assert.rejects(
    () => runGather(fakeBuffer, fakeBuffer, 1, 8, 8, {
      embeddingDtype: 'f16',
      outputDtype: 'f32',
    }),
    /embeddingDtype=f16 requires shader-f16 support/
  );
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ hasF16: false }));
  await assert.rejects(
    () => runGather(fakeBuffer, fakeBuffer, 1, 8, 8, {
      embeddingDtype: 'f32',
      outputDtype: 'f16',
    }),
    /outputDtype=f16 requires shader-f16 support/
  );
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ hasF16: true }));
  await assert.rejects(
    () => runGather(fakeBuffer, fakeBuffer, 1, 6, 8, {
      useVec4: true,
      embeddingDtype: 'f32',
      outputDtype: 'f32',
    }),
    /useVec4=true requires hiddenSize to be divisible by 4/
  );
  resetRuntimeState();
}

console.log('gather-contract.test: ok');
