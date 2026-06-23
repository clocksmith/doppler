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
const { runGather, runGatherSplit, runGatherSplit4, runGatherSplit8 } = await import('../../src/gpu/kernels/gather.js');

function createFakeDevice({ hasF16 = false, maxStorageBuffersPerShaderStage = 8 } = {}) {
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
      maxStorageBuffersPerShaderStage,
      maxUniformBufferBindingSize: 65536,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    createBindGroup() {
      throw new Error('createBindGroup should not be reached');
    },
    createShaderModule() {
      return {};
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

{
  resetRuntimeState(createFakeDevice({ hasF16: false, maxStorageBuffersPerShaderStage: 10 }));
  await assert.rejects(
    () => runGatherSplit8(fakeBuffer, {
      kind: 'split_weight_buffer',
      sections: [{ buffer: fakeBuffer, rowStart: 0, rowCount: 8 }],
      dtype: 'f16',
      layout: 'row',
      shape: [8, 8],
    }, 1, 8, 8, {
      embeddingDtype: 'f16',
      outputDtype: 'f16',
    }),
    /gather_split8 requires shader-f16 support/
  );
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ hasF16: true, maxStorageBuffersPerShaderStage: 8 }));
  await assert.rejects(
    () => runGatherSplit(fakeBuffer, {
      kind: 'split_weight_buffer',
      sections: [
        { buffer: fakeBuffer, rowStart: 0, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 1, rowCount: 1 },
      ],
      dtype: 'f16',
      layout: 'row',
      shape: [2, 8],
      metadata: {
        splitGatherSectionCount: 8,
      },
    }, 1, 8, 2, {
      embeddingDtype: 'f16',
      outputDtype: 'f16',
    }),
    /gather_split8 requires 10 storage buffers per shader stage/
  );
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ hasF16: true, maxStorageBuffersPerShaderStage: 8 }));
  await assert.rejects(
    () => runGatherSplit8(fakeBuffer, {
      kind: 'split_weight_buffer',
      sections: [
        { buffer: fakeBuffer, rowStart: 0, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 1, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 2, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 3, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 4, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 5, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 6, rowCount: 1 },
      ],
      dtype: 'f16',
      layout: 'row',
      shape: [7, 8],
    }, 1, 8, 7, {
      embeddingDtype: 'f16',
      outputDtype: 'f32',
    }),
    /gather_split8 requires 10 storage buffers per shader stage/
  );
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ hasF16: true, maxStorageBuffersPerShaderStage: 10 }));
  await assert.rejects(
    () => runGatherSplit8(fakeBuffer, {
      kind: 'split_weight_buffer',
      sections: [
        { buffer: fakeBuffer, rowStart: 0, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 1, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 2, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 3, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 4, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 5, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 6, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 7, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 8, rowCount: 1 },
      ],
      dtype: 'f16',
      layout: 'row',
      shape: [9, 8],
    }, 1, 8, 9, {
      embeddingDtype: 'f16',
      outputDtype: 'f16',
    }),
    /supports at most 8/
  );
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ hasF16: false }));
  await assert.rejects(
    () => runGatherSplit4(fakeBuffer, {
      kind: 'split_weight_buffer',
      sections: [{ buffer: fakeBuffer, rowStart: 0, rowCount: 8 }],
      dtype: 'f16',
      layout: 'row',
      shape: [8, 8],
    }, 1, 8, 8, {
      embeddingDtype: 'f16',
      outputDtype: 'f16',
    }),
    /gather_split4 requires shader-f16 support/
  );
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ hasF16: true }));
  await assert.rejects(
    () => runGatherSplit4(fakeBuffer, {
      kind: 'split_weight_buffer',
      sections: [
        { buffer: fakeBuffer, rowStart: 0, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 1, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 2, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 3, rowCount: 1 },
        { buffer: fakeBuffer, rowStart: 4, rowCount: 1 },
      ],
      dtype: 'f16',
      layout: 'row',
      shape: [5, 8],
    }, 1, 8, 5, {
      embeddingDtype: 'f16',
      outputDtype: 'f16',
    }),
    /supports at most 4/
  );
  resetRuntimeState();
}

console.log('gather-contract.test: ok');
