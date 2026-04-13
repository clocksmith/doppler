import assert from 'node:assert/strict';

globalThis.GPUBufferUsage ??= {
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
const {
  ensurePleScaledProjectionNormWeight,
  destroyPleRuntimeCache,
  scalePerLayerProjectionNormWeights,
  inferPleProjectionNormDtype,
  loadRangeBackedPleProjectionSliceBytes,
} = await import('../../src/inference/pipelines/text/per-layer-inputs.js');

class FakeBuffer {
  constructor({ size, usage, label = '' }) {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.destroyed = false;
    this._arrayBuffer = new ArrayBuffer(size);
  }

  destroy() {
    this.destroyed = true;
  }
}

function writeBytes(targetBuffer, offset, data) {
  const source = data instanceof ArrayBuffer
    ? new Uint8Array(data)
    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  new Uint8Array(targetBuffer._arrayBuffer, offset, source.byteLength).set(source);
}

function createFakeDevice() {
  return {
    lost: new Promise(() => {}),
    queue: {
      submit() {},
      writeBuffer(buffer, offset, data) {
        writeBytes(buffer, offset, data);
      },
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
      minStorageBufferOffsetAlignment: 16,
    },
    createBuffer({ size, usage, label }) {
      return new FakeBuffer({ size, usage, label });
    },
    createBindGroup() {
      return {};
    },
    createShaderModule() {
      return {};
    },
    createCommandEncoder() {
      return {
        finish() {
          return {};
        },
      };
    },
  };
}

function resetRuntime(device = null) {
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
  if (device) {
    getBufferPool().configure({ enablePooling: false });
  }
}

function createPleContext(perLayerInputWeights, rmsNormWeightOffset = false) {
  return {
    config: {
      hiddenSizePerLayerInput: 4,
    },
    weights: new Map([
      ['per_layer_inputs', perLayerInputWeights],
    ]),
    weightConfig: {
      rmsNormWeightOffset,
    },
    debugFlags: {},
  };
}

function assertCloseArray(actual, expected, message) {
  assert.equal(actual.length, expected.length, `${message}: length mismatch`);
  for (let i = 0; i < actual.length; i++) {
    assert.ok(
      Math.abs(actual[i] - expected[i]) < 1e-6,
      `${message}: index ${i} expected ${expected[i]}, got ${actual[i]}`
    );
  }
}

try {
  const combineScale = 2 ** -0.5;
  const sourceValues = new Float32Array([1, 2, 3, 4]);
  const scaledValues = scalePerLayerProjectionNormWeights(sourceValues, combineScale, false);
  assertCloseArray(
    Array.from(scaledValues),
    Array.from(sourceValues, value => value * combineScale),
    'scalePerLayerProjectionNormWeights should apply the fixed PLE combine scale'
  );
  assert.deepEqual(
    Array.from(sourceValues),
    [1, 2, 3, 4],
    'scalePerLayerProjectionNormWeights should not mutate the source norm weights'
  );
  assert.equal(
    scalePerLayerProjectionNormWeights(sourceValues, combineScale, true),
    null,
    'offset-based RMSNorm weights must keep the legacy post-norm scale path'
  );

  resetRuntime(createFakeDevice());

  const rawF16NormBuffer = new FakeBuffer({
    size: 4 * 2,
    usage: GPUBufferUsage.STORAGE,
    label: 'raw_f16_norm',
  });
  assert.equal(
    inferPleProjectionNormDtype(rawF16NormBuffer, 4),
    'f16',
    'raw GPU norm buffers should infer f16 from manifest-sized byte counts'
  );

  const rawF32NormBuffer = new FakeBuffer({
    size: 4 * 4,
    usage: GPUBufferUsage.STORAGE,
    label: 'raw_f32_norm',
  });
  assert.equal(
    inferPleProjectionNormDtype(rawF32NormBuffer, 4),
    'f32',
    'raw GPU norm buffers should infer f32 from manifest-sized byte counts'
  );

  const perLayerInputWeights = {
    embedTokensPerLayer: null,
    perLayerModelProjection: null,
    perLayerProjectionNorm: sourceValues,
  };
  const context = createPleContext(perLayerInputWeights, false);
  const cachedWeightA = await ensurePleScaledProjectionNormWeight(context, combineScale);
  assert.ok(cachedWeightA, 'Gemma 4 raw RMSNorm weights should build a cached scaled norm tensor');
  assert.equal(
    cachedWeightA.dtype,
    'f32',
    'CPU-backed cached norm tensors should preserve a valid tensor dtype'
  );
  assert.deepEqual(
    Array.from(cachedWeightA.shape),
    [4],
    'CPU-backed cached norm tensors should preserve the per-layer hidden shape'
  );
  const cachedBufferViewA = new Float32Array(cachedWeightA.buffer._arrayBuffer, 0, sourceValues.length);
  assertCloseArray(
    Array.from(cachedBufferViewA),
    Array.from(sourceValues, value => value * combineScale),
    'cached scaled norm tensor should be uploaded with the folded combine scale'
  );

  const cachedWeightB = await ensurePleScaledProjectionNormWeight(context, combineScale);
  assert.strictEqual(
    cachedWeightB,
    cachedWeightA,
    'Gemma 4 PLE scaled norm cache should reuse the same tensor across decode steps'
  );

  destroyPleRuntimeCache(perLayerInputWeights);

  const cachedWeightC = await ensurePleScaledProjectionNormWeight(context, combineScale);
  assert.notStrictEqual(
    cachedWeightC.buffer,
    cachedWeightA.buffer,
    'destroyPleRuntimeCache should release the cached scaled norm buffer so a fresh cache can be built'
  );

  const offsetContext = createPleContext({
    embedTokensPerLayer: null,
    perLayerModelProjection: null,
    perLayerProjectionNorm: sourceValues,
  }, true);
  assert.equal(
    await ensurePleScaledProjectionNormWeight(offsetContext, combineScale),
    null,
    'offset-based RMSNorm models must not activate the folded PLE combine-scale cache'
  );

  const rangeRequests = [];
  const projectionSlice = await loadRangeBackedPleProjectionSliceBytes(
    {
      data: {
        kind: 'tensor_range_source',
        async loadRange(offset, length) {
          rangeRequests.push([offset, length]);
          return Uint8Array.from({ length }, (_, index) => index & 0xff);
        },
      },
      dtype: 'f16',
      layout: 'row',
      shape: Object.freeze([8, 4]),
      label: 'per_layer_model_projection',
    },
    3,
    2,
    4,
    'Range-backed projection test'
  );
  assert.deepEqual(rangeRequests, [[48, 16]]);
  assert.equal(projectionSlice.dtype, 'f16');
  assert.equal(projectionSlice.layout, 'row');
  assert.deepEqual(projectionSlice.shape, [2, 4]);
  assert.equal(projectionSlice.bytes.byteLength, 16);
} finally {
  destroyBufferPool();
  setDevice(null);
}

console.log('gemma4-ple-projection-norm-cache.test: ok');
