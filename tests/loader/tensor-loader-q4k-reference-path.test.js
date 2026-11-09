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
const { loadQ4KDequant, loadTensorToGPU } = await import('../../src/loader/tensors/tensor-loader.js');
const {
  quantizeToQ4KM,
  quantizeToQ4KMRowWise,
  dequantizeQ4KM,
  dequantizeQ4KMRowWise,
} = await import('../../src/converter/quantizer.js');

class FakeBuffer {
  constructor({ size, usage }) {
    this.size = size;
    this.usage = usage;
    this.destroyed = false;
    this.bytes = new Uint8Array(size);
  }

  destroy() {
    this.destroyed = true;
  }
}

globalThis.GPUBuffer = FakeBuffer;

function createFakeDevice() {
  return {
    features: new Set(['shader-f16']),
    limits: {
      maxBufferSize: 1 << 26,
      maxStorageBufferBindingSize: 1 << 26,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
    },
    queue: {
      submit() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
      writeBuffer(buffer, offset, data, byteOffset = 0, byteLength = data.byteLength) {
        const bytes = data instanceof Uint8Array
          ? data.subarray(byteOffset, byteOffset + byteLength)
          : new Uint8Array(data.buffer, data.byteOffset + byteOffset, byteLength);
        buffer.bytes.set(bytes, offset);
      },
    },
    createBindGroup(descriptor) {
      return descriptor;
    },
    createBuffer({ size, usage }) {
      return new FakeBuffer({ size, usage });
    },
  };
}

function resetRuntimeState(device) {
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
  if (device) {
    getBufferPool().configure({ enablePooling: false });
  }
}

function decodeF32Prefix(buffer, count) {
  return Array.from(new Float32Array(buffer.bytes.buffer, 0, count));
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);

  const shape = [8, 2560];
  const source = new Float32Array(shape[0] * shape[1]);
  for (let i = 0; i < source.length; i += 1) {
    source[i] = Math.sin(i * 0.013) * 0.5 + Math.cos(i * 0.007) * 0.25;
  }

  const { quantized, numBlocks } = quantizeToQ4KM(source, shape);
  const expected = dequantizeQ4KM(quantized, numBlocks, shape);

  const result = await loadQ4KDequant(
    quantized,
    {
      size: quantized.byteLength,
      shape,
      dtype: 'Q4_K_M',
      role: 'matmul',
      layout: 'row',
    },
    'translategemma_q_proj',
    {
      useFusedQ4K: false,
      keepF32Weights: true,
      allowF32UpcastNonMatmul: false,
      q4kLayout: 'row',
      gpuCapabilities: { hasF16: true, hasSubgroups: true },
      loaderDebug: { preferCpuDequant: true },
    }
  );

  assert.equal(result.data.dtype, 'f32');
  assert.equal(result.data.layout, 'row');
  assert.deepEqual(result.data.shape, shape);
  assert.equal(result.allocatedBuffers.length, 1);
  assert.equal(result.allocatedBuffers[0], result.data.buffer);

  const actualPrefix = decodeF32Prefix(result.data.buffer, 16);
  const expectedPrefix = Array.from(expected.slice(0, 16));
  assert.deepEqual(actualPrefix, expectedPrefix);
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);

  const shape = [3, 300];
  const source = new Float32Array(shape[0] * shape[1]);
  for (let i = 0; i < source.length; i += 1) {
    source[i] = ((i % 37) - 18) / 7.5;
  }

  const { quantized } = quantizeToQ4KMRowWise(source, shape);
  const expected = dequantizeQ4KMRowWise(quantized, shape);

  const result = await loadQ4KDequant(
    quantized,
    {
      size: quantized.byteLength,
      shape,
      dtype: 'Q4_K_M',
      role: 'matmul',
      layout: 'row',
    },
    'rowwise_q4k_weight',
    {
      useFusedQ4K: false,
      keepF32Weights: true,
      allowF32UpcastNonMatmul: false,
      q4kLayout: 'row',
      gpuCapabilities: { hasF16: true, hasSubgroups: true },
      loaderDebug: { preferCpuDequant: true },
    }
  );

  assert.equal(result.data.dtype, 'f32');
  assert.equal(result.data.layout, 'row');
  assert.deepEqual(
    decodeF32Prefix(result.data.buffer, 24),
    Array.from(expected.slice(0, 24))
  );
}

resetRuntimeState(null);

{
  const device = createFakeDevice();
  device.features = new Set();
  resetRuntimeState(device);
  setDevice(device, {
    platformConfig: {
      platform: {
        id: 'basic',
        name: 'Basic',
      },
      capabilities: {
        hasF16: false,
        hasSubgroups: false,
      },
    },
  });

  const shape = [3, 300];
  const source = new Float32Array(shape[0] * shape[1]);
  for (let i = 0; i < source.length; i += 1) {
    source[i] = ((i % 23) - 11) / 5.0;
  }

  const { quantized } = quantizeToQ4KMRowWise(source, shape);
  const expected = dequantizeQ4KMRowWise(quantized, shape);

  const result = await loadTensorToGPU(
    quantized,
    {
      size: quantized.byteLength,
      shape,
      dtype: 'Q4_K_M',
      role: 'matmul',
      layout: 'row',
    },
    'basic_backend_q4k_weight',
    {
      useFusedQ4K: false,
      keepF32Weights: true,
      allowF32UpcastNonMatmul: false,
      q4kLayout: 'row',
      gpuCapabilities: { hasF16: false, hasSubgroups: false },
    }
  );

  assert.equal(result.data.dtype, 'f32');
  assert.equal(result.data.layout, 'row');
  assert.deepEqual(
    decodeF32Prefix(result.data.buffer, 24),
    Array.from(expected.slice(0, 24))
  );
}

resetRuntimeState(null);

{
  const device = createFakeDevice();
  device.limits.maxBufferSize = 1 << 20;
  device.limits.maxStorageBufferBindingSize = 1 << 20;
  device.features = new Set(['shader-f16', 'subgroups']);
  resetRuntimeState(device);

  const shape = [4096, 256];
  const quantized = new Uint8Array(shape[0] * 144);
  for (let i = 0; i < quantized.length; i += 1) {
    quantized[i] = i & 0xff;
  }

  const result = await loadTensorToGPU(
    quantized,
    {
      size: quantized.byteLength,
      shape,
      dtype: 'Q4_K_M',
      role: 'matmul',
      layout: 'row',
    },
    'low_limit_q4k_down_proj',
    {
      useFusedQ4K: false,
      keepF32Weights: false,
      allowF32UpcastNonMatmul: false,
      q4kLayout: 'row',
      q4kMaterializationMode: 'dense',
      gpuCapabilities: { hasF16: true, hasSubgroups: true },
    }
  );

  assert.equal(result.data.dtype, 'q4k');
  assert.equal(result.data.layout, 'row');
  assert.deepEqual(result.data.shape, shape);
  assert.equal(result.data.buffer.size, quantized.byteLength);
  assert.equal(result.allocatedBuffers.length, 1);
}

// The explicit CPU reference and GPU materialization share packing, not
// inference policy. A flat filter must not acquire per-row padding.
{
  resetRuntimeState(createFakeDevice());
  const config = { useFusedQ4K: false, keepF32Weights: true, q4kLayout: 'row',
    loaderDebug: { preferCpuDequant: true } };
  const shape = [512, 1, 3];
  const source = Float32Array.from({ length: 1536 }, (_, i) => Math.sin(i * 0.03));
  const { quantized, numBlocks } = quantizeToQ4KM(source, shape);
  const expected = dequantizeQ4KM(quantized, numBlocks, shape);
  const result = await loadQ4KDequant(quantized, {
    shape, size: quantized.byteLength, dtype: 'Q4_K_M', role: 'other',
  }, 'flat-filter-cpu-reference', config);
  assert.deepEqual(decodeF32Prefix(result.data.buffer, expected.length), Array.from(expected));
  // Column storage is row-padded after transpose, including on basic backends.
  const rowShape = [4, 3];
  const rowValues = source.slice(0, 12);
  const rowBytes = quantizeToQ4KMRowWise(rowValues, rowShape).quantized;
  const column = await loadQ4KDequant(rowBytes, {
    shape: [3, 4], size: rowBytes.byteLength, dtype: 'Q4_K_M', role: 'other', layout: 'column',
  }, 'column-cpu-reference', config);
  assert.equal(column.data.layout, 'column');
  assert.deepEqual(decodeF32Prefix(column.data.buffer, 12),
    Array.from(dequantizeQ4KMRowWise(rowBytes, rowShape)));
}

resetRuntimeState(null);
console.log('tensor-loader-q4k-reference-path.test: ok');
