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
const { loadFloat } = await import('../../src/loader/tensors/tensor-loader.js');

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

function createFakeDevice() {
  return {
    features: new Set(['shader-f16']),
    limits: {
      maxBufferSize: 1 << 20,
      maxStorageBufferBindingSize: 1 << 20,
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
    getBufferPool().configure({ enablePooling: true });
  }
}

async function loadRetainedBuffer(bytes, location, name, config = {}) {
  const result = await loadFloat(bytes, location, name, {
    allowF32UpcastNonMatmul: false,
    ...config,
  });
  return result.data?.buffer ?? result.data;
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);

  const firstBytes = new Uint8Array([1, 0, 2, 0, 3, 0, 4, 0]);
  const secondBytes = new Uint8Array([5, 0, 6, 0, 7, 0, 8, 0]);
  const location = {
    size: firstBytes.byteLength,
    shape: [2, 2],
    dtype: 'F16',
    role: 'matmul',
  };

  const first = await loadFloat(firstBytes, location, 'gate_proj', {
    allowF32UpcastNonMatmul: false,
  });
  const second = await loadFloat(secondBytes, location, 'up_proj', {
    allowF32UpcastNonMatmul: false,
  });

  assert.notEqual(first.data.buffer, second.data.buffer);
  assert.deepEqual(Array.from(first.data.buffer.bytes.slice(0, firstBytes.length)), Array.from(firstBytes));
  assert.deepEqual(Array.from(second.data.buffer.bytes.slice(0, secondBytes.length)), Array.from(secondBytes));
  assert.equal(getBufferPool().getStats().activeBuffers, 2);
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);

  const normBytes = new Uint8Array([9, 0, 10, 0, 11, 0, 12, 0]);
  const secondNormBytes = new Uint8Array([13, 0, 14, 0, 15, 0, 16, 0]);
  const location = {
    size: normBytes.byteLength,
    shape: [4],
    dtype: 'F16',
    role: 'norm',
  };

  const firstBuffer = await loadRetainedBuffer(normBytes, location, 'input_norm');
  const secondBuffer = await loadRetainedBuffer(secondNormBytes, location, 'post_attn_norm');

  assert.notEqual(firstBuffer, secondBuffer);
  assert.deepEqual(Array.from(firstBuffer.bytes.slice(0, normBytes.length)), Array.from(normBytes));
  assert.deepEqual(Array.from(secondBuffer.bytes.slice(0, secondNormBytes.length)), Array.from(secondNormBytes));
  assert.equal(getBufferPool().getStats().activeBuffers, 2);
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);

  const residualBytes = new Float32Array([1.25, 2.5]).buffer;
  const secondResidualBytes = new Float32Array([3.75, 5]).buffer;
  const location = {
    size: residualBytes.byteLength,
    shape: [2],
    dtype: 'F32',
    role: 'residual',
  };

  const firstBuffer = await loadRetainedBuffer(new Uint8Array(residualBytes), location, 'attn_residual');
  const secondBuffer = await loadRetainedBuffer(new Uint8Array(secondResidualBytes), location, 'ffn_residual');

  assert.notEqual(firstBuffer, secondBuffer);
  assert.deepEqual(
    Array.from(firstBuffer.bytes.slice(0, residualBytes.byteLength)),
    Array.from(new Uint8Array(residualBytes))
  );
  assert.deepEqual(
    Array.from(secondBuffer.bytes.slice(0, secondResidualBytes.byteLength)),
    Array.from(new Uint8Array(secondResidualBytes))
  );
  assert.equal(getBufferPool().getStats().activeBuffers, 2);
}

resetRuntimeState(null);
console.log('tensor-loader-weight-ownership.test: ok');
