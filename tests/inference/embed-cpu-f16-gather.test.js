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
  constructor({ size, usage = 0, label = null }) {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.destroyed = false;
    this._arrayBuffer = new ArrayBuffer(size);
  }

  destroy() {
    this.destroyed = true;
  }

  async mapAsync() {
    return Promise.resolve();
  }

  getMappedRange(offset = 0, size = this.size - offset) {
    return this._arrayBuffer.slice(offset, offset + size);
  }

  unmap() {}
}

globalThis.GPUBuffer = FakeBuffer;

const { setDevice } = await import('../../src/gpu/device.js');
const { createCpuWeightBuffer } = await import('../../src/gpu/weight-buffer.js');

// Capture writeBuffer calls to inspect what data was uploaded
const writeCalls = [];
function createFakeDevice(options = {}) {
  const hasF16 = options.hasF16 === true;
  return {
    lost: new Promise(() => {}),
    features: hasF16 ? new Set(['shader-f16']) : new Set(),
    limits: {
      maxStorageBufferBindingSize: 1 << 30,
      maxBufferSize: 1 << 30,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
    },
    queue: {
      submit(commandBuffers) {
        for (const commandBuffer of commandBuffers) {
          for (const op of commandBuffer.ops ?? []) {
            const src = new Uint8Array(op.src._arrayBuffer, op.srcOffset, op.size);
            const dst = new Uint8Array(op.dst._arrayBuffer, op.dstOffset, op.size);
            dst.set(src);
          }
        }
      },
      writeBuffer(buffer, offset, data) {
        const bytes = data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        new Uint8Array(buffer._arrayBuffer, offset, bytes.byteLength).set(bytes);
        writeCalls.push({
          buffer,
          offset,
          data: data instanceof Float32Array ? new Float32Array(data) : data,
        });
      },
      onSubmittedWorkDone() { return Promise.resolve(); },
    },
    createBuffer({ size, label = null }) {
      return new FakeBuffer({ size, label });
    },
    createBindGroupLayout() { return {}; },
    createBindGroup() { return {}; },
    createShaderModule() { return {}; },
    createComputePipeline() { return { getBindGroupLayout() { return {}; } }; },
    createComputePipelineAsync() {
      return Promise.resolve({ getBindGroupLayout() { return {}; } });
    },
    createCommandEncoder() {
      const ops = [];
      return {
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          ops.push({ src, srcOffset, dst, dstOffset, size });
        },
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        finish() { return { ops }; },
      };
    },
  };
}

const fakeDevice = createFakeDevice();
setDevice(fakeDevice, { platformConfig: null });

const { embed } = await import('../../src/inference/pipelines/text/embed.js');
const { f16ToF32 } = await import('../../src/loader/dtype-utils.js');

// F16 bit-pattern constants
const F16_1_0 = 0x3C00;   // 1.0
const F16_0_5 = 0x3800;   // 0.5
const F16_2_0 = 0x4000;   // 2.0
const F16_NEG1 = 0xBC00;  // -1.0
const F16_3_0 = 0x4200;   // 3.0

const hiddenSize = 4;
const vocabSize = 3;

// Embedding table (row-major, F16): 3 tokens × 4 dims
// token 0: [1.0, 0.5, 2.0, -1.0]
// token 1: [0.5, 1.0, -1.0, 3.0]
// token 2: [3.0, 2.0, 0.5, 0.5]
const embData = new Uint16Array([
  F16_1_0, F16_0_5, F16_2_0, F16_NEG1,
  F16_0_5, F16_1_0, F16_NEG1, F16_3_0,
  F16_3_0, F16_2_0, F16_0_5, F16_0_5,
]);

// === Test 1: F16 CPU gather decodes bit-patterns as F32 values ===
{
  const embedBuffer = createCpuWeightBuffer(embData, 'f16', 'row', [vocabSize, hiddenSize], 'test_f16_embed');
  writeCalls.length = 0;

  await embed([1, 0], embedBuffer, {
    hiddenSize,
    vocabSize,
    scaleEmbeddings: false,
    debug: false,
    activationDtype: 'f32',
    embeddingDtype: 'f16',
  });

  assert.ok(writeCalls.length >= 1, 'writeBuffer should be called for CPU embed output');
  const written = writeCalls[writeCalls.length - 1].data;
  assert.ok(written instanceof Float32Array, 'embed output should be Float32Array');
  assert.equal(written.length, 2 * hiddenSize, 'output length should be numTokens * hiddenSize');

  // Token 1 (index 0 in result): [0.5, 1.0, -1.0, 3.0]
  assert.ok(Math.abs(written[0] - 0.5) < 1e-3, `token1[0]: expected 0.5, got ${written[0]}`);
  assert.ok(Math.abs(written[1] - 1.0) < 1e-3, `token1[1]: expected 1.0, got ${written[1]}`);
  assert.ok(Math.abs(written[2] - (-1.0)) < 1e-3, `token1[2]: expected -1.0, got ${written[2]}`);
  assert.ok(Math.abs(written[3] - 3.0) < 1e-3, `token1[3]: expected 3.0, got ${written[3]}`);

  // Token 0 (index 1 in result): [1.0, 0.5, 2.0, -1.0]
  assert.ok(Math.abs(written[4] - 1.0) < 1e-3, `token0[0]: expected 1.0, got ${written[4]}`);
  assert.ok(Math.abs(written[5] - 0.5) < 1e-3, `token0[1]: expected 0.5, got ${written[5]}`);
  assert.ok(Math.abs(written[6] - 2.0) < 1e-3, `token0[2]: expected 2.0, got ${written[6]}`);
  assert.ok(Math.abs(written[7] - (-1.0)) < 1e-3, `token0[3]: expected -1.0, got ${written[7]}`);

  // Confirm bit-copy would have been wrong: F16_0_5 = 0x3800 = 14336 as integer
  assert.ok(written[0] < 10, `F16 value should decode to ~0.5, not integer bit-pattern ~${F16_0_5}`);
}

// === Test 2: F32 CPU gather still works correctly ===
{
  const f32Data = new Float32Array([
    1.0, 0.5, 2.0, -1.0,
    0.5, 1.0, -1.0, 3.0,
    3.0, 2.0, 0.5, 0.5,
  ]);
  const embedBuffer = createCpuWeightBuffer(f32Data, 'f32', 'row', [vocabSize, hiddenSize], 'test_f32_embed');
  writeCalls.length = 0;

  await embed([2], embedBuffer, {
    hiddenSize,
    vocabSize,
    scaleEmbeddings: false,
    debug: false,
    activationDtype: 'f32',
    embeddingDtype: 'f32',
  });

  const written = writeCalls[writeCalls.length - 1].data;
  assert.ok(written instanceof Float32Array);
  assert.ok(Math.abs(written[0] - 3.0) < 1e-6, `f32 token2[0]: expected 3.0, got ${written[0]}`);
  assert.ok(Math.abs(written[1] - 2.0) < 1e-6, `f32 token2[1]: expected 2.0, got ${written[1]}`);
}

// === Test 3: Float32Array data tagged dtype='f16' (loader-decoded path) uses values directly ===
// The loader's f16_to_f32 CPU dispatch decodes F16 bytes into Float32Array before creating
// CpuWeightBuffer. In that case dtype='f16' but data is already F32 — no second decode needed.
{
  const decodedF32 = new Float32Array([0.5, 1.0, -1.0, 2.0]);
  // dtype='f16' but data is Float32Array (as produced by loadTensorToCPU f16_to_f32 path)
  const embedBuffer = createCpuWeightBuffer(decodedF32, 'f16', 'row', [1, hiddenSize], 'test_f32_mislabeled_f16');
  writeCalls.length = 0;

  await embed([0], embedBuffer, {
    hiddenSize,
    vocabSize: 1,
    scaleEmbeddings: false,
    debug: false,
    activationDtype: 'f32',
    embeddingDtype: 'f16',
  });

  const written = writeCalls[writeCalls.length - 1].data;
  assert.ok(written instanceof Float32Array);
  // Values must pass through as-is (not double-decoded via f16ToF32)
  assert.ok(Math.abs(written[0] - 0.5) < 1e-6, `loader-decoded[0]: expected 0.5, got ${written[0]}`);
  assert.ok(Math.abs(written[1] - 1.0) < 1e-6, `loader-decoded[1]: expected 1.0, got ${written[1]}`);
  assert.ok(Math.abs(written[2] - (-1.0)) < 1e-6, `loader-decoded[2]: expected -1.0, got ${written[2]}`);
  assert.ok(Math.abs(written[3] - 2.0) < 1e-6, `loader-decoded[3]: expected 2.0, got ${written[3]}`);
}

// === Test 4 (was 3): Unsupported dtype throws actionable error ===
{
  const badData = new Uint32Array(vocabSize * hiddenSize);
  const embedBuffer = createCpuWeightBuffer(badData, 'q4k', 'row', [vocabSize, hiddenSize], 'test_bad_dtype');

  await assert.rejects(
    () => embed([0], embedBuffer, {
      hiddenSize,
      vocabSize,
      scaleEmbeddings: false,
      debug: false,
      activationDtype: 'f32',
      embeddingDtype: 'q4k',
    }),
    (err) => {
      assert.ok(err.message.includes('unsupported dtype'), `expected unsupported dtype error, got: ${err.message}`);
      assert.ok(err.message.includes('q4k'), `error should mention the bad dtype`);
      return true;
    }
  );
}

// === Test 5: F16 with scaleEmbeddings applies scale after decoding ===
{
  const embedBuffer = createCpuWeightBuffer(embData, 'f16', 'row', [vocabSize, hiddenSize], 'test_f16_scaled');
  writeCalls.length = 0;
  const scaleFactor = Math.sqrt(hiddenSize);

  await embed([0], embedBuffer, {
    hiddenSize,
    vocabSize,
    scaleEmbeddings: true,
    debug: false,
    activationDtype: 'f32',
    embeddingDtype: 'f16',
  });

  const written = writeCalls[writeCalls.length - 1].data;
  assert.ok(written instanceof Float32Array);
  // token 0: [1.0, 0.5, 2.0, -1.0] × sqrt(4)=2
  assert.ok(Math.abs(written[0] - 1.0 * scaleFactor) < 1e-3, `scaled token0[0]: expected ${1.0 * scaleFactor}, got ${written[0]}`);
  assert.ok(Math.abs(written[1] - 0.5 * scaleFactor) < 1e-3, `scaled token0[1]: expected ${0.5 * scaleFactor}, got ${written[1]}`);
}

// === Test 6: F16 activation requests fail fast when shader-f16 is unavailable ===
{
  const embedBuffer = createCpuWeightBuffer(embData, 'f16', 'row', [vocabSize, hiddenSize], 'test_f16_requires_feature');

  await assert.rejects(
    () => embed([0], embedBuffer, {
      hiddenSize,
      vocabSize,
      scaleEmbeddings: false,
      debug: false,
      activationDtype: 'f16',
      embeddingDtype: 'f16',
    }),
    /activationDtype="f16" requires shader-f16 support/
  );
}

// === Test 7: Range-backed CPU gather reads only the requested F16 rows ===
{
  const embedBytes = new Uint8Array(embData.buffer.slice(0));
  const rangeCalls = [];
  const embedBuffer = createCpuWeightBuffer({
    kind: 'tensor_range_source',
    sourceDtype: 'f16',
    async loadRange(byteOffset, byteLength) {
      rangeCalls.push([byteOffset, byteLength]);
      return embedBytes.slice(byteOffset, byteOffset + byteLength);
    },
  }, 'f16', 'row', [vocabSize, hiddenSize], 'test_f16_range_embed');
  writeCalls.length = 0;

  await embed([1, 0], embedBuffer, {
    hiddenSize,
    vocabSize,
    scaleEmbeddings: false,
    debug: false,
    activationDtype: 'f32',
    embeddingDtype: 'f16',
  });

  const written = writeCalls[writeCalls.length - 1].data;
  assert.ok(written instanceof Float32Array, 'range-backed embed output should be Float32Array');
  assert.deepEqual(rangeCalls, [
    [8, 8],
    [0, 8],
  ]);
  assert.ok(Math.abs(written[0] - 0.5) < 1e-3, `range token1[0]: expected 0.5, got ${written[0]}`);
  assert.ok(Math.abs(written[1] - 1.0) < 1e-3, `range token1[1]: expected 1.0, got ${written[1]}`);
  assert.ok(Math.abs(written[4] - 1.0) < 1e-3, `range token0[0]: expected 1.0, got ${written[4]}`);
  assert.ok(Math.abs(written[5] - 0.5) < 1e-3, `range token0[1]: expected 0.5, got ${written[5]}`);
}

// === Test 8: GPU token buffers can drive CPU embedding gather via readback ===
{
  const embedBuffer = createCpuWeightBuffer(embData, 'f16', 'row', [vocabSize, hiddenSize], 'test_f16_gpu_tokens');
  const tokenBuffer = fakeDevice.createBuffer({
    size: 3 * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: 'test_token_ids',
  });
  fakeDevice.queue.writeBuffer(tokenBuffer, 0, new Uint32Array([2, 1, 0]));
  writeCalls.length = 0;

  await embed(tokenBuffer, embedBuffer, {
    hiddenSize,
    vocabSize,
    numTokens: 2,
    indexOffset: 1,
    scaleEmbeddings: false,
    debug: false,
    activationDtype: 'f32',
    embeddingDtype: 'f16',
  });

  const written = writeCalls[writeCalls.length - 1].data;
  assert.ok(written instanceof Float32Array);
  assert.equal(written.length, 2 * hiddenSize);
  assert.ok(Math.abs(written[0] - 0.5) < 1e-3, `gpu-token token1[0]: expected 0.5, got ${written[0]}`);
  assert.ok(Math.abs(written[1] - 1.0) < 1e-3, `gpu-token token1[1]: expected 1.0, got ${written[1]}`);
  assert.ok(Math.abs(written[4] - 1.0) < 1e-3, `gpu-token token0[0]: expected 1.0, got ${written[4]}`);
  assert.ok(Math.abs(written[5] - 0.5) < 1e-3, `gpu-token token0[1]: expected 0.5, got ${written[5]}`);
}

// === Test 9: Explicit CPU gather cast is allowed under require_cast_step ===
{
  const fakeF16Device = createFakeDevice({ hasF16: true });
  setDevice(fakeF16Device, { platformConfig: null });
  const embedBuffer = createCpuWeightBuffer(embData, 'f16', 'row', [vocabSize, hiddenSize], 'test_f16_explicit_cast');

  const result = await embed([0], embedBuffer, {
    hiddenSize,
    vocabSize,
    scaleEmbeddings: false,
    debug: false,
    activationDtype: 'f16',
    embeddingDtype: 'f16',
    executionPolicies: {
      dtypeTransition: 'require_cast_step',
    },
  });

  assert.equal(result.dtype, 'f16');
  setDevice(fakeDevice, { platformConfig: null });
}

setDevice(null, { platformConfig: null });

console.log('embed-cpu-f16-gather.test: ok');
