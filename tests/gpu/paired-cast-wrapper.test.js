import assert from 'node:assert/strict';
import { expandF32ToPaired, castPairedToF16 } from '../../src/gpu/kernels/paired-cast.js';

const scalar = { dtype: 'f32', shape: [3], buffer: { size: 24 } };
for (const metadata of [
  { precision: 'garbage' }, { precision: 'f32' }, { components: 1 },
  { elementStrideBytes: 8 }, { precision: 'two-f32', components: 2, elementStrideBytes: 8 },
]) {
  await assert.rejects(expandF32ToPaired({ ...scalar, ...metadata }), /scalar F32 input/,
    'malformed or explicit non-scalar metadata must reject before GPU allocation');
}
for (const metadata of [{}, { precision: 'two-f32' }, { precision: 'two-f32', components: 1, elementStrideBytes: 8 }]) {
  await assert.rejects(castPairedToF16({ ...scalar, ...metadata }), /explicit interleaved/);
}
await assert.rejects(expandF32ToPaired({ ...scalar, shape: [0] }), /positive/);
await assert.rejects(expandF32ToPaired({ ...scalar, shape: [0x100000000] }), /U32/);
await assert.rejects(expandF32ToPaired({ ...scalar, buffer: { size: 8 } }), /undersized/);
await assert.rejects(expandF32ToPaired(scalar, { outputBuffer: scalar.buffer }), /distinct/);
await assert.rejects(expandF32ToPaired(scalar, { outputBuffer: { size: 20 } }), /sufficient/);
console.log('paired cast pre-allocation guards passed');

// Failure ownership is tested with a fake device; this is not shader evidence.
globalThis.GPUBufferUsage = { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
const { setDevice } = await import('../../src/gpu/device.js');
const { getBufferPool, destroyBufferPool } = await import('../../src/memory/buffer-pool.js');
const { recordExpandF32ToPaired, recordCastPairedToF16 } = await import('../../src/gpu/kernels/paired-cast.js');
class Buffer {
  constructor({ size, usage }) { this.size = size; this.usage = usage; this.destroyed = false; }
  destroy() { this.destroyed = true; }
}
const oldBuffer = globalThis.GPUBuffer;
globalThis.GPUBuffer = Buffer;
function device() {
  return {
    features: new Set(), limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20,
      maxComputeWorkgroupSizeX: 256, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupStorageSize: 16384,
      maxStorageBuffersPerShaderStage: 8, maxUniformBufferBindingSize: 65536, maxComputeWorkgroupsPerDimension: 65535 },
    queue: { submit() {}, writeBuffer() {}, onSubmittedWorkDone: async () => {} },
    createBuffer: spec => new Buffer(spec),
    createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}),
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup() { throw new Error('controlled bind failure'); },
  };
}
try {
  for (const recorded of [false, true]) for (const paired of [false, true]) for (const borrowed of [false, true]) {
    destroyBufferPool(); const gpu = device(); setDevice(gpu, { platformConfig: null });
    getBufferPool().configure({ enablePooling: false });
    const inputBuffer = new Buffer({ size: 24, usage: 128 });
    const input = { dtype: 'f32', shape: [3], buffer: inputBuffer,
      ...(paired ? { precision: 'two-f32', components: 2, elementStrideBytes: 8 } : {}) };
    const supplied = borrowed ? new Buffer({ size: 24, usage: 128 }) : null;
    const options = supplied ? { outputBuffer: supplied } : {};
    const recorderUniforms = [];
    const recorder = { device: gpu, createUniformBuffer() { const buffer = new Buffer({ size: 16, usage: 64 }); recorderUniforms.push(buffer); return buffer; }, beginComputePass() { throw new Error('must fail before recording'); } };
    const operation = paired ? (recorded ? recordCastPairedToF16 : castPairedToF16)
      : (recorded ? recordExpandF32ToPaired : expandF32ToPaired);
    await assert.rejects(recorded ? operation(recorder, input, options) : operation(input, options), /controlled bind failure/);
    assert.equal(getBufferPool().getStats().activeBuffers, 0, 'failed output/uniform allocations settle');
    assert.equal(inputBuffer.destroyed, false, 'input remains borrowed');
    if (supplied) assert.equal(supplied.destroyed, false, 'caller output remains borrowed');
    for (const buffer of recorderUniforms) {
      assert.equal(buffer.destroyed, false, 'recorded uniform stays with its recorder owner');
      buffer.destroy();
    }
  }
} finally {
  destroyBufferPool(); setDevice(null, { platformConfig: null });
  if (oldBuffer === undefined) delete globalThis.GPUBuffer; else globalThis.GPUBuffer = oldBuffer;
}
console.log('paired cast immediate/recorded owned and borrowed failure cleanup passed');
