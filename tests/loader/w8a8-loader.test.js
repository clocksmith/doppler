import assert from 'node:assert/strict';
import { W8A8_POLICY } from '../../src/config/schema/w8a8.schema.js';
import { quantizeToW8A8Rows } from '../../src/converter/w8a8-quantizer.js';

globalThis.GPUBufferUsage = { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
const { setDevice } = await import('../../src/gpu/device.js');
const { destroyBufferPool, getBufferPool, releaseBuffer } = await import('../../src/memory/buffer-pool.js');
const { loadW8A8Packed } = await import('../../src/loader/tensors/w8a8-loader.js');
let failUpload = false;
const device = {
  features: new Set(), limits: { maxBufferSize: 1048576, maxStorageBufferBindingSize: 1048576, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupStorageSize: 16384 },
  createBindGroup(descriptor) { return descriptor; },
  createBuffer({ size, usage }) { return { size, usage, bytes: new Uint8Array(size), destroy() {} }; },
  queue: {
    submit() {}, onSubmittedWorkDone() { return Promise.resolve(); },
    writeBuffer(buffer, offset, data, dataOffset = 0, length = data.byteLength) {
      if (failUpload) throw new Error('injected upload failure');
      const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      buffer.bytes.set(bytes.subarray(dataOffset, dataOffset + length), offset);
    },
  },
};
destroyBufferPool(); setDevice(device, { platformConfig: null });
getBufferPool().configure({ enablePooling: true });
const encoded = quantizeToW8A8Rows(new Float32Array([127, -2, 1, 0, -127, 0]), [2, 3], W8A8_POLICY);
const location = { dtype: 'W8A8', role: 'matmul', layout: 'row', shape: [2, 3], size: encoded.quantized.length, storage: encoded.storage };
const config = { w8a8Policy: W8A8_POLICY };
const loaded = await loadW8A8Packed(encoded.quantized, location, 'test', config);
assert.equal(loaded.data.dtype, 'w8a8'); assert.equal(loaded.allocatedBuffers.length, 1);
assert.deepEqual(loaded.data.buffer.bytes.slice(0, encoded.quantized.length), encoded.quantized);
assert.equal(loaded.data.metadata.rowStrideBytes, 8);
releaseBuffer(loaded.data.buffer);
assert.equal(getBufferPool().getStats().activeBuffers, 0);
for (const bad of [ { ...location, role: 'embedding' }, { ...location, layout: 'column' }, { ...location, storage: null } ]) {
  await assert.rejects(loadW8A8Packed(encoded.quantized, bad, 'test', config), /W8A8/);
  assert.equal(getBufferPool().getStats().activeBuffers, 0);
}
await assert.rejects(loadW8A8Packed(encoded.quantized, location, 'test', {}), /explicitly/);
const corrupted = encoded.quantized.slice(); corrupted[4] = 128;
await assert.rejects(loadW8A8Packed(corrupted, location, 'test', config), /invalid code/);
failUpload = true;
await assert.rejects(loadW8A8Packed(encoded.quantized, location, 'test', config), /injected upload/);
assert.equal(getBufferPool().getStats().activeBuffers, 0);
destroyBufferPool(); setDevice(null, { platformConfig: null });
console.log('W8A8 packed upload, fail-closed admission and resource cleanup passed.');
