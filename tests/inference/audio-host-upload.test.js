import assert from 'node:assert/strict';
import { setDevice } from '../../src/gpu/device.js';
import { uploadHostArray } from '../../src/experimental/audio/gemma4.js';

// The Gemma 4 audio encoder used to call uploadData(melFeatures, label), but
// uploadData writes into an existing buffer. Host arrays must go through an
// acquired buffer or the encoder throws before any GPU work.
const originalGPUBuffer = globalThis.GPUBuffer;
const originalGPUBufferUsage = globalThis.GPUBufferUsage;
globalThis.GPUBufferUsage = { STORAGE: 0x80, COPY_SRC: 0x04, COPY_DST: 0x08, UNIFORM: 0x40, MAP_READ: 0x01 };
globalThis.GPUBuffer = class GPUBuffer {};
const writes = [];
setDevice({
  features: new Set(),
  limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20 },
  createBindGroup() { throw new Error('upload test must not create bind groups.'); },
  createBuffer(descriptor) { return Object.assign(new globalThis.GPUBuffer(), descriptor, { destroy() {} }); },
  queue: {
    writeBuffer(buffer, offset, data) { writes.push({ buffer, offset, byteLength: data.byteLength }); },
    submit() { throw new Error('upload test must not submit GPU work.'); },
    onSubmittedWorkDone: async () => {},
  },
}, { platformConfig: null });

try {
  const mel = new Float32Array([0.5, -1, 2, 3.25, 0, 1]);
  const buffer = uploadHostArray(mel, 'audio_mel_input');
  assert.ok(buffer instanceof globalThis.GPUBuffer, 'an acquired GPU buffer is returned');
  assert.ok(buffer.size >= mel.byteLength, 'the buffer holds the whole array');
  assert.equal(writes.length, 1);
  assert.equal(writes[0].buffer, buffer);
  assert.equal(writes[0].offset, 0);
  assert.equal(writes[0].byteLength, mel.byteLength);
  assert.throws(() => uploadHostArray([1, 2, 3], 'not-typed'), /requires a typed array/);
  console.log('audio host upload ok');
} finally {
  globalThis.GPUBuffer = originalGPUBuffer;
  globalThis.GPUBufferUsage = originalGPUBufferUsage;
}
