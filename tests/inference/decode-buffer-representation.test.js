import assert from 'node:assert/strict';
import { DecodeBufferManager } from '../../src/inference/decode-buffers.js';
import { setDevice } from '../../src/gpu/device.js';
globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_DST: 8, COPY_SRC: 4 };
const created = [];
setDevice({ features: new Set(), limits: {}, createBindGroup() { return {}; }, createBuffer({ size }) {
  const buffer = { size, destroyed: false, destroy() { this.destroyed = true; } };
  created.push(buffer); return buffer;
} }, { platformConfig: null });
const manager = new DecodeBufferManager();
const scalar = { hiddenSize: 257, intermediateSize: 513, activationDtype: 'f32', enablePingPong: true };
const paired = { ...scalar, activationPrecision: 'two-f32', activationComponents: 2, activationElementStrideBytes: 8 };
try {
  const original = manager.ensureBuffers(scalar);
  const next = manager.ensureBuffers(paired);
  assert.notEqual(next.hidden, original.hidden, 'same component dtype does not imply the same storage representation');
  assert.equal(original.hidden.destroyed, true);
  assert.equal(next.hidden.size, 512 * 8);
  assert.equal(next.ffnIntermediate.size, 768 * 8);
  assert.equal(manager.getStats().totalBytes, 3 * 512 * 8 + 768 * 8, 'report padded physical allocation');
  assert.equal(manager.ensureBuffers(paired), next);
  assert.throws(() => manager.ensureBuffers({ ...paired, activationElementStrideBytes: 4 }), /eight-byte stride/);
  assert.equal(next.hidden.destroyed, false, 'invalid replacement preserves currently owned buffers');
  assert.equal(manager.ensureBuffers(paired), next);
} finally { manager.release(); setDevice(null, { platformConfig: null }); }
assert(created.every(buffer => buffer.destroyed));
console.log('decode buffer representation identity, padded capacity, reuse and invalid replacement passed');
