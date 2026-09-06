import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { bootstrapNodeWebGPUProvider, releaseNodeWebGPU } from './node_modules/doppler-gpu/src/tooling/node-webgpu.js';

// Installed host adapter with no optional dependencies; synthetic GPU only.
assert.throws(() => createRequire(import.meta.url).resolve('doe-gpu/node-webgpu'), /Cannot find/);
const providerModule = `data:text/javascript,${encodeURIComponent(`
  export const globals = { GPUBufferUsage: class {}, GPUShaderStage: class {}, GPUMapMode: class {}, GPUTextureUsage: class {} };
  export const create = () => ({ requestAdapter: async () => ({ requestDevice() {} }) });
`)}`;
try {
  const result = await bootstrapNodeWebGPUProvider(providerModule);
  assert.equal(result.ok, true);
  assert.equal(result.receipt.implementation, 'doppler');
  assert.equal(globalThis.navigator.gpu, result.session.gpu);
  assert.equal(result.receipt.attempts.length, 1);
} finally {
  const released = await releaseNodeWebGPU();
  assert.equal(released.released, true);
  assert.equal(released.receipt.globals.restored, true);
}
console.log('Installed standalone Node provider smoke passed (synthetic; Doe absent).');
