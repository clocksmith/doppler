// Opens a WebGPU device through Doppler's Node WebGPU host (src/tooling/node-webgpu.js), with the adapter's buffer
// limits (keys are 78.6 MB each at N = 2^16).

import { bootstrapNodeWebGPU, releaseNodeWebGPU } from '../../../src/tooling/node-webgpu.js';

export async function openDevice() {
  const boot = await bootstrapNodeWebGPU();
  if (!boot.ok) throw new Error(`no Node WebGPU provider: ${boot.detail}`);
  const adapter = boot.session.adapter;
  const limits = adapter.limits;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxBufferSize: limits.maxBufferSize,
    },
  });
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') console.error(`[he-gpu] device lost: ${info.message}`);
  });
  const info = adapter.info ?? {};
  return {
    device,
    provider: boot.provider,
    adapter: { vendor: info.vendor ?? '', architecture: info.architecture ?? '', device: info.device ?? '', description: info.description ?? '' },
    limits: { maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize, maxBufferSize: limits.maxBufferSize },
    async close() {
      device.destroy();
      await releaseNodeWebGPU();
    },
  };
}
