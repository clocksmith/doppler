import assert from 'node:assert/strict';

import { runNodeCommand } from '../../src/tooling/node-command-runner.js';

const originalWebgpuModule = process.env.DOPPLER_NODE_WEBGPU_MODULE;
const originalBufferUsage = globalThis.GPUBufferUsage;
const originalShaderStage = globalThis.GPUShaderStage;

try {
  globalThis.GPUBufferUsage = undefined;
  globalThis.GPUShaderStage = undefined;
  process.env.DOPPLER_NODE_WEBGPU_MODULE = `doppler-webgpu-missing-${Date.now()}`;

  await assert.rejects(
    () => runNodeCommand({
      command: 'verify',
      suite: 'kernels',
    }),
    /node command: WebGPU runtime is incomplete in Node\./
  );
} finally {
  if (originalBufferUsage === undefined) {
    delete globalThis.GPUBufferUsage;
  } else {
    globalThis.GPUBufferUsage = originalBufferUsage;
  }

  if (originalShaderStage === undefined) {
    delete globalThis.GPUShaderStage;
  } else {
    globalThis.GPUShaderStage = originalShaderStage;
  }

  if (originalWebgpuModule === undefined) {
    delete process.env.DOPPLER_NODE_WEBGPU_MODULE;
  } else {
    process.env.DOPPLER_NODE_WEBGPU_MODULE = originalWebgpuModule;
  }
}

console.log('node-command-runtime-gating.test: ok');
