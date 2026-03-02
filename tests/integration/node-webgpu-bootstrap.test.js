import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { bootstrapNodeWebGPU } from '../../src/tooling/node-webgpu.js';

function snapshotState() {
  return {
    moduleEnv: process.env.DOPPLER_NODE_WEBGPU_MODULE,
    hadNavigator: typeof globalThis.navigator !== 'undefined',
    navigatorGpuDescriptor: typeof globalThis.navigator !== 'undefined'
      ? Object.getOwnPropertyDescriptor(globalThis.navigator, 'gpu')
      : null,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    GPUShaderStage: globalThis.GPUShaderStage,
    GPUMapMode: globalThis.GPUMapMode,
    GPUTextureUsage: globalThis.GPUTextureUsage,
    markerA: globalThis.__dopplerNodeWebgpuMarkerA,
    markerB: globalThis.__dopplerNodeWebgpuMarkerB,
  };
}

function restoreState(snapshot) {
  if (snapshot.moduleEnv === undefined) {
    delete process.env.DOPPLER_NODE_WEBGPU_MODULE;
  } else {
    process.env.DOPPLER_NODE_WEBGPU_MODULE = snapshot.moduleEnv;
  }

  if (snapshot.GPUBufferUsage === undefined) {
    delete globalThis.GPUBufferUsage;
  } else {
    globalThis.GPUBufferUsage = snapshot.GPUBufferUsage;
  }
  if (snapshot.GPUShaderStage === undefined) {
    delete globalThis.GPUShaderStage;
  } else {
    globalThis.GPUShaderStage = snapshot.GPUShaderStage;
  }
  if (snapshot.GPUMapMode === undefined) {
    delete globalThis.GPUMapMode;
  } else {
    globalThis.GPUMapMode = snapshot.GPUMapMode;
  }
  if (snapshot.GPUTextureUsage === undefined) {
    delete globalThis.GPUTextureUsage;
  } else {
    globalThis.GPUTextureUsage = snapshot.GPUTextureUsage;
  }

  if (snapshot.hadNavigator && typeof globalThis.navigator !== 'undefined') {
    if (snapshot.navigatorGpuDescriptor) {
      Object.defineProperty(globalThis.navigator, 'gpu', snapshot.navigatorGpuDescriptor);
    } else {
      delete globalThis.navigator.gpu;
    }
  }

  if (snapshot.markerA === undefined) {
    delete globalThis.__dopplerNodeWebgpuMarkerA;
  } else {
    globalThis.__dopplerNodeWebgpuMarkerA = snapshot.markerA;
  }
  if (snapshot.markerB === undefined) {
    delete globalThis.__dopplerNodeWebgpuMarkerB;
  } else {
    globalThis.__dopplerNodeWebgpuMarkerB = snapshot.markerB;
  }
}

function setNavigatorGpu(value) {
  if (typeof globalThis.navigator === 'undefined') {
    Object.defineProperty(globalThis, 'navigator', {
      value: value === undefined ? {} : { gpu: value },
      writable: true,
      configurable: true,
      enumerable: false,
    });
    return;
  }
  if (value === undefined) {
    delete globalThis.navigator.gpu;
    return;
  }
  Object.defineProperty(globalThis.navigator, 'gpu', {
    value,
    writable: true,
    configurable: true,
    enumerable: false,
  });
}

function clearRuntime() {
  setNavigatorGpu(undefined);
  delete globalThis.GPUBufferUsage;
  delete globalThis.GPUShaderStage;
  delete globalThis.GPUMapMode;
  delete globalThis.GPUTextureUsage;
  delete globalThis.__dopplerNodeWebgpuMarkerA;
  delete globalThis.__dopplerNodeWebgpuMarkerB;
}

{
  const snapshot = snapshotState();
  try {
    clearRuntime();
    setNavigatorGpu({ async requestAdapter() { return null; } });
    globalThis.GPUBufferUsage = { COPY_SRC: 1 };
    globalThis.GPUShaderStage = { COMPUTE: 1 };
    process.env.DOPPLER_NODE_WEBGPU_MODULE = `missing-webgpu-module-${Date.now()}`;

    const ready = await bootstrapNodeWebGPU();
    assert.equal(ready, true);
  } finally {
    restoreState(snapshot);
  }
}

{
  const snapshot = snapshotState();
  try {
    clearRuntime();
    process.env.DOPPLER_NODE_WEBGPU_MODULE = `missing-webgpu-module-${Date.now()}`;

    const ready = await bootstrapNodeWebGPU();
    assert.equal(ready, false);
  } finally {
    restoreState(snapshot);
  }
}

{
  const snapshot = snapshotState();
  const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-webgpu-direct-'));
  try {
    clearRuntime();
    const modulePath = path.join(tempDir, 'webgpu-direct.mjs');
    writeFileSync(modulePath, `
export const gpu = { async requestAdapter() { return null; } };
export const globals = { __dopplerNodeWebgpuMarkerA: 'direct' };
export const GPUBufferUsage = { COPY_SRC: 4, COPY_DST: 8 };
export const GPUShaderStage = { COMPUTE: 4 };
export const GPUMapMode = { READ: 1, WRITE: 2 };
export const GPUTextureUsage = { COPY_SRC: 1, COPY_DST: 2 };
`, 'utf8');
    process.env.DOPPLER_NODE_WEBGPU_MODULE = modulePath;

    const ready = await bootstrapNodeWebGPU();
    assert.equal(ready, true);
    assert.equal(typeof globalThis.navigator?.gpu?.requestAdapter, 'function');
    assert.equal(globalThis.GPUBufferUsage.COPY_SRC, 4);
    assert.equal(globalThis.GPUShaderStage.COMPUTE, 4);
    assert.equal(globalThis.GPUMapMode.READ, 1);
    assert.equal(globalThis.GPUTextureUsage.COPY_DST, 2);
    assert.equal(globalThis.__dopplerNodeWebgpuMarkerA, 'direct');
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
    restoreState(snapshot);
  }
}

{
  const snapshot = snapshotState();
  const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-webgpu-main-'));
  try {
    clearRuntime();
    writeFileSync(path.join(tempDir, 'package.json'), JSON.stringify({
      name: 'doppler-webgpu-main',
      version: '1.0.0',
      type: 'module',
      main: 'main.js',
    }), 'utf8');
    writeFileSync(path.join(tempDir, 'main.js'), `
export default {
  gpu: { async requestAdapter() { return null; } },
};
export const GPUBufferUsage = { COPY_SRC: 64 };
export const GPUShaderStage = { COMPUTE: 8 };
`, 'utf8');
    process.env.DOPPLER_NODE_WEBGPU_MODULE = tempDir;

    const ready = await bootstrapNodeWebGPU();
    assert.equal(ready, true);
    assert.equal(globalThis.GPUBufferUsage.COPY_SRC, 64);
    assert.equal(globalThis.GPUShaderStage.COMPUTE, 8);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
    restoreState(snapshot);
  }
}

{
  const snapshot = snapshotState();
  const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-webgpu-exports-'));
  try {
    clearRuntime();
    writeFileSync(path.join(tempDir, 'package.json'), JSON.stringify({
      name: 'doppler-webgpu-exports',
      version: '1.0.0',
      type: 'module',
      exports: {
        './node': './node-runtime.js',
      },
    }), 'utf8');
    writeFileSync(path.join(tempDir, 'node-runtime.js'), `
export function create(args) {
  if (Array.isArray(args) && args.length > 0) {
    throw new Error('retry-without-args');
  }
  return {
    gpu: { async requestAdapter() { return null; } },
  };
}
export const globals = { __dopplerNodeWebgpuMarkerB: 'exports-node' };
export const GPUBufferUsage = { COPY_SRC: 512 };
export const GPUShaderStage = { COMPUTE: 16 };
`, 'utf8');
    process.env.DOPPLER_NODE_WEBGPU_MODULE = tempDir;

    const ready = await bootstrapNodeWebGPU();
    assert.equal(ready, true);
    assert.equal(globalThis.GPUBufferUsage.COPY_SRC, 512);
    assert.equal(globalThis.GPUShaderStage.COMPUTE, 16);
    assert.equal(globalThis.__dopplerNodeWebgpuMarkerB, 'exports-node');
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
    restoreState(snapshot);
  }
}

{
  const snapshot = snapshotState();
  const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-webgpu-fallback-'));
  try {
    clearRuntime();
    writeFileSync(path.join(tempDir, 'package.json'), JSON.stringify({
      name: 'doppler-webgpu-fallback',
      version: '1.0.0',
      type: 'module',
    }), 'utf8');
    writeFileSync(path.join(tempDir, 'index.js'), `
export default { async requestAdapter() { return null; } };
export const GPUBufferUsage = { COPY_SRC: 1024 };
export const GPUShaderStage = { COMPUTE: 32 };
`, 'utf8');
    process.env.DOPPLER_NODE_WEBGPU_MODULE = tempDir;

    const ready = await bootstrapNodeWebGPU();
    assert.equal(ready, true);
    assert.equal(globalThis.GPUBufferUsage.COPY_SRC, 1024);
    assert.equal(globalThis.GPUShaderStage.COMPUTE, 32);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
    restoreState(snapshot);
  }
}

{
  const snapshot = snapshotState();
  const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-webgpu-invalid-'));
  try {
    clearRuntime();
    const modulePath = path.join(tempDir, 'webgpu-invalid.mjs');
    writeFileSync(modulePath, `
export const gpu = {};
export const GPUBufferUsage = { COPY_SRC: 1 };
export const GPUShaderStage = { COMPUTE: 1 };
`, 'utf8');
    process.env.DOPPLER_NODE_WEBGPU_MODULE = modulePath;

    const ready = await bootstrapNodeWebGPU();
    assert.equal(ready, false);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
    restoreState(snapshot);
  }
}

console.log('node-webgpu-bootstrap.test: ok');
