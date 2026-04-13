import assert from 'node:assert/strict';

class NativeBuffer {}

const ORIGINAL_GPU_BUFFER = globalThis.GPUBuffer;
globalThis.GPUBuffer = NativeBuffer;

const { isGpuBufferInstance } = await import('../../src/gpu/weight-buffer.js');

const providerBuffer = {
  _raw: {},
  size: 64,
  usage: 0x0004 | 0x0008,
  async mapAsync() {},
  getMappedRange() {
    return new ArrayBuffer(64);
  },
  unmap() {},
  destroy() {},
};

try {
  assert.equal(isGpuBufferInstance(new NativeBuffer()), true);
  assert.equal(isGpuBufferInstance(providerBuffer), true);
  assert.equal(isGpuBufferInstance({
    size: 64,
    usage: 0x0004 | 0x0008,
    destroy() {},
  }), false);
} finally {
  if (ORIGINAL_GPU_BUFFER === undefined) {
    delete globalThis.GPUBuffer;
  } else {
    globalThis.GPUBuffer = ORIGINAL_GPU_BUFFER;
  }
}

console.log('weight-buffer-provider-buffer.test: ok');
