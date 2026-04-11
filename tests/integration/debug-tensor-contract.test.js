import assert from 'node:assert/strict';

import { snapshotTensor } from '../../src/debug/tensor.js';
import { setGPUDevice } from '../../src/debug/config.js';

setGPUDevice(null);

const result = await snapshotTensor({ size: 16 }, [4], 'f32');

assert.equal(result.ok, false);
assert.match(String(result.error || ''), /GPU device not initialized/);
assert.deepEqual(result.shape, [4]);
assert.equal(result.sample.length, 0);

globalThis.GPUBufferUsage ??= {
  COPY_DST: 0x0008,
  MAP_READ: 0x0001,
};
globalThis.GPUMapMode ??= {
  READ: 1 << 0,
};

class FakeBuffer {
  constructor(size) {
    this.size = size;
    this.data = new Uint8Array(size);
  }

  async mapAsync() {}

  getMappedRange() {
    return this.data.slice().buffer;
  }

  unmap() {}

  destroy() {}
}

const fakeDevice = {
  queue: {
    submit(commandBuffers) {
      for (const commandBuffer of commandBuffers) {
        for (const op of commandBuffer.ops) {
          const bytes = op.src.data.subarray(op.srcOffset, op.srcOffset + op.size);
          op.dst.data.set(bytes, op.dstOffset);
        }
      }
    },
  },
  createBuffer({ size }) {
    return new FakeBuffer(size);
  },
  createCommandEncoder() {
    const ops = [];
    return {
      copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
        ops.push({ src, srcOffset, dst, dstOffset, size });
      },
      finish() {
        return { ops };
      },
    };
  },
};

const f16Buffer = new FakeBuffer(8);
new Uint16Array(f16Buffer.data.buffer).set([0x3c00, 0x7e00, 0xbc00, 0x0000]);
setGPUDevice(fakeDevice);

const f16Result = await snapshotTensor(f16Buffer, [4], 'f16');

assert.equal(f16Result.ok, true);
assert.equal(f16Result.dtype, 'f16');
assert.equal(f16Result.sample[0], 1);
assert.equal(Number.isNaN(f16Result.sample[1]), true);
assert.equal(f16Result.sample[2], -1);
assert.equal(f16Result.sample[3], 0);
assert.equal(f16Result.hasNaN, true);

setGPUDevice(null);

console.log('debug-tensor-contract.test: ok');
