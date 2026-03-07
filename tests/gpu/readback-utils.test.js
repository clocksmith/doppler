import assert from 'node:assert/strict';

globalThis.GPUMapMode = {
  READ: 1 << 0,
  WRITE: 1 << 1,
};

const { withMappedReadBuffer, withMappedReadBuffers } = await import('../../src/gpu/readback-utils.js');

class FakeBuffer {
  constructor({ mapReject = false, size = 16 }) {
    this.mapReject = mapReject;
    this.size = size;
    this.unmapped = false;
    this.destroyed = false;
    this.range = new ArrayBuffer(size);
  }

  async mapAsync() {
    if (this.mapReject) {
      throw new Error('map failed');
    }
  }

  getMappedRange(offset = 0, size = this.size - offset) {
    return this.range.slice(offset, offset + size);
  }

  unmap() {
    this.unmapped = true;
  }

  destroy() {
    this.destroyed = true;
  }
}

{
  const buffer = new FakeBuffer({});
  const byteLength = await withMappedReadBuffer(buffer, (range) => range.byteLength);
  assert.equal(byteLength, 16);
  assert.equal(buffer.unmapped, true);
  assert.equal(buffer.destroyed, false);
}

{
  const buffer = new FakeBuffer({ mapReject: true });
  await assert.rejects(() => withMappedReadBuffer(buffer, () => 0), /map failed/);
  assert.equal(buffer.unmapped, false);
  assert.equal(buffer.destroyed, false);
}

{
  const first = new FakeBuffer({});
  const second = new FakeBuffer({ mapReject: true });
  await assert.rejects(
    () => withMappedReadBuffers(
      [
        { buffer: first, destroy: true },
        { buffer: second, destroy: true },
      ],
      () => 0
    ),
    /map failed/
  );
  assert.equal(first.unmapped, true);
  assert.equal(first.destroyed, true);
  assert.equal(second.unmapped, false);
  assert.equal(second.destroyed, true);
}

console.log('readback-utils.test: ok');
