import assert from 'node:assert/strict';

const { parseGGUF } = await import('../../src/formats/gguf/types.js');

function writeUint64(view, offset, value) {
  const low = Number(value & 0xffffffffn);
  const high = Number((value >> 32n) & 0xffffffffn);
  view.setUint32(offset, low, true);
  view.setUint32(offset + 4, high, true);
}

{
  const buffer = new ArrayBuffer(24);
  const view = new DataView(buffer);
  view.setUint32(0, 0x46554747, true);
  view.setUint32(4, 3, true);
  writeUint64(view, 8, BigInt(Number.MAX_SAFE_INTEGER) + 1n);
  writeUint64(view, 16, 0n);

  assert.throws(
    () => parseGGUF(buffer),
    /tensor count exceeds JavaScript safe integer range/
  );
}

console.log('gguf-types-safe-int.test: ok');
