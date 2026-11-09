import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { destroyDevice } from '../../src/gpu/device.js';
import { destroyBufferPool, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { loadQ4KDequant } from '../../src/loader/tensors/tensor-loader.js';
import { quantizeQ4_KRef, dequantQ4_KRef } from '../kernels/reference/dequant.js';

const gpu = await probeNodeGPU({ installFileFetchShim: true });
if (!gpu.ready) {
  console.log(`q4k-packing-gpu: skipped (${gpu.reason})`);
  process.exit(0);
}
const config = { useFusedQ4K: false, keepF32Weights: true, q4kLayout: 'row' };
function packedValues(count) {
  return Float32Array.from({ length: count }, (_, i) => Math.sin(i * 0.031) * 0.3 + 0.1);
}
async function check(shape, bytes, expected, layout) {
  const location = { dtype: 'Q4_K_M', role: 'other', shape, size: bytes.byteLength };
  if (layout) location.layout = layout;
  const result = await loadQ4KDequant(bytes, location, 'packing-regression', config);
  try {
    const actual = new Float32Array(await readBuffer(result.data.buffer, expected.length * 4));
    let maxAbs = 0;
    for (let i = 0; i < expected.length; i++) {
      assert.ok(Number.isFinite(actual[i]), `Non-finite value at ${i}`);
      maxAbs = Math.max(maxAbs, Math.abs(actual[i] - expected[i]));
    }
    assert.ok(maxAbs < 1e-6, `shape=${shape} layout=${layout} maxAbs=${maxAbs}`);
    console.log(`q4k-packing-gpu: shape=${shape} layout=${layout ?? 'absent'} maxAbs=${maxAbs}`);
  } finally {
    result.allocatedBuffers.forEach(releaseBuffer);
  }
}
try {
  // The converter flat-packs non-expert 3D filters, even under a row-layout recipe.
  for (const shape of [[512, 1, 3], [5, 1, 3]]) {
    const count = shape.reduce((a, b) => a * b, 1);
    const blocks = Math.ceil(count / 256);
    const bytes = quantizeQ4_KRef(packedValues(blocks * 256), blocks);
    await check(shape, bytes, dequantQ4_KRef(bytes, blocks).slice(0, count));
  }
  // Explicit row-padded expert storage and legacy 2D row storage must stay intact.
  const rows = 4;
  const columns = 3;
  const bytes = new Uint8Array(rows * 144);
  const expected = new Float32Array(rows * columns);
  for (let row = 0; row < rows; row++) {
    const block = quantizeQ4_KRef(packedValues(256).map(v => v + row * 0.2), 1);
    bytes.set(block, row * 144);
    expected.set(dequantQ4_KRef(block, 1).slice(0, columns), row * columns);
  }
  await check([2, 2, columns], bytes, expected, 'row');
  await check([rows, columns], bytes, expected);
  // Column packing pads rows of the transposed storage matrix.
  await check([columns, rows], bytes, expected, 'column');
  await assert.rejects(loadQ4KDequant(bytes.subarray(0, 144), {
    dtype: 'Q4_K_M', role: 'other', shape: [rows, columns], size: 144, layout: 'row',
  }, 'truncated-rows', config), /Q4K packed size.*expected 576.*got 144/);
} finally {
  destroyBufferPool();
  destroyDevice();
}
