import assert from 'node:assert/strict';
import { readPartitionF16Boundary } from "../../src/inference/pipelines/text/partitions/partition-wire.js";
import { acquireBuffer, releaseBuffer, readBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { writeBufferAligned } from '../../src/loader/tensors/buffer-write.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { castF16ToF32 } from '../../src/gpu/kernels/cast.js';
import { float32ToFloat16 } from '../../src/formats/float16.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) {
  if (process.env.DOPPLER_REQUIRE_PARTITION_WIRE_GPU === '1') throw new Error(probe.reason);
  console.log(`partition-wire-gpu: skipped (${probe.reason})`);
  process.exit(0);
}
assert(getDevice().features.has('shader-f16'));
const report = { provider: probe.providerReceipt, cases: [] };
try {
  for (const shape of [[1, 1, 3], [1, 1, 10240], [1, 85, 10240]]) {
    const count = shape.reduce((a, b) => a * b, 1);
    const values = Float32Array.from({ length: count }, (_, i) => (i % 97 - 48) / 32);
    const input = acquireBuffer(values.byteLength, undefined, 'wire_test_input');
    let encoded, restored;
    try {
      uploadData(input, values);
      const start = performance.now();
      const bytes = await readPartitionF16Boundary(input, shape, new AbortController().signal);
      const encodeReadbackMs = performance.now() - start;
      assert.equal(bytes.byteLength, count * 2);
      assert.deepEqual(new Uint16Array(bytes), Uint16Array.from(values, float32ToFloat16));
      encoded = acquireBuffer(bytes.byteLength, undefined, 'wire_test_encoded');
      writeBufferAligned(getDevice(), encoded, bytes);
      restored = (await castF16ToF32(createTensor(encoded, 'f16', shape))).buffer;
      assert.deepEqual(new Float32Array(await readBuffer(restored, values.byteLength)), values);
      assert.deepEqual(new Float32Array(await readBuffer(input, values.byteLength)), values, 'borrowed input unchanged');
      report.cases.push({ shape, wireBytes: bytes.byteLength, executionBytes: values.byteLength, encodeReadbackMs });
    } finally { if (restored) releaseBuffer(restored); if (encoded) releaseBuffer(encoded); releaseBuffer(input); }
  }
  for (const value of [65505, -65505, Infinity, -Infinity, NaN]) {
    const input = acquireBuffer(12, undefined, 'wire_test_invalid');
    try {
      uploadData(input, new Float32Array([1, value, 2]));
      await assert.rejects(readPartitionF16Boundary(input, [1, 1, 3], new AbortController().signal), /nonfinite or overflowing/);
    } finally { releaseBuffer(input); }
  }
  const input = acquireBuffer(12, undefined, 'wire_test_cancel');
  try {
    const controller = new AbortController(); controller.abort();
    await assert.rejects(readPartitionF16Boundary(input, [1, 1, 3], controller.signal), /abort/i);
  } finally { releaseBuffer(input); }
  console.log(JSON.stringify(report));
  console.log('partition-wire-gpu: GPU finite narrowing, expansion, range guards and ownership passed');
} finally { destroyDevice(); await releaseNodeWebGPU(); }
