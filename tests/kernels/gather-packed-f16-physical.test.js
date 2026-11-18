import assert from 'node:assert/strict';
import { bootstrapNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { initDevice, getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { runGatherSplit, recordGatherSplit } from '../../src/gpu/kernels/gather.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { f32ToF16Array } from '../../src/inference/kv-cache/types.js';

// Deliberately avoid gpu-probe.js: it compiles an f16 shader, which this target lacks.
if (process.env.DOPPLER_RUN_PACKED_EMBEDDING_GPU !== '1') throw new Error('Explicit maintenance authorization required: DOPPLER_RUN_PACKED_EMBEDDING_GPU=1');
const bootstrap = await bootstrapNodeWebGPU();
assert(bootstrap.ok, 'Native WebGPU bootstrap required');
installNodeFileFetchShim();
await initDevice();
const device = getDevice(), buffers = [];
function upload(values) {
  const b = device.createBuffer({ size: values.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  buffers.push(b); device.queue.writeBuffer(b, 0, values); return b;
}
try {
  const width = 8, rows = 10;
  const source = Float32Array.from({ length: rows * width }, (_, i) => (i - 37) / 8);
  const sections = [0, 4, 8].map(start => ({ rowStart: start, rowCount: Math.min(4, rows - start),
    buffer: upload(f32ToF16Array(source.slice(start * width, Math.min(start + 4, rows) * width))) }));
  const embedding = { kind: 'split_weight_buffer', dtype: 'f16', layout: 'row', shape: [rows, width], sections, metadata: { packing: 'packed-f16-u32', splitGatherSectionCount: 4 } };
  // Both sides of section boundaries, final row, invalid token, and a nonzero index offset.
  const tokenIds = [99, 0, 3, 4, 7, 8, 9, 4, 10], ids = upload(new Uint32Array(tokenIds));
  for (const recorded of [false, true]) for (const hiddenOffset of [0, 1, 4]) {
    const outputWidth = 4;
    let tensor, recorder;
    try {
      const options = { embeddingDtype: 'f16', outputDtype: 'f32', indexOffset: 1, inputHiddenSize: width, hiddenOffset };
      if (recorded) { recorder = createCommandRecorder('split4-f32-parity');
        tensor = await recordGatherSplit(recorder, ids, embedding, tokenIds.length - 1, outputWidth, rows, options);
        await recorder.submitAndWait();
      } else tensor = await runGatherSplit(ids, embedding, tokenIds.length - 1, outputWidth, rows, options);
      const actual = new Float32Array(await readBuffer(tensor.buffer));
      const expected = tokenIds.slice(1).flatMap(id => id < rows ? [...source.slice(id * width + hiddenOffset, id * width + hiddenOffset + outputWidth)] : Array(outputWidth).fill(0));
      assert.deepEqual([...actual.slice(0, expected.length)], expected);
    } finally { await recorder?.abort(); if (tensor) releaseBuffer(tensor.buffer); }
  }
  console.log(JSON.stringify({ test: 'gather-packed-f16-physical', passed: true, hasF16: getKernelCapabilities().hasF16, exact: true, recordedAndImmediate: true, hiddenOffsets: [0, 1, 4], sourceRows: rows, provider: bootstrap.receipt }));
} finally { for (const b of buffers) b.destroy(); destroyDevice(); await releaseNodeWebGPU(); }
