import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { recordCheckFiniteness } from '../../src/gpu/kernels/check-finiteness.js';
import { recordCheckFinitenessF32 } from '../../src/gpu/kernels/check-finiteness-f32.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical finiteness oracle requires WebGPU: ${probe.reason}`);
const device = getDevice(), errors = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const status = acquireBuffer(16);
let cases = 0;
try {
  for (const dtype of ['f32', 'f16']) {
    if (dtype === 'f16' && !device.features.has('shader-f16')) throw new Error('F16 oracle requires shader-f16.');
    const threshold = dtype === 'f32' ? 65504 : 1;
    // Include a single failing lane in a second workgroup, both signs and non-finite values.
    const fixtures = dtype === 'f32'
      ? [[65504, false], [-65504, false], [65505, true], [-65505, true], [Infinity, true], [-Infinity, true], [NaN, true]]
      : [[0x3c00, false], [0xbc00, false], [0x3c01, true], [0xbc01, true], [0x7c00, true], [0xfc00, true], [0x7e00, true]];
    for (const [value, expected] of fixtures) {
      const data = dtype === 'f32' ? new Float32Array(258) : new Uint16Array(258);
      data[256] = value;
      const input = acquireBuffer(data.byteLength);
      try {
        uploadData(input, data); uploadData(status, new Uint32Array(4));
        const recorder = createCommandRecorder(`finiteness-${dtype}`);
        const run = dtype === 'f32' ? recordCheckFinitenessF32 : recordCheckFiniteness;
        await run(recorder, input, 257, status, 9, 17, threshold);
        await run(recorder, input, 257, status, 11, 23, threshold);
        await recorder.submitAndWait();
        const result = Array.from(new Uint32Array(await readBuffer(status, 16)));
        assert.deepEqual(result, expected ? [1, 9, 17, 0] : [0, 0, 0, 0]);
        cases++;
      } finally { releaseBuffer(input); }
    }
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ cases, exact: true, firstFailureMetadataPreserved: true, gpuErrors: errors }));
} finally {
  await device.queue.onSubmittedWorkDone(); releaseBuffer(status); destroyDevice();
}
