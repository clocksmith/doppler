import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { runRepeatChannels, recordRepeatChannels } from '../../src/gpu/kernels/repeat_channels.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { float32ToFloat16 } from '../../src/formats/float16.js';
import { createKernelRegistry, enterKernelRegistry } from '../../src/gpu/kernels/kernel-configs.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical repeat-channel verification unavailable: ${probe.reason}`);
const errors = [], receipts = [];
getDevice().addEventListener('uncapturederror', event => errors.push(event.error.message));
try {
  for (const dtype of ['f32', 'f16']) {
    for (const [inChannels, height, width, repeats] of [[53, 1, 1, 2048], [3, 17, 19, 5], [1, 1, 1, 1], [129, 1, 1, 16385]]) {
      const count = inChannels * height * width;
      const values = Float32Array.from({ length: count }, (_, i) => Math.sin(i * 0.13) * 0.1);
      const encoded = dtype === 'f16' ? Uint16Array.from(values, float32ToFloat16) : values;
      const padded = new Uint8Array(Math.ceil(encoded.byteLength / 4) * 4);
      padded.set(new Uint8Array(encoded.buffer, encoded.byteOffset, encoded.byteLength));
      const input = acquireBuffer(padded.byteLength);
      uploadData(input, padded);
      // Use explicit registry geometry to exercise a real two-dimensional grid
      // without allocating hundreds of MB for this focused operator check.
      const variant = dtype === 'f16' ? 'default_f16' : 'default';
      const leave = inChannels === 129 ? enterKernelRegistry(createKernelRegistry({ extensions: {
        repeat_channels: { variants: { [variant]: { wgsl: dtype === 'f16' ? 'repeat_channels_f16.wgsl' : 'repeat_channels.wgsl',
          entryPoint: 'main', workgroup: [32, 1, 1], wgslOverrides: { WORKGROUP_SIZE: 32 } } } },
      } })) : () => {};
      try {
        for (const recorded of [false, true]) {
          const recorder = recorded ? createCommandRecorder('repeat-channels-verification') : null;
          let output;
          try {
            const tensor = createTensor(input, dtype, [inChannels, height, width]);
            const options = { inChannels, height, width, repeats };
            output = recorder ? await recordRepeatChannels(recorder, tensor, options)
              : await runRepeatChannels(tensor, options);
            const workgroups = recorder?.getStats().dispatches[0]?.workgroups ?? null;
            if (inChannels === 129 && recorder) {
              assert.equal(workgroups[1], 2, 'Large linear work is tiled into a real two-dimensional GPU grid');
            }
            if (recorder) await recorder.submitAndWait();
            const bytes = await readBuffer(output.buffer, Math.ceil(count * repeats * (dtype === 'f16' ? 2 : 4) / 4) * 4);
            const actual = dtype === 'f16' ? new Uint16Array(bytes) : new Float32Array(bytes);
            for (let i = 0; i < count * repeats; i++) {
              const spatial = height * width, channel = Math.floor(i / spatial / repeats);
              const expected = encoded[channel * spatial + i % spatial];
              assert.equal(actual[i], expected, `${dtype} repeat output[${i}]`);
            }
            receipts.push({ dtype, inChannels, height, width, repeats, recorded, workgroups });
          } finally {
            recorder?.abort();
            if (output) releaseBuffer(output.buffer);
          }
        }
      } finally { leave(); releaseBuffer(input); }
    }
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ status: 'passed', provider: probe.providerReceipt, receipts }, null, 2));
} finally { destroyDevice(); }
