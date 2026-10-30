import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createProbeServer, closeProbeServer, probeOrigin } from '../../tools/pooled-equal-weights/probe-server.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const server = await createProbeServer({ entry: 'unused', config: {}, routes: [['/src/', path.join(root, 'src')]] });
let browser;
try {
  browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu'],
    ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
  const page = await browser.newPage(), errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(`${probeOrigin(server)}/probe-config.json`);
  const result = await page.evaluate(async () => {
    const { initDevice, getDevice, getKernelCapabilities, destroyDevice } = await import('/src/gpu/device.js');
    const { acquireBuffer, uploadData, readBuffer, releaseBuffer } = await import('/src/memory/buffer-pool.js');
    const { createTensor } = await import('/src/gpu/tensor.js');
    const { createCommandRecorder } = await import('/src/gpu/command-recorder.js');
    const { runRepeatChannels, recordRepeatChannels } = await import('/src/gpu/kernels/repeat_channels.js');
    const { float32ToFloat16 } = await import('/src/formats/float16.js');
    const { createKernelRegistry, enterKernelRegistry } = await import('/src/gpu/kernels/kernel-configs.js');
    await initDevice();
    const gpuErrors = [], cases = [], device = getDevice();
    device.addEventListener('uncapturederror', event => gpuErrors.push(event.error.message));
    try {
      for (const dtype of ['f32', 'f16']) for (const [channels, spatial, repeats] of [[53, 1, 2048], [3, 323, 5], [129, 1, 16385]]) {
        const inputValues = Float32Array.from({ length: channels * spatial }, (_, i) => Math.sin(i * 0.13) * 0.1);
        const values = dtype === 'f16' ? Uint16Array.from(inputValues, float32ToFloat16) : inputValues;
        const bytes = new Uint8Array(Math.ceil(values.byteLength / 4) * 4);
        bytes.set(new Uint8Array(values.buffer));
        const buffer = acquireBuffer(bytes.byteLength); uploadData(buffer, bytes);
        const variant = dtype === 'f16' ? 'default_f16' : 'default';
        const leave = channels === 129 ? enterKernelRegistry(createKernelRegistry({ extensions: { repeat_channels: {
          variants: { [variant]: { wgsl: dtype === 'f16' ? 'repeat_channels_f16.wgsl' : 'repeat_channels.wgsl',
            entryPoint: 'main', workgroup: [32, 1, 1], wgslOverrides: { WORKGROUP_SIZE: 32 } } },
        } } })) : () => {};
        try {
          for (const recorded of [false, true]) {
            const recorder = recorded ? createCommandRecorder('browser-repeat-check') : null;
            let output;
            try {
              const tensor = createTensor(buffer, dtype, [channels, 1, spatial]);
              const options = { inChannels: channels, height: 1, width: spatial, repeats };
              output = recorder ? await recordRepeatChannels(recorder, tensor, options) : await runRepeatChannels(tensor, options);
              const workgroups = recorder?.getStats().dispatches[0]?.workgroups ?? null;
              if (channels === 129 && recorded && workgroups[1] !== 2) throw Error('Expected two-dimensional grid');
              if (recorder) await recorder.submitAndWait();
              const data = await readBuffer(output.buffer, Math.ceil(channels * spatial * repeats * (dtype === 'f16' ? 2 : 4) / 4) * 4);
              const actual = dtype === 'f16' ? new Uint16Array(data) : new Float32Array(data);
              let index = 0;
              for (let channel = 0; channel < channels; channel++) for (let repeat = 0; repeat < repeats; repeat++) {
                for (let s = 0; s < spatial; s++, index++) if (actual[index] !== values[channel * spatial + s]) throw Error(`Repeat mismatch at ${index}`);
              }
              cases.push({ dtype, channels, spatial, repeats, recorded, workgroups });
            } finally { await recorder?.abort(); if (output) releaseBuffer(output.buffer); }
          }
        } finally { leave(); releaseBuffer(buffer); }
      }
      await device.queue.onSubmittedWorkDone();
      return { cases, gpuErrors, adapter: getKernelCapabilities().adapterInfo };
    } finally { destroyDevice(); }
  });
  assert.deepEqual(result.gpuErrors, []); assert.deepEqual(errors, []);
  console.log(JSON.stringify({ test: 'repeat-channels-browser', passed: true, browser: browser.version(), ...result }));
} finally { await browser?.close(); await closeProbeServer(server); }
