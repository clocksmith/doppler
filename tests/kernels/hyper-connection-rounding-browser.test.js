import assert from 'node:assert/strict';
import path from 'node:path';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../../src/tooling/node-browser-command-runner.js';

const server = await createStaticFileServer({ rootDir: path.resolve(import.meta.dirname, '../..'), port: 0 });
let browser;
try {
  const args = ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'];
  if (process.platform === 'darwin') args.push('--use-angle=metal');
  if (process.platform === 'linux') args.push('--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface');
  browser = await chromium.launch({ channel: 'chrome', headless: true, args });
  const page = await browser.newPage();
  await page.goto(`${server.baseUrl}/tests/capsule/browser-capsule-qualification.html`);
  const result = await page.evaluate(async () => {
    const gpu = await import('/src/gpu/device.js');
    const memory = await import('/src/memory/buffer-pool.js');
    const { createTensor } = await import('/src/gpu/tensor.js');
    const { runHyperConnectionBranch, runHyperConnectionInject } = await import('/src/gpu/kernels/hyper-connection.js');
    await gpu.initDevice();
    const adapter = gpu.getKernelCapabilities().adapterInfo;
    if (/swiftshader|llvmpipe|lavapipe|software/i.test(JSON.stringify(adapter))) throw new Error('Physical adapter required');
    const owned = new Set(), errors = [];
    gpu.getDevice().addEventListener('uncapturederror', event => errors.push(event.error.message));
    const upload = (data, shape) => {
      const buffer = memory.acquireBuffer(data.byteLength);
      owned.add(buffer);
      memory.uploadData(buffer, data);
      return createTensor(buffer, 'f32', shape);
    };
    try {
      const tokens = 3, streams = 4, width = 257, a = Math.fround(1 + 2 ** -23), b = Math.fround(1 - 2 ** -23);
      const mix = new Float32Array(tokens * streams * width), norm = new Float32Array(mix.length);
      for (let t = 0; t < tokens; t++) for (let w = 0; w < width; w++) {
        mix[t * streams * width + w] = 1;
        norm[t * streams * width + w] = w % 2 ? 1 : -1;
        mix[(t * streams + 1) * width + w] = b;
        norm[(t * streams + 1) * width + w] = a;
      }
      const options = { numTokens: tokens, streamCount: streams, streamWidth: width };
      const branch = await runHyperConnectionBranch(upload(mix, [tokens, streams * width]), upload(norm, [tokens, streams * width]), options);
      owned.add(branch.buffer);
      const inject = await runHyperConnectionInject(
        upload(Float32Array.from({ length: norm.length }, (_, i) => i % width % 2 ? 1 : -1), [tokens, streams * width]),
        upload(new Float32Array(tokens * width).fill(a), [tokens, width]),
        upload(new Float32Array(tokens * streams).fill(b), [tokens, streams]), options);
      owned.add(inject.buffer);
      const rows = [];
      for (const [operation, tensor, count, nonzero] of [['branch', branch, tokens * width, 0.5], ['inject', inject, norm.length, 2]]) {
        const values = new Float32Array(await memory.readBuffer(tensor.buffer, count * 4));
        rows.push({ operation, count, mismatches: values.filter((v, i) => v !== (i % width % 2 ? nonzero : 0)).length });
      }
      return { adapter, errors, rows };
    } finally {
      await gpu.getDevice().queue.onSubmittedWorkDone();
      for (const buffer of owned) memory.releaseBuffer(buffer);
      gpu.destroyDevice();
    }
  });
  console.log(JSON.stringify({ test: 'hyper-connection-rounding-browser', browserVersion: browser.version(), ...result,
    scope: 'Physical browser arithmetic-boundary regression; no model qualification or timing' }));
  assert.deepEqual(result.errors, []);
  assert(result.rows.every(row => row.mismatches === 0));
} finally {
  await browser?.close();
  await server.close();
}
