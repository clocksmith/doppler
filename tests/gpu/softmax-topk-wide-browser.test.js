import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import { chromium } from 'playwright';

const root = process.cwd();
const server = http.createServer(async (request, response) => {
  const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
  if (pathname === '/') { response.end('<!doctype html><title>GPU qualification</title>'); return; }
  const file = path.resolve(root, '.' + pathname);
  if (!file.startsWith(root + path.sep)) { response.writeHead(403).end(); return; }
  try {
    const type = file.endsWith('.js') ? 'text/javascript' : file.endsWith('.json') ? 'application/json' : 'text/plain';
    response.setHeader('Content-Type', type); response.end(await fs.readFile(file));
  } catch { response.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const result = await page.evaluate(async () => {
    const { initDevice, getDevice, getKernelCapabilities, destroyDevice } = await import('/src/gpu/device.js');
    const { acquireBuffer, uploadData, readBuffer, releaseBuffer, getBufferPool, destroyBufferPool } = await import('/src/memory/buffer-pool.js');
    const { runSoftmaxTopK, recordSoftmaxTopK } = await import('/src/gpu/kernels/softmax.js');
    const { CommandRecorder } = await import('/src/gpu/command-recorder.js');
    await initDevice();
    const hardware = getKernelCapabilities().adapterInfo;
    if (String(hardware.description).toLowerCase().includes('swiftshader')) throw Error('Software adapter is not physical GPU proof.');
    const errors = [];
    getDevice().addEventListener('uncapturederror', event => errors.push(event.error.message));
    const owned = [];
    try {
      const n = 512, k = 10;
      const logits = Float32Array.from({ length: n }, (_, index) => ((index * 137) % 521) / 32);
      const order = [...logits].map((value, index) => ({ value, index })).sort((a, b) => b.value - a.value).slice(0, k);
      const total = order.reduce((sum, entry) => sum + Math.exp(entry.value - order[0].value), 0);
      const input = acquireBuffer(logits.byteLength); owned.push(input); uploadData(input, logits);
      for (const recorded of [false, true]) {
        const recorder = recorded ? new CommandRecorder(getDevice(), 'browser_wide_topk') : null;
        const options = { variant: 'fused_ranked_512', normalize: true, inputDtype: 'f32', weightsDtype: 'f32' };
        const output = recorder ? await recordSoftmaxTopK(recorder, input, 1, n, k, options)
          : await runSoftmaxTopK(input, 1, n, k, options);
        owned.push(output.indices, output.weights);
        if (recorder) await recorder.submitAndWait();
        const indices = new Uint32Array(await readBuffer(output.indices, k * 4));
        const weights = new Float32Array(await readBuffer(output.weights, k * 4));
        for (let pick = 0; pick < k; pick++) {
          const expected = Math.exp(order[pick].value - order[0].value) / total;
          if (indices[pick] !== order[pick].index || Math.abs(weights[pick] - expected) > 2e-7 + 2e-6 * expected) {
            throw Error(`Browser routing mismatch at ${pick}, recorded=${recorded}`);
          }
        }
      }
      if (errors.length) throw Error(errors.join('; '));
      return { hardware, experts: n, topK: k, immediate: 'pass', recorded: 'pass', errors };
    } finally {
      await getDevice().queue.onSubmittedWorkDone();
      for (const buffer of owned) releaseBuffer(buffer);
      await getBufferPool().settleDeferredDestruction();
      if (getBufferPool().getStats().activeBuffers !== 0) throw Error('Browser probe leaked buffers.');
      destroyBufferPool(); destroyDevice();
    }
  });
  assert.equal(result.immediate, 'pass');
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
