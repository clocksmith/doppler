import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createProbeServer, closeProbeServer, probeOrigin } from '../../tools/pooled-equal-weights/probe-server.js';
import { quantizeToQ4KMRowWise } from '../../src/converter/quantizer.js';
import { dequantizeQ4KMRowWise } from '../../src/formats/q4k-dequant.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const fixtures = [1, 257, 2048].flatMap(K => ['token', 'route'].map(inputMode => {
  const N = 37, numExperts = 3, topK = 2, numRoutes = 4;
  const indices = [2, 0, 1, 2], weights = [0.25, 0.75, 0.6, 0.4];
  const rows = inputMode === 'token' ? numRoutes / topK : numRoutes;
  const input = Float32Array.from({ length: rows * K }, (_, i) => Math.sin(i * 0.13) * 0.1);
  const { quantized } = quantizeToQ4KMRowWise(Float32Array.from({ length: numExperts * N * K },
    (_, i) => Math.cos(i * 0.07) * 0.2), [numExperts, N, K]);
  const decoded = dequantizeQ4KMRowWise(quantized, [numExperts, N, K]);
  const expected = Array.from({ length: numRoutes * N }, (_, index) => {
    const route = Math.floor(index / N), column = index % N;
    const row = inputMode === 'token' ? Math.floor(route / topK) : route;
    let sum = 0;
    for (let k = 0; k < K; k++) sum += input[row * K + k] * decoded[(indices[route] * N + column) * K + k];
    return sum;
  });
  return { K, N, numExperts, numRoutes, topK, inputMode, rows, input: Array.from(input),
    quantized: Array.from(quantized), indices, weights, expected };
}));
const server = await createProbeServer({ entry: 'unused', config: {},
  routes: [['/src/', path.join(root, 'src')], ['/node_modules/', path.join(root, 'node_modules')]] });
let browser;
try {
  browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu'],
    ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
  const page = await browser.newPage(), errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(`${probeOrigin(server)}/probe-config.json`);
  const result = await page.evaluate(async fixtures => {
    const { initDevice, getDevice, getKernelCapabilities, destroyDevice } = await import('/src/gpu/device.js');
    const { acquireBuffer, uploadData, readBuffer, releaseBuffer } = await import('/src/memory/buffer-pool.js');
    const { createTensor } = await import('/src/gpu/tensor.js');
    const { createWeightBuffer } = await import('/src/gpu/weight-buffer.js');
    const { createCommandRecorder } = await import('/src/gpu/command-recorder.js');
    const { runRouteQ4Matmul, recordRouteQ4Matmul, runRouteScatter, recordRouteScatter } = await import('/src/gpu/kernels/moe-route.js');
    await initDevice();
    const gpuErrors = [], device = getDevice(), cases = [];
    device.addEventListener('uncapturederror', event => gpuErrors.push(event.error.message));
    try {
      for (const fixture of fixtures) for (const recorded of [false, true]) {
        const owned = [], upload = values => {
          const buffer = acquireBuffer(values.byteLength); owned.push(buffer); uploadData(buffer, values); return buffer;
        };
        const recorder = recorded ? createCommandRecorder('browser-route-check') : null;
        try {
          const tensor = createTensor(upload(new Float32Array(fixture.input)), 'f32', [fixture.rows, fixture.K]);
          const bank = createWeightBuffer(upload(new Uint8Array(fixture.quantized)), 'q4k', 'row', [fixture.numExperts, fixture.N, fixture.K]);
          const indices = upload(new Uint32Array(fixture.indices)), weights = upload(new Float32Array(fixture.weights));
          const options = { ...fixture, alpha: 1 };
          const output = recorder ? await recordRouteQ4Matmul(recorder, tensor, indices, bank, options)
            : await runRouteQ4Matmul(tensor, indices, bank, options);
          owned.push(output.buffer);
          const scatter = { numTokens: fixture.numRoutes / fixture.topK, hiddenSize: fixture.N, topK: fixture.topK };
          const combined = recorder ? await recordRouteScatter(recorder, output, weights, scatter)
            : await runRouteScatter(output, weights, scatter);
          owned.push(combined.buffer);
          if (recorder) await recorder.submitAndWait();
          cases.push({ K: fixture.K, inputMode: fixture.inputMode, recorded,
            actual: Array.from(new Float32Array(await readBuffer(output.buffer, fixture.numRoutes * fixture.N * 4))),
            combined: Array.from(new Float32Array(await readBuffer(combined.buffer, scatter.numTokens * fixture.N * 4))) });
        } finally {
          await recorder?.abort();
          for (const buffer of owned) releaseBuffer(buffer);
        }
      }
      await device.queue.onSubmittedWorkDone();
      return { cases, gpuErrors, adapter: getKernelCapabilities().adapterInfo };
    } finally { destroyDevice(); }
  }, fixtures);
  let maxAbs = 0;
  for (const row of result.cases) {
    const fixture = fixtures.find(value => value.K === row.K && value.inputMode === row.inputMode);
    const expectedCombined = Array.from({ length: row.combined.length }, (_, i) => {
      const token = Math.floor(i / fixture.N), column = i % fixture.N;
      let sum = 0;
      for (let k = 0; k < fixture.topK; k++) sum += fixture.weights[token * fixture.topK + k] * fixture.expected[(token * fixture.topK + k) * fixture.N + column];
      return sum;
    });
    for (const [actual, expected] of [[row.actual, fixture.expected], [row.combined, expectedCombined]]) {
      for (let i = 0; i < expected.length; i++) {
        const error = Math.abs(actual[i] - expected[i]); maxAbs = Math.max(maxAbs, error);
        assert.ok(Number.isFinite(actual[i]) && error <= 0.0001 + 0.00001 * Math.abs(expected[i]));
      }
    }
  }
  assert.deepEqual(result.gpuErrors, []); assert.deepEqual(errors, []);
  console.log(JSON.stringify({ test: 'qwen35-route-browser', passed: true, cases: result.cases.length,
    adapter: result.adapter, browser: browser.version(), maxAbs }));
} finally {
  await browser?.close();
  await closeProbeServer(server);
}
