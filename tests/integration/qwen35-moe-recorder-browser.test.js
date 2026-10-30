import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createProbeServer, closeProbeServer, probeOrigin } from '../../tools/pooled-equal-weights/probe-server.js';
import { quantizeToQ4KMRowWise } from '../../src/converter/quantizer.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const T = 2, H = 256, I = 256, E = 3, S = 3;
const values = (n, f) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * f) * 0.1);
const fixture = { T, H, I, E, S, input: Array.from(values(T * H, 0.13)),
  router: Array.from(values(E * H, 0.07)),
  gateUp: Array.from(quantizeToQ4KMRowWise(values(E * 2 * I * H, 0.03), [E, 2 * I, H]).quantized),
  down: Array.from(quantizeToQ4KMRowWise(values(E * H * I, 0.09), [E, H, I]).quantized),
  sharedGate: Array.from(values(S * H, 0.11)), sharedUp: Array.from(values(S * H, 0.05)),
  sharedDown: Array.from(values(H * S, 0.17)), sharedRouter: Array.from(values(H, 0.19)) };
const server = await createProbeServer({ entry: 'unused', config: {}, routes: [['/src/', path.join(root, 'src')]] });
let browser;
try {
  browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu'],
    ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
  const page = await browser.newPage(), errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(`${probeOrigin(server)}/probe-config.json`);
  const result = await page.evaluate(async f => {
    const { initDevice, getDevice, getKernelCapabilities, destroyDevice } = await import('/src/gpu/device.js');
    const { acquireBuffer, uploadData, readBuffer, releaseBuffer } = await import('/src/memory/buffer-pool.js');
    const { createTensor } = await import('/src/gpu/tensor.js');
    const { createWeightBuffer } = await import('/src/gpu/weight-buffer.js');
    const { createCommandRecorder } = await import('/src/gpu/command-recorder.js');
    const { MoERouter } = await import('/src/inference/moe-router.js');
    const { runMoEFFNGPU } = await import('/src/inference/pipelines/text/ffn/moe.js');
    const { getRuntimeConfig, setRuntimeConfig } = await import('/src/config/runtime.js');
    await initDevice();
    const owned = [], gpuErrors = [], device = getDevice(), previous = getRuntimeConfig();
    device.addEventListener('uncapturederror', event => gpuErrors.push(event.error.message));
    const upload = values => { const buffer = acquireBuffer(values.byteLength); owned.push(buffer); uploadData(buffer, values); return buffer; };
    const matrix = (values, shape) => createWeightBuffer(upload(new Float32Array(values)), 'f32', 'row', shape);
    let recorder;
    try {
      setRuntimeConfig({ inference: { moe: { routing: { activeExpertSelection: 'topk-route' } } } });
      const { T, H, I, E, S } = f;
      const source = upload(new Float32Array(f.input)), pending = upload(new Float32Array(T * H));
      const router = new MoERouter({ numExperts: E, topK: 2, hiddenSize: H, normalizeWeights: true });
      router.loadWeights(matrix(f.router, [E, H]));
      const packed = { numExperts: E, expertFormat: 'qwen3_5',
        gateUp: createWeightBuffer(upload(new Uint8Array(f.gateUp)), 'q4k', 'row', [E, 2 * I, H]),
        down: createWeightBuffer(upload(new Uint8Array(f.down)), 'q4k', 'row', [E, H, I]) };
      const context = { config: { modelType: 'transformer', hiddenSize: H, intermediateSize: I,
        moeExpertIntermediateSize: I, moeSharedExpertIntermediateSize: S, numExperts: E, moeTopK: 2,
        expertFormat: 'qwen3_5', hiddenActivation: 'silu', swigluLimit: null, rmsNormEps: 1e-6 },
        moeRouter: router, expertWeights: new Map([['layer_0_expert_0', packed]]),
        expertLoader: { loadExpert() { throw Error('Resident bank must not reload'); } },
        weights: new Map([['layer_0', { sharedExpertGate: matrix(f.sharedGate, [S, H]),
          sharedExpertUp: matrix(f.sharedUp, [S, H]), sharedExpertDown: matrix(f.sharedDown, [H, S]),
          sharedExpertRouter: matrix(f.sharedRouter, [1, H]) }]]), kernelPath: null, executionPolicies: null };
      const control = await runMoEFFNGPU(0, createTensor(source, 'f32', [T, H]), T, context); owned.push(control.buffer);
      const expected = new Float32Array(await readBuffer(control.buffer, T * H * 4));
      if (!expected.some(value => Math.abs(value) > 0.0001)) throw Error('Degenerate stale-input fixture');
      recorder = createCommandRecorder('browser-moe-ordering');
      recorder.getEncoder().copyBufferToBuffer(source, 0, pending, 0, T * H * 4);
      const output = await runMoEFFNGPU(0, createTensor(pending, 'f32', [T, H]), T, { ...context, recorder }); owned.push(output.buffer);
      await recorder.submitAndWait();
      const observed = new Float32Array(await readBuffer(output.buffer, T * H * 4));
      let maxAbs = 0;
      for (let i = 0; i < expected.length; i++) {
        maxAbs = Math.max(maxAbs, Math.abs(observed[i] - expected[i]));
        if (!Number.isFinite(observed[i]) || Math.abs(observed[i] - expected[i]) > 0.00001) throw Error(`Recorded MoE mismatch at ${i}`);
      }
      return { maxAbs, gpuErrors, adapter: getKernelCapabilities().adapterInfo, recorder: recorder.getStats() };
    } finally {
      await recorder?.abort();
      for (const buffer of owned) releaseBuffer(buffer);
      setRuntimeConfig(previous); destroyDevice();
    }
  }, fixture);
  assert.deepEqual(errors, []); assert.deepEqual(result.gpuErrors, []);
  console.log(JSON.stringify({ test: 'qwen35-moe-recorder-browser', passed: true, browser: browser.version(), ...result }));
} finally { await browser?.close(); await closeProbeServer(server); }
