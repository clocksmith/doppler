// Physical decode regression: captured operands resized for chunk and mask boundaries.
// The Float64 oracle is independent of the online algorithm. Not model acceptance.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';
import { float16ToFloat32 } from '../../src/converter/quantizer.js';

const [shaderPath, destination] = process.argv.slice(2);
assert(shaderPath && destination, 'Supply complete shader and receipt paths');
const source = await readFile(shaderPath, 'utf8');
const fixtureBytes = gunzipSync(await readFile(new URL('../fixtures/attention-decode-prompt2-regression.json.gz', import.meta.url)));
const fixture = JSON.parse(fixtureBytes);
const inputs = fixture.data[0].captures.captures.flatMap(row => row.records).find(row => row.boundary === 'inputs');
const tensor = role => Buffer.from(inputs.tensors.find(row => row.role === role).data, 'base64');
const originalQ = tensor('q'), originalK = tensor('cachedK'), originalV = tensor('cachedV');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const cases = [
  ...[0, 1, 22, 191, 255, 256, 257, 511, 512, 769].map(kvLen => ({ headDim: 256, kvLen })),
  ...[64, 128, 512].flatMap(headDim => [191, 257, 769].map(kvLen => ({ headDim, kvLen }))),
  { headDim: 256, kvLen: 769, window: 64 },
  { headDim: 256, kvLen: 769, window: 257 },
  { headDim: 256, kvLen: 511, queryPosition: 190 },
  { headDim: 256, kvLen: 769, softcap: 2 },
  { headDim: 256, kvLen: 511, workgroupSize: 128 },
];
const rows = cases.map(({ headDim, kvLen, window = 0, softcap = 0,
  queryPosition = Math.max(0, kvLen - 1), workgroupSize = 256 }) => {
  const numHeads = 8, numKVHeads = 2;
  const q = new Float32Array(numHeads * headDim);
  const k = new Uint16Array(Math.max(2, kvLen * numKVHeads * headDim)), v = new Uint16Array(k.length);
  for (let h = 0; h < numHeads; h++) for (let d = 0; d < headDim; d++) {
    q[h * headDim + d] = originalQ.readFloatLE((h * 256 + d % 256) * 4);
  }
  for (let p = 0; p < kvLen; p++) for (let h = 0; h < numKVHeads; h++) for (let d = 0; d < headDim; d++) {
    const offset = ((p % 191 * numKVHeads + h) * 256 + d % 256) * 2;
    const index = (p * numKVHeads + h) * headDim + d;
    k[index] = originalK.readUInt16LE(offset); v[index] = originalV.readUInt16LE(offset);
  }
  const scale = Math.fround(1 / Math.sqrt(headDim));
  const expected = [];
  for (let h = 0; h < numHeads; h++) {
    const kvHead = Math.floor(h / (numHeads / numKVHeads));
    const scores = Array.from({ length: kvLen }, (_, p) => {
      if (p > queryPosition || (window > 0 && p + window <= queryPosition)) return -Infinity;
      let score = 0;
      for (let d = 0; d < headDim; d++) score += q[h * headDim + d]
        * float16ToFloat32(k[(p * numKVHeads + kvHead) * headDim + d]);
      score *= scale;
      return softcap > 0 ? Math.tanh(score / softcap) * softcap : score;
    });
    const maximum = Math.max(...scores), probabilities = scores.map(score => Math.exp(score - maximum));
    const denominator = probabilities.reduce((a, b) => a + b, 0);
    for (let d = 0; d < headDim; d++) {
      let value = 0;
      for (let p = 0; p < kvLen; p++) value += probabilities[p]
        * float16ToFloat32(v[(p * numKVHeads + kvHead) * headDim + d]);
      expected.push(kvLen ? value / denominator : 0);
    }
  }
  return { headDim, kvLen, numHeads, numKVHeads, window, softcap, queryPosition, workgroupSize,
    scale, q: Array.from(q), k: Array.from(k), v: Array.from(v), expected };
});
const receipt = { scope: 'Captured operands resized for independent chunk/mask regressions; not frozen model qualification',
  shaderSha256: hash(source), fixtureSha256: hash(fixtureBytes), host: process.platform, results: [] };
assert(['linux', 'darwin'].includes(process.platform));
const server = createServer((_request, response) => response.end('<!doctype html>'));
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu',
    ...(process.platform === 'darwin' ? ['--use-angle=metal']
      : ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'])] });
  receipt.browser = browser.version();
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  for (const row of rows) {
    const actual = await page.evaluate(async ({ source, row }) => {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter || adapter.info.isFallbackAdapter) throw Error('Physical GPU required');
      const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16', 'subgroups'] });
      const owned = [], errors = [];
      device.addEventListener('uncapturederror', event => errors.push(event.error.message));
      const buffer = (size, usage, data) => {
        const b = device.createBuffer({ size: Math.max(4, size), usage }); owned.push(b);
        if (data) device.queue.writeBuffer(b, 0, data); return b;
      };
      let staging;
      try {
        const module = device.createShaderModule({ code: source });
        const diagnostics = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
        if (diagnostics.length) throw Error(diagnostics.map(row => row.message).join('; '));
        const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: {
          module, entryPoint: 'main', constants: { WORKGROUP_SIZE: row.workgroupSize } } });
        const bytes = new ArrayBuffer(64), view = new DataView(bytes);
        const uniforms = [row.numHeads, row.numKVHeads, row.headDim, row.kvLen, 1, row.scale,
          1, row.queryPosition, row.softcap, row.window, 0, 0, 256, 0, 0];
        uniforms.forEach((value, index) => index === 5 || index === 8
          ? view.setFloat32(index * 4, value, true) : view.setUint32(index * 4, value, true));
        const output = buffer(row.numHeads * row.headDim * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const input = (data, usage = GPUBufferUsage.STORAGE) => buffer(data.byteLength,
          usage | GPUBufferUsage.COPY_DST, data);
        const resources = [input(bytes, GPUBufferUsage.UNIFORM), input(new Float32Array(row.q)),
          input(new Uint16Array(row.k)), input(new Uint16Array(row.v)), output,
          input(new Uint32Array([row.kvLen])), input(new Uint32Array(1))];
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
          entries: resources.map((resource, binding) => ({ binding, resource: { buffer: resource } })) });
        staging = buffer(output.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(row.numHeads); pass.end();
        encoder.copyBufferToBuffer(output, 0, staging, 0, output.size);
        device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
        return { values: Array.from(new Float32Array(staging.getMappedRange().slice(0))), errors,
          adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture } };
      } finally {
        if (staging?.mapState === 'mapped') staging.unmap();
        await device.queue.onSubmittedWorkDone().catch(() => {});
        owned.forEach(b => b.destroy()); device.destroy();
      }
    }, { source, row });
    const errors = actual.values.map((v, i) => Math.abs(v - row.expected[i]));
    const result = { ...Object.fromEntries(Object.entries(row).filter(([key]) => !['q', 'k', 'v', 'expected'].includes(key))),
      ...actual, maxError: Math.max(...errors), rmsError: Math.sqrt(errors.reduce((s, e) => s + e * e, 0) / errors.length) };
    receipt.results.push(result);
    assert.deepEqual(actual.errors, []);
    assert(errors.every(Number.isFinite), 'Nonfinite output');
    assert(result.maxError <= 2e-6, `Decode continuation error ${result.maxError}`);
  }
  receipt.passed = true;
  console.log(JSON.stringify({ host: receipt.host, cases: receipt.results.length,
    maxError: Math.max(...receipt.results.map(row => row.maxError)) }));
} catch (error) { receipt.passed = false; receipt.failure = error.message; throw error; }
finally { await writeFile(destination, JSON.stringify(receipt)); await browser?.close(); await new Promise(resolve => server.close(resolve)); }
