/** Physical head-256 F16-KV regression; captured prefill and block boundaries.
 * node tests/integration/attention-platform.js <shader.wgsl> <fixture.json.gz> <receipt.json>
 * Float64 references include F16 storage rounding; this is not generation acceptance.
 */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { gunzipSync } from 'node:zlib';
import { chromium } from 'playwright';
import { KERNEL_TOLERANCES, compareArrays } from '../kernels/harness/tolerance.js';

const [shaderPath, fixturePath, destination] = process.argv.slice(2);
assert(shaderPath && fixturePath && destination, 'Supply shader, compressed fixture, and receipt paths');
const source = await readFile(shaderPath, 'utf8');
const fixtureBytes = gunzipSync(await readFile(fixturePath));
const fixture = JSON.parse(fixtureBytes);
assert.equal(fixture.headDim, 256);
assert.equal(fixture.startPos, 0);
assert.equal(fixture.q.length, fixture.seqLen * fixture.numHeads * fixture.headDim);
assert.equal(fixture.k.length, fixture.seqLen * fixture.numKVHeads * fixture.headDim);
assert.equal(fixture.v.length, fixture.k.length);
assert.equal(fixture.float64Expected.length, fixture.q.length);
const hash = value => createHash('sha256').update(value).digest('hex');
const backends = {
  darwin: ['--use-angle=metal'],
  linux: ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'],
};
assert(Object.hasOwn(backends, process.platform), 'Physical Metal or Vulkan host required');
const receipt = {
  scope: 'Captured head-256 attention and block boundaries; not generation acceptance',
  platform: process.platform,
  seqLen: fixture.seqLen,
  shaderSha256: hash(source),
  fixtureSha256: hash(fixtureBytes),
  tolerance: KERNEL_TOLERANCES.attention,
};
const server = createServer((_request, response) => response.end('<!doctype html>'));
await new Promise(done => server.listen(0, '127.0.0.1', done));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true,
    args: ['--enable-unsafe-webgpu', ...backends[process.platform]] });
  receipt.browser = browser.version();
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result = await page.evaluate(async ({ source, fixture }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw Error('Physical GPU required');
    const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16'] });
    const owned = [], errors = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const buffer = (size, usage, data) => {
      const resource = device.createBuffer({ size, usage });
      owned.push(resource);
      if (data) device.queue.writeBuffer(resource, 0, data);
      return resource;
    };
    const input = data => buffer(data.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, data);
    try {
      const module = device.createShaderModule({ code: source });
      const diagnostics = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
      if (diagnostics.length) throw Error(diagnostics.map(row => row.message).join('; '));
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto',
        compute: { module, entryPoint: 'main' } });
      const bytes = new ArrayBuffer(64), view = new DataView(bytes);
      [fixture.numHeads, fixture.numKVHeads, fixture.headDim, fixture.seqLen, fixture.seqLen]
        .forEach((value, index) => view.setUint32(index * 4, value, true));
      view.setFloat32(20, fixture.scale, true);
      view.setUint32(24, fixture.isCausal ? 1 : 0, true);
      view.setUint32(48, 256, true);
      const output = buffer(fixture.q.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const resources = [buffer(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, bytes),
        input(Float32Array.from(fixture.q)), input(Float16Array.from(fixture.k)),
        input(Float16Array.from(fixture.v)), output, input(new Uint32Array([fixture.seqLen])),
        input(new Uint32Array(1))];
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: resources.map((resource, binding) => ({ binding, resource: { buffer: resource } })) });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(fixture.numHeads * Math.ceil(fixture.seqLen / 32));
      pass.end();
      const readback = buffer(output.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      encoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const values = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
      readback.unmap();
      return { values, errors, adapter: { vendor: adapter.info.vendor,
        architecture: adapter.info.architecture, description: adapter.info.description } };
    } finally {
      await device.queue.onSubmittedWorkDone().catch(() => {});
      for (const resource of owned) resource.destroy();
      device.destroy();
    }
  }, { source, fixture });
  Object.assign(receipt, result);
  receipt.valuesSha256 = hash(Buffer.from(Float32Array.from(result.values).buffer));
  receipt.comparison = compareArrays(fixture.float64Expected, result.values, receipt.tolerance);
  receipt.passed = result.errors.length === 0 && result.values.every(Number.isFinite)
    && receipt.comparison.passed;
  console.log(JSON.stringify({ adapter: receipt.adapter, seqLen: receipt.seqLen,
    passed: receipt.passed, comparison: receipt.comparison, valuesSha256: receipt.valuesSha256 }));
  assert(receipt.passed, 'Captured attention exceeds the existing attention accuracy bound');
} catch (error) {
  receipt.passed = false; receipt.failure = error.message;
  throw error;
} finally {
  await writeFile(destination, JSON.stringify(receipt));
  await browser?.close();
  await new Promise(done => server.close(done));
}
