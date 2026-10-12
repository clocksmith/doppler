// Exact recorded main-entry normalization against an independent F64 fixture.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';

const [fixturePath, shaderPath, destination] = process.argv.slice(2);
assert(fixturePath && shaderPath && destination);
const bytes = await readFile(fixturePath), fixture = JSON.parse(bytes);
const source = await readFile(shaderPath, 'utf8');
const hash = value => createHash('sha256').update(value).digest('hex');
assert.equal(fixture.pipeline.entry, 'main');
assert.equal(fixture.pipeline.constants.PRE_RESIDUAL, 0);
assert.equal(fixture.pipeline.constants.OUTPUT_PRENORM, 0);
const receipt = { scope: 'Captured main RMSNorm; independent accuracy only', host: process.platform,
  fixtureSha256: hash(bytes), shaderSha256: hash(source), pipeline: fixture.pipeline,
  epsilon: fixture.epsilon, rows: fixture.rows, width: fixture.width };
const server = createServer((_request, response) => response.end('<!doctype html>'));
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true,
    args: ['--enable-unsafe-webgpu', ...(process.platform === 'darwin' ? ['--use-angle=metal']
      : ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'])] });
  receipt.browser = browser.version();
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  Object.assign(receipt, await page.evaluate(async ({ source, fixture }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw Error('Physical adapter required');
    const device = await adapter.requestDevice({ requiredFeatures: ['subgroups'] });
    const owned = [], errors = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const buffer = (size, usage, data) => {
      const value = device.createBuffer({ size, usage }); owned.push(value);
      if (data) device.queue.writeBuffer(value, 0, data);
      return value;
    };
    const decode = value => Uint8Array.from(atob(value), c => c.charCodeAt(0));
    let staging;
    try {
      const module = device.createShaderModule({ code: source });
      const messages = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
      if (messages.length) throw Error(messages.map(row => row.message).join('\n'));
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: {
        module, entryPoint: fixture.pipeline.entry, constants: fixture.pipeline.constants } });
      const uniform = new ArrayBuffer(32), view = new DataView(uniform);
      view.setUint32(0, fixture.width, true); view.setUint32(4, fixture.rows, true);
      view.setFloat32(8, fixture.epsilon, true); view.setUint32(16, fixture.rows, true);
      view.setFloat32(20, 1, true);
      const input = decode(fixture.input), weight = decode(fixture.weight);
      const output = buffer(input.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const resources = [buffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, uniform),
        buffer(input.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, input),
        buffer(weight.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, weight), output,
        buffer(input.length, GPUBufferUsage.STORAGE), buffer(input.length, GPUBufferUsage.STORAGE)];
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: resources.map((resource, binding) => ({ binding, resource: { buffer: resource } })) });
      staging = buffer(input.length, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(fixture.rows); pass.end();
      encoder.copyBufferToBuffer(output, 0, staging, 0, input.length);
      device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
      return { values: Array.from(new Float32Array(staging.getMappedRange().slice(0))), errors,
        adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture } };
    } finally {
      if (staging?.mapState === 'mapped') staging.unmap();
      await device.queue.onSubmittedWorkDone();
      for (const resource of owned) resource.destroy(); device.destroy();
    }
  }, { source, fixture }));
  const expected = Buffer.from(fixture.expected, 'base64'), observed = Buffer.from(fixture.observed, 'base64');
  const compare = values => {
    let maximum = 0, squares = 0;
    for (let index = 0; index < values.length; index++) {
      assert(Number.isFinite(values[index]));
      const difference = Math.abs(values[index] - expected.readDoubleLE(index * 8));
      maximum = Math.max(maximum, difference); squares += difference ** 2;
    }
    return { maximum, rms: Math.sqrt(squares / values.length) };
  };
  receipt.comparison = compare(receipt.values);
  receipt.baseline = compare(Array.from({ length: fixture.rows * fixture.width }, (_, i) => observed.readFloatLE(i * 4)));
  receipt.valuesSha256 = hash(Buffer.from(Float32Array.from(receipt.values).buffer));
  receipt.matchesCaptured = receipt.valuesSha256 === hash(observed);
  assert.deepEqual(receipt.errors, []);
  receipt.executed = true;
  console.log(JSON.stringify({ ...receipt, values: undefined }));
} catch (error) { receipt.failure = error.message; throw error; }
finally {
  await writeFile(destination, JSON.stringify(receipt)); await browser?.close();
  server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
}
