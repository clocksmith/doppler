// Isolated physical replay of captured operands; not model qualification.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';

const [fixturePath, shaderPath, destination] = process.argv.slice(2);
const observeAccumulation = process.env.DOPPLER_Q4_OBSERVE_ACCUMULATION === '1';
assert(fixturePath && shaderPath && destination, 'Supply captured fixture, declared shader and receipt');
const fixtureBytes = await readFile(fixturePath);
const fixture = JSON.parse(fixtureBytes), source = await readFile(shaderPath, 'utf8');
const hash = value => createHash('sha256').update(value).digest('hex');
assert.equal(hash(Buffer.from(fixture.input, 'base64')), fixture.inputSha256);
assert.equal(hash(Buffer.from(fixture.weights, 'base64')), fixture.weightsSha256);
const expected = Buffer.from(fixture.oracleFloat64, 'base64');
assert.equal(expected.length, fixture.M * fixture.N * 8);
const receipt = { scope: 'Identical captured F32/Q4_K operands against an independent Float64 projection',
  host: process.platform, fixtureSha256: hash(fixtureBytes), shaderSha256: hash(source),
  geometry: { M: fixture.M, N: fixture.N, K: fixture.K },
  maximumAllowedError: 2e-6, requiredPeakErrorReduction: 2 };
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
  const result = await page.evaluate(async ({ source, fixture, observeAccumulation }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter?.features.has('shader-f16')) throw new Error('Physical shader-f16 adapter required');
    const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16'] });
    const errors = [], owned = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const buffer = (size, usage, data) => {
      const value = device.createBuffer({ size, usage }); owned.push(value);
      if (data) device.queue.writeBuffer(value, 0, data);
      return value;
    };
    const decode = text => Uint8Array.from(atob(text), char => char.charCodeAt(0));
    let staging, observedStaging;
    try {
      const module = device.createShaderModule({ code: source });
      const messages = (await module.getCompilationInfo()).messages.filter(item => item.type === 'error');
      if (messages.length) throw new Error(messages.map(item => item.message).join('\n'));
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: {
        module, entryPoint: 'main', constants: { TILE_M: 4, TILE_N: 256 } } });
      const uniform = new ArrayBuffer(32), view = new DataView(uniform);
      [fixture.M, fixture.N, fixture.K].forEach((value, index) => view.setUint32(index * 4, value, true));
      view.setFloat32(12, 1, true); view.setUint32(16, fixture.K / 256, true);
      const input = decode(fixture.input), weights = decode(fixture.weights);
      const output = buffer(fixture.M * fixture.N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const resources = [buffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, uniform),
        buffer(input.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, input),
        buffer(weights.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, weights), output];
      const observed = observeAccumulation ? buffer(output.size * 10,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) : null;
      if (observed) resources.push(observed);
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: resources.map((resource, binding) => ({ binding, resource: { buffer: resource } })) });
      staging = buffer(output.size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(fixture.N / 256), Math.ceil(fixture.M / 4)); pass.end();
      encoder.copyBufferToBuffer(output, 0, staging, 0, output.size);
      if (observed) {
        observedStaging = buffer(observed.size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        encoder.copyBufferToBuffer(observed, 0, observedStaging, 0, observed.size);
      }
      device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
      let accumulation = null;
      if (observedStaging) {
        await observedStaging.mapAsync(GPUMapMode.READ);
        accumulation = Array.from(new Float32Array(observedStaging.getMappedRange().slice(0)));
      }
      return { values: Array.from(new Float32Array(staging.getMappedRange().slice(0))), accumulation, errors,
        adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture } };
    } finally {
      if (staging?.mapState === 'mapped') staging.unmap();
      if (observedStaging?.mapState === 'mapped') observedStaging.unmap();
      await device.queue.onSubmittedWorkDone();
      for (const resource of owned) resource.destroy(); device.destroy();
    }
  }, { source, fixture, observeAccumulation });
  Object.assign(receipt, result);
  const compare = values => {
    let maximum = 0, squares = 0, worstIndex = null;
    for (let index = 0; index < values.length; index++) {
      assert(Number.isFinite(values[index]));
      const error = Math.abs(values[index] - expected.readDoubleLE(index * 8));
      if (error > maximum) { maximum = error; worstIndex = index; }
      squares += error * error;
    }
    return { maxAbsoluteError: maximum, rmsError: Math.sqrt(squares / values.length), worstIndex };
  };
  receipt.comparison = compare(result.values);
  const baseline = Buffer.from(fixture.baseline026, 'base64');
  receipt.baseline = compare(Array.from({ length: result.values.length }, (_, index) => baseline.readFloatLE(index * 4)));
  receipt.valuesSha256 = hash(Buffer.from(Float32Array.from(result.values).buffer));
  assert.deepEqual(result.errors, []);
  assert(receipt.comparison.maxAbsoluteError <= receipt.maximumAllowedError);
  assert(receipt.comparison.maxAbsoluteError * receipt.requiredPeakErrorReduction < receipt.baseline.maxAbsoluteError);
  receipt.passed = true;
  console.log(JSON.stringify({ ...receipt, values: undefined, accumulation: undefined }));
} catch (error) { receipt.passed = false; receipt.failure = error.message; throw error; }
finally {
  await writeFile(destination, JSON.stringify(receipt)); await browser?.close();
  server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
}
