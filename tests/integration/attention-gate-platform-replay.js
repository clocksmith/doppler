// Captured attention gating against Float64. This does not qualify a model.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { gunzipSync } from 'node:zlib';
import { chromium } from 'playwright';

const [fixturePath, shaderPath, destination] = process.argv.slice(2);
assert(fixturePath && shaderPath && destination);
const fixtureBytes = gunzipSync(await readFile(fixturePath));
const fixture = JSON.parse(fixtureBytes);
const original = await readFile(shaderPath, 'utf8');
const hash = value => createHash('sha256').update(value).digest('hex');
assert.equal(hash(original), fixture.contract.shaderSha256);
const rows = fixture.data.map(row => row.captures.captures.flatMap(capture => capture.records));
const tensor = (records, boundary, role) => records.find(row => row.boundary === boundary)
  .tensors.find(item => item.role === role);
for (const [boundary, role] of [['inputs', 'gate'], ['outputs', 'core']]) {
  assert.equal(tensor(rows[0], boundary, role).data, tensor(rows[1], boundary, role).data,
    `Local error requires identical ${role}`);
}
const values = item => {
  assert.equal(item.dtype, 'f32');
  const bytes = Buffer.from(item.data, 'base64');
  assert.equal(bytes.length, item.bytes);
  return Array.from({ length: item.elements }, (_, index) => bytes.readFloatLE(index * 4));
};
const input = values(tensor(rows[0], 'outputs', 'core'));
const gate = values(tensor(rows[0], 'inputs', 'gate'));
const capturedCount = input.length;
for (let i = 0; i <= 8192; i++) {
  input.push(Math.fround(-4 + 8 * ((i * 31337) % 8193) / 8192));
  gate.push(Math.fround(-80 + 160 * i / 8192));
}
const candidate = process.env.DOPPLER_ATTENTION_GATE_DIAGNOSTIC ?? null;
assert(candidate === null || (candidate === 'refined-sigmoid'
  && process.env.DOPPLER_TEST_ONLY_ARITHMETIC === '1'));
let source = original;
if (candidate) {
  const before = `    let clamped = clamp(x, -15.0, 15.0);
    return 1.0 / (1.0 + exp(-clamped));`;
  assert.equal(source.split(before).length, 2);
  source = source.replace(before, `    let clamped = clamp(x, -15.0, 15.0);
    let z = exp_refined(-abs(clamped));
    let numerator = select(z, 1.0, clamped >= 0.0);
    let denominator = 1.0 + z;
    let inverse = reciprocal_refined(denominator);
    let quotient = numerator * inverse;
    return fma(fma(-quotient, denominator, numerator), inverse, quotient);`);
}
const backends = { darwin: ['--use-angle=metal'],
  linux: ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'] };
assert(Object.hasOwn(backends, process.platform));
const host = process.platform === 'darwin' ? 'mac' : 'linux';
const captured = values(tensor(rows[fixture.data.findIndex(row => row.host === host)], 'outputs', 'projectionInput'));
const server = createServer((_request, response) => response.end('<!doctype html>'));
await new Promise(done => server.listen(0, '127.0.0.1', done));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true,
    args: ['--enable-unsafe-webgpu', ...backends[process.platform]] });
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result = await page.evaluate(async ({ source, input, gate, constants }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw Error('Physical GPU required');
    const device = await adapter.requestDevice(), owned = [], errors = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    let staging;
    const buffer = (size, usage, data) => {
      const b = device.createBuffer({ size, usage }); owned.push(b);
      if (data) device.queue.writeBuffer(b, 0, data);
      return b;
    };
    try {
      const module = device.createShaderModule({ code: source });
      const diagnostics = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
      if (diagnostics.length) throw Error(diagnostics.map(row => row.message).join('; '));
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: {
        module, entryPoint: 'main', constants } });
      const bytes = input.length * 4;
      const uniform = buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        new Uint32Array([input.length, input.length, 0, 0]));
      const a = buffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, Float32Array.from(input));
      const output = buffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const g = buffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, Float32Array.from(gate));
      staging = buffer(bytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: [uniform, a, output, g].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(input.length / 256)); pass.end();
      encoder.copyBufferToBuffer(output, 0, staging, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const values = Array.from(new Float32Array(staging.getMappedRange().slice(0))); staging.unmap();
      await device.queue.onSubmittedWorkDone();
      return { values, errors, adapter: { vendor: adapter.info.vendor,
        architecture: adapter.info.architecture, description: adapter.info.description } };
    } finally {
      await device.queue.onSubmittedWorkDone().catch(() => {});
      if (staging?.mapState === 'mapped') staging.unmap();
      for (const b of owned) b.destroy();
      device.destroy();
    }
  }, { source, input, gate, constants: fixture.contract.constants });
  assert.deepEqual(result.errors, []);
  const compare = (actual, expected) => {
    assert.equal(actual.length, expected.length);
    let maxError = 0, squared = 0, different = 0;
    for (let i = 0; i < actual.length; i++) {
      const error = Math.abs(actual[i] - expected[i]); assert(Number.isFinite(error));
      maxError = Math.max(maxError, error); squared += error * error;
      if (actual[i] !== expected[i]) different++;
    }
    return { maxError, rmsError: Math.sqrt(squared / actual.length), different };
  };
  const reference = input.map((x, i) => x / (1 + Math.exp(-Math.max(-15, Math.min(15, gate[i])))));
  const receipt = { scope: 'Identical captured operands and gate sweep; not model acceptance',
    host, candidate, fixtureSha256: hash(fixtureBytes), originalShaderSha256: hash(original),
    shaderSha256: hash(source), browser: browser.version(), contract: fixture.contract,
    capturedCount, ...result, capturedAgreement: compare(result.values.slice(0, capturedCount), captured),
    independentCaptured: compare(result.values.slice(0, capturedCount), reference.slice(0, capturedCount)),
    independentSweep: compare(result.values.slice(capturedCount), reference.slice(capturedCount)) };
  await writeFile(destination, JSON.stringify(receipt));
  console.log(JSON.stringify({ host, candidate, capturedAgreement: receipt.capturedAgreement,
    independentCaptured: receipt.independentCaptured, independentSweep: receipt.independentSweep }));
  if (candidate === null) assert.equal(receipt.capturedAgreement.maxError, 0, 'Baseline must reproduce the actual gate');
  else assert(receipt.independentCaptured.maxError <= 3e-7 && receipt.independentSweep.maxError <= 5e-7,
    'Refined sigmoid exceeds the independent accuracy bound');
} finally {
  await browser?.close(); await new Promise(done => server.close(done));
}
