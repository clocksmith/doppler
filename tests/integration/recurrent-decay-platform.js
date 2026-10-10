/** Isolated arithmetic qualification. This does not qualify model generation.
 * node tests/integration/recurrent-decay-platform.js <shader.wgsl> <receipt.json>
 * A baseline shader may be extracted with git show; never replace the model reference.
 */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';

const [shaderPath, destination] = process.argv.slice(2);
assert(shaderPath && destination, 'Supply a canonical shader and receipt destination');
const source = await readFile(shaderPath, 'utf8');
const start = source.indexOf('fn softplus('), end = source.indexOf('@compute');
assert(start >= 0 && end > start, 'Missing recurrent arithmetic helpers');
const helpers = source.slice(start, end);
const inputs = [];
const add = (x, aLog, logInput) => inputs.push([
  Math.fround(x), Math.fround(aLog), Math.fround(logInput),
]);
// Retained first divergence: token seven, value head three. The log argument
// is the identical GPU exponential result plus one, not a recomputed oracle.
add(6.29249382019043, -0.56640625, 541.4995727539062);
for (let i = 0; i <= 8192; i++) {
  const x = -80 + 160 * i / 8192;
  add(x, -8 + 12 * ((i * 31337) % 8193) / 8192, Math.exp(x));
}
// Thresholds and adjacent F32 values, including log cancellation near one.
for (const x of [-80, -20.000002, -20, -19.999998, -1e-6, 0, 1e-6,
  19.999998, 20, 20.000002, 80]) add(x, 0, 1 + Math.exp(x));
for (const x of [1 - 2 ** -24, 1, 1 + 2 ** -23, 2 ** -126, 2 ** 127]) add(0, 0, x);
const fields = ['expX', 'expA', 'logInput', 'logarithm', 'softplus', 'logDecay', 'decay', 'directLog'];
const shader = `${helpers}
@group(0) @binding(0) var<storage, read> operands: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> results: array<f32>;
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&operands)) { return; }
  let input = operands[id.x];
  let ex = exp_refined(input.x);
  let ea = exp_refined(input.y);
  let argument = 1.0 + ex;
  let sp = softplus(input.x);
  let g = -ea * sp;
  let offset = id.x * 8u;
  results[offset] = ex;
  results[offset + 1u] = ea;
  results[offset + 2u] = argument;
  results[offset + 3u] = log_refined(argument);
  results[offset + 4u] = sp;
  results[offset + 5u] = g;
  results[offset + 6u] = exp_refined(g);
  results[offset + 7u] = log_refined(input.z);
}`;
const backends = {
  darwin: ['--use-angle=metal'],
  linux: ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'],
};
assert(Object.hasOwn(backends, process.platform), 'Requires a physical Metal or Vulkan host');
const server = createServer((_req, res) => {
  res.setHeader('content-type', 'text/html'); res.end('<!doctype html>');
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true,
    args: ['--enable-unsafe-webgpu', ...backends[process.platform]] });
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const execution = await page.evaluate(async ({ shader, inputs }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw Error('Physical GPU required');
    const device = await adapter.requestDevice(), owned = [], errors = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const buffer = (size, usage) => {
      const value = device.createBuffer({ size, usage }); owned.push(value); return value;
    };
    try {
      const module = device.createShaderModule({ code: shader });
      const diagnostics = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
      if (diagnostics.length) throw Error(diagnostics.map(m => m.message).join('; '));
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const values = Float32Array.from(inputs.flatMap(row => [...row, 0]));
      const size = inputs.length * 8 * 4;
      const input = buffer(values.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const output = buffer(size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const readback = buffer(size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      device.queue.writeBuffer(input, 0, values);
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: [input, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(inputs.length / 128)); pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, size);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const result = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
      readback.unmap();
      return { values: result, errors, adapter: { vendor: adapter.info.vendor,
        architecture: adapter.info.architecture, device: adapter.info.device,
        description: adapter.info.description } };
    } finally {
      await device.queue.onSubmittedWorkDone().catch(() => {});
      for (const resource of owned) resource.destroy();
      device.destroy();
    }
  }, { shader, inputs });
  assert.deepEqual(execution.errors, []);
  assert(execution.values.every(Number.isFinite));
  const metrics = {};
  for (const [field, column, reference] of [
    ['logarithm', 3, (row, offset) => Math.log(execution.values[offset + 2])],
    ['directLog', 7, row => Math.log(row[2])],
    ['softplus', 4, row => Math.log1p(Math.exp(row[0]))],
    ['logDecay', 5, row => -Math.exp(row[1]) * Math.log1p(Math.exp(row[0]))],
  ]) {
    let max = 0, squared = 0;
    for (let i = 0; i < inputs.length; i++) {
      const error = Math.abs(execution.values[i * 8 + column] - reference(inputs[i], i * 8));
      max = Math.max(max, error); squared += error * error;
    }
    metrics[field] = { maxAbsoluteError: max, rmsError: Math.sqrt(squared / inputs.length) };
  }
  const receipt = { scope: 'Isolated recurrent scalar operations against Float64; not full-model acceptance',
    shaderSha256: createHash('sha256').update(source).digest('hex'),
    diagnosticSha256: createHash('sha256').update(shader).digest('hex'),
    platform: process.platform, browser: browser.version(), inputs, fields, ...execution, metrics };
  await writeFile(destination, JSON.stringify(receipt));
  console.log(JSON.stringify({ rows: inputs.length, metrics, adapter: execution.adapter }));
  assert.equal(execution.values[3], Math.fround(Math.log(execution.values[2])),
    'First divergent composed log operation must round to the independent reference');
} finally {
  await browser?.close();
  await new Promise(done => server.close(done));
}
