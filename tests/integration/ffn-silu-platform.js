/** Isolated fused-FFN SiLU qualification. This does not qualify model generation.
 * node tests/integration/ffn-silu-platform.js <shader.wgsl> <receipt.json>
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
function functionSource(name) {
  const start = source.indexOf(`fn ${name}(`);
  if (start < 0) return '';
  let depth = 0, opened = false;
  for (let i = start; i < source.length; i++) {
    if (source[i] === '{') { depth++; opened = true; }
    if (source[i] === '}' && --depth === 0 && opened) return source.slice(start, i + 1);
  }
  throw Error(`Unclosed shader function ${name}`);
}
const helpers = ['sigmoid', 'silu', 'exp_refined', 'reciprocal_refined']
  .map(functionSource).filter(Boolean).join('\n');
assert(functionSource('silu'), 'Missing SiLU');
const clampsInput = functionSource('silu').includes('clamp(x, -15.0, 15.0)') ||
  functionSource('silu').includes('sigmoid(x)') && functionSource('sigmoid').includes('clamp(x, -15.0, 15.0)');
const fixture = JSON.parse(await readFile(new URL('../fixtures/ffn-silu-regression.json', import.meta.url)));
const inputs = fixture.inputs.map(row => [row.gate, row.up, 0]);
for (let i = 0; i <= 8192; i++) inputs.push([
  Math.fround(-80 + 160 * i / 8192), Math.fround(-4 + 8 * ((i * 31337) % 8193) / 8192), 0,
]);
const fields = ['gated'];
const shader = `${helpers}
@group(0) @binding(0) var<storage, read> operands: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> results: array<f32>;
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&operands)) { return; }
  let input = operands[id.x];
  results[id.x] = silu(input.x) * input.y;
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
      const size = inputs.length * 4;
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
  let maxAbsoluteError = 0, squared = 0;
  const errors = inputs.map(([x, up], i) => {
    const exponent = clampsInput ? Math.max(-15, Math.min(15, x)) : x;
    const reference = x / (1 + Math.exp(-exponent)) * up;
    const error = Math.abs(execution.values[i] - reference);
    maxAbsoluteError = Math.max(maxAbsoluteError, error); squared += error * error;
    return { error, reference };
  });
  const metrics = { maxAbsoluteError, rmsError: Math.sqrt(squared / inputs.length) };
  const receipt = { scope: 'Isolated fused-FFN SiLU against Float64; not full-model acceptance',
    shaderSha256: createHash('sha256').update(source).digest('hex'),
    diagnosticSha256: createHash('sha256').update(shader).digest('hex'),
    platform: process.platform, browser: browser.version(), inputs, fields, ...execution, metrics };
  await writeFile(destination, JSON.stringify(receipt));
  console.log(JSON.stringify({ rows: inputs.length, metrics, adapter: execution.adapter }));
  assert(errors.slice(0, fixture.inputs.length).every(row => row.error <= 1e-7),
    'Captured SwiGLU outputs exceed the independent accuracy regression bound');
  assert(errors.every(row => row.error <= 1e-7 + Math.abs(row.reference) * 3e-7),
    'SiLU range sweep exceeds the independent accuracy bound');
} finally {
  await browser?.close();
  await new Promise(done => server.close(done));
}
