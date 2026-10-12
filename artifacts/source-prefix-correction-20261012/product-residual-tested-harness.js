// Independent exact-double product residual check of a test-only F32 helper.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';

const [shaderPath, destination] = process.argv.slice(2);
assert(shaderPath && destination);
const source = await readFile(shaderPath, 'utf8');
const helper = source.slice(source.indexOf('fn add_u64('), source.indexOf('var<workgroup> shared_error:'));
assert(helper.includes('fn product_residual('));
const words = new Uint32Array(1), floats = new Float32Array(words.buffer);
const fromBits = bits => { words[0] = bits; return floats[0]; };
const toBits = value => { floats[0] = value; return words[0]; };
let seed = 0x624f901b;
const random = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return seed >>> 0; };
const input = [], expected = [], labels = [];
const add = (a, b, label) => {
  const product = Math.fround(a * b);
  if (![a, b, product].every(Number.isFinite)) return;
  const exact = a * b;
  const neighboring = (value, up) => {
    if (value === 0) return fromBits(up ? 1 : 0x80000001);
    return fromBits(toBits(value) + ((value > 0) === up ? 1 : -1));
  };
  const variants = [[product, label + ':nearest']];
  if (product !== exact) {
    variants.push([product > exact ? neighboring(product, false) : product, label + ':floor']);
    variants.push([product < exact ? neighboring(product, true) : product, label + ':ceil']);
  }
  for (const [rounded, name] of variants) {
    if (!Number.isFinite(rounded)) continue;
    input.push(a, b, rounded, 0);
    // A product of two binary32 operands is exact in binary64 (48 significand bits).
    expected.push(toBits(Math.fround(exact - rounded)));
    labels.push(name);
  }
};
const edges = [0, 1, 0x007fffff, 0x00800000, 0x00800001, 0x3effffff,
  0x3f000000, 0x3f000001, 0x3f7fffff, 0x3f800000, 0x3f800001,
  0x3fffffff, 0x40000000, 0x40000001, 0x7f7fffff];
for (const a of edges) for (const b of edges) for (const sign of [0, 0x80000000]) {
  add(fromBits(a | sign), fromBits(b), 'boundary');
}
for (let index = 0; index < 100000; index++) add(fromBits(random()), fromBits(random()), 'random');
const bytes = Buffer.from(Float32Array.from(input).buffer);
const hash = value => createHash('sha256').update(value).digest('hex');
const receipt = { scope: 'Exact finite F32 product residual; subnormal observations retained',
  shaderSha256: hash(source), harnessSha256: hash(await readFile(import.meta.filename)), inputSha256: hash(bytes), count: expected.length, seed: '624f901b' };
const server = createServer((_request, response) => response.end('<!doctype html>'));
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true,
    args: ['--enable-unsafe-webgpu', ...(process.platform === 'darwin' ? ['--use-angle=metal']
      : ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'])] });
  receipt.browser = browser.version();
  const page = await browser.newPage(); await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result = await page.evaluate(async ({ helper, input, count }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw Error('Physical adapter required');
    const device = await adapter.requestDevice(), owned = [], errors = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const buffer = (size, usage) => { const value = device.createBuffer({ size, usage }); owned.push(value); return value; };
    let staging;
    try {
      const module = device.createShaderModule({ code: helper + `
@group(0) @binding(0) var<storage, read> inputs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputs: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&inputs)) { return; }
  let v = inputs[id.x]; let p = v.x * v.y;
  outputs[id.x * 3u] = bitcast<u32>(product_residual(v.x, v.y, v.z));
  outputs[id.x * 3u + 1u] = bitcast<u32>(p);
  outputs[id.x * 3u + 2u] = bitcast<u32>(product_residual(v.x, v.y, p));
}` });
      const messages = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
      if (messages.length) throw Error(messages.map(row => row.message).join('\n'));
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const inputs = buffer(input.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(inputs, 0, new Float32Array(input));
      const outputs = buffer(count * 12, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      staging = buffer(count * 12, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
        [inputs, outputs].map((resource, binding) => ({ binding, resource: { buffer: resource } })) });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
      encoder.copyBufferToBuffer(outputs, 0, staging, 0, count * 12);
      device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
      return { values: Array.from(new Uint32Array(staging.getMappedRange().slice(0))), errors,
        adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture } };
    } finally {
      if (staging?.mapState === 'mapped') staging.unmap();
      await device.queue.onSubmittedWorkDone(); for (const value of owned) value.destroy(); device.destroy();
    }
  }, { helper, input, count: expected.length });
  receipt.adapter = result.adapter; receipt.errors = result.errors;
  receipt.mismatches = []; receipt.actualProductMismatches = [];
  for (let index = 0; index < expected.length; index++) {
    const actual = result.values[index * 3];
    const product = fromBits(result.values[index * 3 + 1]);
    const residual = result.values[index * 3 + 2];
    const liveExpected = toBits(Math.fround(input[index * 4] * input[index * 4 + 1] - product));
    const equalBits = (a, b) => a === b || ((a & 0x7fffffff) === 0 && (b & 0x7fffffff) === 0);
    if (!equalBits(actual, expected[index])) receipt.mismatches.push({ index, label: labels[index],
      operands: input.slice(index * 4, index * 4 + 3).map(toBits), actual, expected: expected[index] });
    if (!equalBits(residual, liveExpected)) receipt.actualProductMismatches.push({ index, label: labels[index],
      operands: input.slice(index * 4, index * 4 + 2).map(toBits), product: toBits(product), actual: residual, expected: liveExpected });
  }
  receipt.passed = receipt.mismatches.length === 0 && receipt.actualProductMismatches.length === 0 && result.errors.length === 0;
  console.log(JSON.stringify({ ...receipt, mismatches: receipt.mismatches.length, actualProductMismatches: receipt.actualProductMismatches.length }));
} finally {
  await writeFile(destination, JSON.stringify(receipt, null, 2)); await browser?.close();
  server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
}
assert(receipt.passed, 'F32 product residual failed independent finite-domain checks');
