/** Physical Q/K normalization regression; this does not qualify generation.
 * node tests/integration/qk-normalization-platform.js <shader.wgsl> <receipt.json>
 * Compare receipt values across hosts only when fixture and shader hashes match.
 */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';
import { KERNEL_TOLERANCES, compareArrays } from '../kernels/harness/tolerance.js';

const [shaderPath, destination] = process.argv.slice(2);
assert(shaderPath && destination, 'Supply a canonical shader and receipt destination');
const source = await readFile(shaderPath, 'utf8');
const fixtureBytes = await readFile(new URL('../fixtures/qk-normalization-regression.json', import.meta.url));
const fixture = JSON.parse(fixtureBytes);
const hash = value => createHash('sha256').update(value).digest('hex');
const half = bits => {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 31;
  const fraction = bits & 1023;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (exponent ? (1 + fraction / 1024) * 2 ** (exponent - 15) : fraction * 2 ** -24);
};
const weights = name => {
  const bytes = Buffer.from(fixture[name], 'base64');
  return Array.from({ length: bytes.length / 2 }, (_, i) => bytes.readUInt16LE(i * 2));
};
const qWeights = weights('qWeight'), kWeights = weights('kWeight');
const cases = [{ name: 'captured-layer-three', headDim: fixture.headDim,
  q: fixture.q, k: fixture.k, qWeights, kWeights }];
for (const headDim of [64, 128, 256, 512, 1024]) {
  // F32-rounded operands and exactly representable F16 weights are identical
  // on every host. Three rows also exercise a partial dispatch row.
  const operands = offset => Array.from({ length: headDim * 3 }, (_, i) =>
    Math.fround(((i * 31337 + offset) % 8193 - 4096) / 257));
  const repeat = values => Array.from({ length: headDim }, (_, i) => values[i % values.length]);
  cases.push({ name: `head-${headDim}`, headDim, q: operands(17), k: operands(97),
    qWeights: repeat(qWeights), kWeights: repeat(kWeights) });
}
const backends = {
  darwin: ['--use-angle=metal'],
  linux: ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'],
};
assert(Object.hasOwn(backends, process.platform), 'Requires a physical Metal or Vulkan host');
const receipt = { scope: 'Full Q/K shader against Float64; not full-model acceptance',
  shaderSha256: hash(source), fixtureSha256: hash(fixtureBytes),
  inputsSha256: hash(JSON.stringify(cases)), constants: fixture.constants,
  epsilon: Math.fround(fixture.epsilon), tolerance: KERNEL_TOLERANCES.rmsnorm,
  platform: process.platform, cases: [] };
const server = createServer((_req, res) => {
  res.setHeader('content-type', 'text/html'); res.end('<!doctype html>');
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true,
    args: ['--enable-unsafe-webgpu', ...backends[process.platform]] });
  receipt.browser = browser.version();
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const execution = await page.evaluate(async ({ source, cases, constants, epsilon }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw Error('Physical GPU required');
    const device = await adapter.requestDevice(), errors = [], results = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    try {
      const module = device.createShaderModule({ code: source });
      const diagnostics = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
      if (diagnostics.length) throw Error(diagnostics.map(m => m.message).join('; '));
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto',
        compute: { module, entryPoint: 'main', constants } });
      for (const row of cases) {
        const owned = [];
        const buffer = (size, usage, data) => {
          const value = device.createBuffer({ size, usage }); owned.push(value);
          if (data) device.queue.writeBuffer(value, 0, data);
          return value;
        };
        try {
          const uniform = new ArrayBuffer(32), view = new DataView(uniform);
          const qRows = row.q.length / row.headDim, kRows = row.k.length / row.headDim;
          [qRows, kRows, row.headDim, 2].forEach((v, i) => view.setUint32(i * 4, v, true));
          view.setFloat32(16, epsilon, true);
          const input = values => buffer(values.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, values);
          const output = size => buffer(size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const qOutput = output(row.q.length * 4), kOutput = output(row.k.length * 4);
          const bindings = [buffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, uniform),
            input(Float32Array.from(row.q)), input(Uint16Array.from(row.qWeights)), qOutput,
            input(Float32Array.from(row.k)), input(Uint16Array.from(row.kWeights)), kOutput];
          const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
            entries: bindings.map((buffer, binding) => ({ binding, resource: { buffer } })) });
          const size = (row.q.length + row.k.length) * 4;
          const readback = buffer(size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
          pass.setPipeline(pipeline); pass.setBindGroup(0, group);
          pass.dispatchWorkgroups(2, Math.ceil((qRows + kRows) / 2)); pass.end();
          encoder.copyBufferToBuffer(qOutput, 0, readback, 0, row.q.length * 4);
          encoder.copyBufferToBuffer(kOutput, 0, readback, row.q.length * 4, row.k.length * 4);
          device.queue.submit([encoder.finish()]);
          await readback.mapAsync(GPUMapMode.READ);
          results.push(Array.from(new Float32Array(readback.getMappedRange().slice(0))));
          readback.unmap();
        } finally {
          await device.queue.onSubmittedWorkDone().catch(() => {});
          for (const resource of owned) resource.destroy();
        }
      }
      return { results, errors, adapter: { vendor: adapter.info.vendor,
        architecture: adapter.info.architecture, device: adapter.info.device,
        description: adapter.info.description } };
    } finally { device.destroy(); }
  }, { source, cases, constants: fixture.constants, epsilon: receipt.epsilon });
  receipt.adapter = execution.adapter; receipt.errors = execution.errors;
  for (const [index, row] of cases.entries()) {
    const expected = [];
    for (const side of ['q', 'k']) {
      const operands = row[side], weight = row[`${side}Weights`].map(half);
      for (let start = 0; start < operands.length; start += row.headDim) {
        let sum = 0;
        for (let i = 0; i < row.headDim; i++) sum += operands[start + i] ** 2;
        const inverse = 1 / Math.sqrt(sum / row.headDim + receipt.epsilon);
        for (let i = 0; i < row.headDim; i++) {
          expected.push(operands[start + i] * inverse * (1 + weight[i]));
        }
      }
    }
    const values = execution.results[index];
    receipt.cases.push({ name: row.name, headDim: row.headDim, values,
      valuesSha256: hash(Buffer.from(Float32Array.from(values).buffer)),
      finite: values.every(Number.isFinite), ...compareArrays(expected, values, receipt.tolerance) });
  }
  receipt.passed = execution.errors.length === 0 && receipt.cases.every(row => row.finite && row.passed);
  console.log(JSON.stringify({ adapter: receipt.adapter, cases: receipt.cases.map(({ values, ...row }) => row) }));
  assert(receipt.passed, 'Q/K normalization exceeds the existing RMSNorm accuracy bound');
} catch (error) {
  receipt.passed = false; receipt.failure = error.message;
  throw error;
} finally {
  await writeFile(destination, JSON.stringify(receipt));
  await browser?.close();
  await new Promise(done => server.close(done));
}
