/** Physical combined Q/K rotation on captured operands; not generation acceptance.
 * node tests/integration/rope-platform.js <precompute.wgsl> <rope_qk.wgsl> <receipt.json>
 */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';
import { KERNEL_TOLERANCES, compareArrays } from '../kernels/harness/tolerance.js';

const [precomputePath, rotationPath, destination] = process.argv.slice(2);
assert(precomputePath && rotationPath && destination, 'Supply both canonical shaders and a receipt path');
const precompute = await readFile(precomputePath, 'utf8');
const rotation = await readFile(rotationPath, 'utf8');
const fixtureBytes = await readFile(new URL('../../artifacts/recovery-20261009/rope-captured-inputs.json', import.meta.url));
const fixture = JSON.parse(fixtureBytes);
const hash = value => createHash('sha256').update(value).digest('hex');
const maxSeqLen = 4096, halfDim = fixture.rotaryDim / 2;
const cos = [], sin = [];
for (let position = 0; position < maxSeqLen; position++) {
  for (let dimension = 0; dimension < halfDim; dimension++) {
    const frequency = Math.fround(1 / fixture.theta ** (dimension * 2 / fixture.frequencyBaseDim));
    const angle = Math.fround(position * frequency);
    cos.push(Math.fround(Math.cos(angle))); sin.push(Math.fround(Math.sin(angle)));
  }
}
const backends = {
  darwin: ['--use-angle=metal'],
  linux: ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'],
};
assert(Object.hasOwn(backends, process.platform), 'Physical Metal or Vulkan host required');
const receipt = { scope: 'Captured combined Q/K rotation and fixed-table control; not generation acceptance',
  platform: process.platform, fixtureSha256: hash(fixtureBytes), precomputeSha256: hash(precompute),
  rotationSha256: hash(rotation), tolerance: KERNEL_TOLERANCES.rope, cases: [] };
const server = createServer((_request, response) => response.end('<!doctype html>'));
await new Promise(done => server.listen(0, '127.0.0.1', done));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true,
    args: ['--enable-unsafe-webgpu', ...backends[process.platform]] });
  receipt.browser = browser.version();
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result = await page.evaluate(async ({ precompute, rotation, fixture, cos, sin, maxSeqLen }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw Error('Physical GPU required');
    const device = await adapter.requestDevice(), owned = [], errors = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const buffer = (size, usage, data) => {
      const value = device.createBuffer({ size, usage }); owned.push(value);
      if (data) device.queue.writeBuffer(value, 0, data);
      return value;
    };
    const storage = data => buffer(data.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, data);
    const uniform = data => buffer(data.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, data);
    const pipeline = async source => {
      const module = device.createShaderModule({ code: source });
      const diagnostics = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
      if (diagnostics.length) throw Error(diagnostics.map(row => row.message).join('; '));
      return device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    };
    const execute = (pipeline, buffers, workgroups) => {
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(workgroups); pass.end(); device.queue.submit([encoder.finish()]);
    };
    const read = async source => {
      const target = buffer(source.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(source, 0, target, 0, source.size); device.queue.submit([encoder.finish()]);
      await target.mapAsync(GPUMapMode.READ);
      const values = Array.from(new Float32Array(target.getMappedRange().slice(0))); target.unmap();
      return values;
    };
    try {
      const halfDim = fixture.rotaryDim / 2, count = maxSeqLen * halfDim;
      const frequencyPipeline = await pipeline(precompute), rotationPipeline = await pipeline(rotation);
      const bytes = new ArrayBuffer(64), view = new DataView(bytes);
      [maxSeqLen, fixture.rotaryDim, fixture.frequencyBaseDim, 0]
        .forEach((value, index) => view.setUint32(index * 4, value, true));
      [fixture.theta, 1, 1, 1, 1, 0].forEach((value, index) => view.setFloat32(16 + index * 4, value, true));
      view.setUint32(40, count, true);
      const gpuCos = storage(new Float32Array(count)), gpuSin = storage(new Float32Array(count));
      execute(frequencyPipeline, [uniform(bytes), storage(new Uint32Array(1)), gpuCos, gpuSin], Math.ceil(count / 256));
      const tables = { cos: await read(gpuCos), sin: await read(gpuSin) }, cases = [];
      for (const [name, cosBuffer, sinBuffer] of [
        ['generated-table', gpuCos, gpuSin],
        ['fixed-table-control', storage(Float32Array.from(cos)), storage(Float32Array.from(sin))],
      ]) {
        const q = storage(Float32Array.from(fixture.sides.q.input));
        const k = storage(Float32Array.from(fixture.sides.k.input));
        const uniforms = Uint32Array.from([fixture.seqLen, fixture.sides.q.input.length / fixture.headDim,
          fixture.sides.k.input.length / fixture.headDim, fixture.headDim, fixture.startPos,
          fixture.rotaryDim, fixture.interleaved ? 1 : 0, fixture.pairSpanDim]);
        execute(rotationPipeline, [uniform(uniforms), q, k, cosBuffer, sinBuffer],
          Math.ceil((fixture.sides.q.input.length + fixture.sides.k.input.length) / fixture.headDim * halfDim / 256));
        cases.push({ name, q: await read(q), k: await read(k) });
      }
      return { tables, cases, errors, adapter: { vendor: adapter.info.vendor,
        architecture: adapter.info.architecture, description: adapter.info.description } };
    } finally {
      await device.queue.onSubmittedWorkDone().catch(() => {});
      for (const resource of owned) resource.destroy(); device.destroy();
    }
  }, { precompute, rotation, fixture, cos, sin, maxSeqLen });
  receipt.adapter = result.adapter; receipt.errors = result.errors; receipt.tables = result.tables;
  for (const row of result.cases) {
    const comparisons = {};
    for (const side of ['q', 'k']) {
      const input = fixture.sides[side].input, expected = [...input];
      for (let head = 0; head < input.length / fixture.headDim; head++) {
        for (let dimension = 0; dimension < halfDim; dimension++) {
          const angle = fixture.startPos / fixture.theta ** (dimension * 2 / fixture.frequencyBaseDim);
          const index = fixture.startPos * halfDim + dimension;
          const c = row.name === 'fixed-table-control' ? cos[index] : Math.cos(angle);
          const s = row.name === 'fixed-table-control' ? sin[index] : Math.sin(angle);
          const first = head * fixture.headDim + dimension, second = first + fixture.pairSpanDim / 2;
          expected[first] = input[first] * c - input[second] * s;
          expected[second] = input[first] * s + input[second] * c;
        }
      }
      comparisons[side] = { finite: row[side].every(Number.isFinite),
        valuesSha256: hash(Buffer.from(Float32Array.from(row[side]).buffer)),
        ...compareArrays(expected, row[side], receipt.tolerance) };
    }
    receipt.cases.push({ ...row, comparisons });
  }
  receipt.passed = receipt.errors.length === 0 && receipt.cases.every(row =>
    Object.values(row.comparisons).every(side => side.finite && side.passed));
  console.log(JSON.stringify({ adapter: receipt.adapter, cases: receipt.cases.map(({ q, k, ...row }) => row) }));
  assert(receipt.passed, 'Captured RoPE exceeds the existing rotation accuracy bound');
} catch (error) {
  receipt.passed = false; receipt.failure = error.message; throw error;
} finally {
  await writeFile(destination, JSON.stringify(receipt));
  await browser?.close(); await new Promise(done => server.close(done));
}
