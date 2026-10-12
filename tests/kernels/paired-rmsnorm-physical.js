import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';
const destination = process.argv[2];
if (!destination) throw new Error('Provide receipt destination.');
await mkdir(dirname(destination), { recursive: true });
const source = await readFile(new URL('../../src/gpu/kernels/rmsnorm_paired.wgsl', import.meta.url), 'utf8');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const cases = [];
let seed = 624901;
function random() { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 2 ** 32; }
for (const hidden of [128, 256, 1024]) for (const pre of [false, true]) {
  const input = [], residual = [], weight = [];
  for (let i = 0; i < hidden * 2; i++) {
    const hi = Math.fround((random() - .5) * 4), low = Math.fround((random() - .5) * 2 ** -25);
    input.push(hi, low); residual.push(Math.fround((random() - .5) * .2), Math.fround((random() - .5) * 2 ** -28));
  }
  for (let i = 0; i < hidden; i++) weight.push(Math.fround(.5 + random()));
  cases.push({ hidden, tokens: 2, input, residual, weight, eps: pre ? 1e-6 : 1e-5, pre, post: !pre, offset: pre });
}
const receipt = { schema: 'doppler-paired-rmsnorm-physical-v1', shaderSha256: hash(source),
  fixtureSha256: hash(JSON.stringify(cases)), seed: 624901, tolerance: 2e-12,
  scope: 'Independent Number/F64 RMS equation, source epsilon, weight offset and pre/post residual. No whole-model acceptance.', cases: [] };
const server = createServer((_req, res) => res.end('<!doctype html>'));
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu',
    ...(process.platform === 'darwin' ? ['--use-angle=metal'] : ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'])] });
  const page = await browser.newPage(); await page.goto(`http://127.0.0.1:${server.address().port}`);
  const physical = await page.evaluate(async ({ source, cases }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw new Error('Physical adapter required.');
    const device = await adapter.requestDevice(), errors = [], results = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    try {
      const module = device.createShaderModule({ code: source });
      const bad = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
      if (bad.length) throw new Error(bad.map(row => row.message).join('\n'));
      for (const fixture of cases) {
        const owned = [], mapped = [];
        const alloc = (size, usage) => { const buffer = device.createBuffer({ size, usage }); owned.push(buffer); return buffer; };
        try {
          const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main', constants: {
            WORKGROUP_SIZE: 64, WEIGHT_F16: false, WEIGHT_OFFSET: fixture.offset,
            PRE_RESIDUAL: fixture.pre, POST_RESIDUAL: fixture.post, OUTPUT_PRENORM: fixture.pre,
          } } });
          const uniform = alloc(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
          const uniformBytes = new ArrayBuffer(32), view = new DataView(uniformBytes);
          view.setUint32(0, fixture.hidden, true); view.setUint32(4, fixture.tokens, true);
          view.setFloat32(8, Math.fround(fixture.eps), true); view.setFloat32(12, fixture.eps - Math.fround(fixture.eps), true);
          view.setUint32(16, fixture.tokens, true); view.setFloat32(20, 1, true);
          device.queue.writeBuffer(uniform, 0, uniformBytes);
          const input = alloc(fixture.input.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
          const residual = alloc(fixture.residual.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
          const weight = alloc(fixture.weight.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
          device.queue.writeBuffer(input, 0, new Float32Array(fixture.input));
          device.queue.writeBuffer(residual, 0, new Float32Array(fixture.residual));
          device.queue.writeBuffer(weight, 0, new Float32Array(fixture.weight));
          const bytes = fixture.input.length * 4;
          const output = alloc(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const prenorm = alloc(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const staging = alloc(bytes * 2, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
          const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [uniform, input, weight, output, residual, prenorm].map((buffer, binding) => ({ binding, resource: { buffer } })) });
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
          pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(fixture.tokens); pass.end();
          encoder.copyBufferToBuffer(output, 0, staging, 0, bytes); encoder.copyBufferToBuffer(prenorm, 0, staging, bytes, bytes);
          device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ); mapped.push(staging);
          results.push(Array.from(new Float32Array(staging.getMappedRange().slice(0))));
        } finally { for (const b of mapped) b.unmap(); await device.queue.onSubmittedWorkDone(); for (const b of owned) b.destroy(); }
      }
      return { results, errors, adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture } };
    } finally { await device.queue.onSubmittedWorkDone(); device.destroy(); }
  }, { source, cases });
  receipt.adapter = physical.adapter; receipt.browser = browser.version(); receipt.errors = physical.errors;
  for (let c = 0; c < cases.length; c++) {
    const f = cases[c], values = physical.results[c]; let maxError = 0, failures = 0;
    for (let token = 0; token < f.tokens; token++) {
      const inputs = [], residuals = [];
      for (let i = 0; i < f.hidden; i++) {
        const index = (token * f.hidden + i) * 2;
        const residual = f.residual[index] + f.residual[index + 1]; residuals.push(residual);
        inputs.push(f.input[index] + f.input[index + 1] + (f.pre ? residual : 0));
      }
      const inverse = 1 / Math.sqrt(inputs.reduce((sum, value) => sum + value * value, 0) / f.hidden + f.eps);
      for (let i = 0; i < f.hidden; i++) {
        const index = (token * f.hidden + i) * 2;
        const expected = inputs[i] * inverse * (f.weight[i] + (f.offset ? 1 : 0)) + (f.post ? residuals[i] : 0);
        const error = Math.abs(values[index] + values[index + 1] - expected) / Math.max(1, Math.abs(expected));
        maxError = Math.max(maxError, error); if (!Number.isFinite(error) || error > receipt.tolerance) failures++;
        if (f.pre && values[index + f.input.length] + values[index + 1 + f.input.length] !== inputs[i]) {
          const err = Math.abs(values[index + f.input.length] + values[index + 1 + f.input.length] - inputs[i]);
          if (err > 2e-14) failures++;
        }
      }
    }
    receipt.cases.push({ hidden: f.hidden, pre: f.pre, maxError, failures });
  }
  receipt.passed = receipt.cases.every(row => row.failures === 0) && receipt.errors.length === 0;
} catch (error) { receipt.error = error.stack; throw error; }
finally { await browser?.close(); server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); await writeFile(destination, JSON.stringify(receipt, null, 2) + '\n'); }
console.log(JSON.stringify(receipt)); assert(receipt.passed, 'Paired RMSNorm controls failed.');
