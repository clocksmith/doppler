import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';
const destination = process.argv[2];
if (!destination) throw new Error('Provide receipt destination.');
await mkdir(dirname(destination), { recursive: true });
// Optional program: matmul_paired (default) or matmul_paired_compensated, the declared
// replacement with per-lane compensated accumulation; same constants and dispatch.
const program = process.argv[3] ?? 'matmul_paired';
if (program !== 'matmul_paired' && program !== 'matmul_paired_compensated') throw new Error('Program must be matmul_paired or matmul_paired_compensated.');
const source = await readFile(new URL(`../../src/gpu/kernels/${program}.wgsl`, import.meta.url), 'utf8');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
let seed = 624901;
function random() { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 2 ** 32; }
function halfValue(bits) { const e = bits >>> 10 & 31, m = bits & 1023; return (bits & 32768 ? -1 : 1) * (e ? (1024 + m) * 2 ** (e - 25) : m * 2 ** -24); }
const cases = [];
for (const dtype of ['f32', 'f16', 'q4k']) for (const scalar of [false, true]) {
  const M = 3, N = 7, K = 256, input = [], values = [];
  for (let i = 0; i < M * K; i++) input.push(Math.fround((random() - .5) * 2), Math.fround((random() - .5) * 2 ** -26));
  let bytes;
  if (dtype === 'f32') {
    for (let i = 0; i < N * K; i++) values.push(Math.fround((random() - .5) * 3));
    bytes = new Uint8Array(new Float32Array(values).buffer);
  } else if (dtype === 'f16') {
    const raw = new Uint16Array(N * K);
    for (let i = 0; i < raw.length; i++) { raw[i] = 0x3000 + Math.floor(random() * 4096) | (random() < .5 ? 0x8000 : 0); values.push(halfValue(raw[i])); }
    bytes = new Uint8Array(raw.buffer);
  } else {
    bytes = new Uint8Array(N * 144); const view = new DataView(bytes.buffer);
    for (let row = 0; row < N; row++) {
      const base = row * 144, scale = [], minimum = [], quant = [];
      const dBits = 0x2abc, mBits = 0x21ab, d = halfValue(dBits), dm = halfValue(mBits);
      view.setUint16(base, dBits, true); view.setUint16(base + 2, mBits, true);
      for (let j = 0; j < 8; j++) { scale.push(Math.floor(random() * 64)); minimum.push(Math.floor(random() * 64)); }
      for (let j = 0; j < 4; j++) {
        bytes[base + 4 + j] = scale[j] | ((scale[j + 4] >> 4) << 6);
        bytes[base + 8 + j] = minimum[j] | ((minimum[j + 4] >> 4) << 6);
        bytes[base + 12 + j] = (scale[j + 4] & 15) | ((minimum[j + 4] & 15) << 4);
      }
      for (let k = 0; k < K; k++) { quant.push(Math.floor(random() * 16)); values.push(Math.fround(d * scale[k >>> 5] * quant[k] - dm * minimum[k >>> 5])); }
      for (let group = 0; group < 4; group++) for (let k = 0; k < 32; k++) bytes[base + 16 + group * 32 + k] = quant[group * 64 + k] | (quant[group * 64 + 32 + k] << 4);
    }
  }
  cases.push({ dtype, scalar, M, N, K, input, bytes: Array.from(bytes), values, alpha: 1.1 });
}
const receipt = { schema: 'doppler-paired-matmul-physical-v1', ...(program === 'matmul_paired' ? {} : { program }), shaderSha256: hash(source), fixtureSha256: hash(JSON.stringify(cases)), seed: 624901, tolerance: 2e-12,
  scope: 'Independent F64 dot products of exact represented inputs/weights; Q4K fixture built from independent scales/minima/quants, fixed source F32 weight materialization. Public scalar output must match nearest F32 bits.', cases: [] };
const server = createServer((_req, res) => res.end('<!doctype html>'));
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', ...(process.platform === 'darwin' ? ['--use-angle=metal'] : ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'])] });
  const page = await browser.newPage(); await page.goto(`http://127.0.0.1:${server.address().port}`);
  const physical = await page.evaluate(async ({ source, cases }) => {
    const adapter = await navigator.gpu.requestAdapter(); if (!adapter || adapter.info.isFallbackAdapter) throw new Error('Physical adapter required.');
    const device = await adapter.requestDevice(), errors = [], results = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    try {
      const module = device.createShaderModule({ code: source }); const bad = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
      if (bad.length) throw new Error(bad.map(row => row.message).join('\n'));
      for (const f of cases) {
        const owned = []; let staging;
        const alloc = (size, usage) => { const b = device.createBuffer({ size, usage }); owned.push(b); return b; };
        try {
          const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main', constants: { WORKGROUP_SIZE: 64, WEIGHT_F16: f.dtype === 'f16', WEIGHT_Q4K: f.dtype === 'q4k', TRANSPOSE_B: true, SCALAR_OUTPUT: f.scalar } } });
          const uniforms = alloc(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST), raw = new ArrayBuffer(32), view = new DataView(raw);
          [f.M, f.N, f.K, f.M * f.N].forEach((n, i) => view.setUint32(i * 4, n, true)); view.setFloat32(16, Math.fround(f.alpha), true); view.setFloat32(20, f.alpha - Math.fround(f.alpha), true); device.queue.writeBuffer(uniforms, 0, raw);
          const input = alloc(f.input.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST), weight = alloc(f.bytes.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
          device.queue.writeBuffer(input, 0, new Float32Array(f.input)); device.queue.writeBuffer(weight, 0, new Uint8Array(f.bytes));
          const bytes = f.M * f.N * (f.scalar ? 4 : 8), output = alloc(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC); staging = alloc(bytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
          const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [uniforms, input, weight, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(f.M * f.N); pass.end(); encoder.copyBufferToBuffer(output, 0, staging, 0, bytes); device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ); results.push(Array.from(new Float32Array(staging.getMappedRange().slice(0))));
        } finally { if (staging?.mapState === 'mapped') staging.unmap(); await device.queue.onSubmittedWorkDone(); for (const b of owned) b.destroy(); }
      }
      return { results, errors, adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture } };
    } finally { await device.queue.onSubmittedWorkDone(); device.destroy(); }
  }, { source, cases });
  receipt.adapter = physical.adapter; receipt.browser = browser.version(); receipt.errors = physical.errors;
  for (let c = 0; c < cases.length; c++) {
    const f = cases[c], values = physical.results[c]; let maxError = 0, failures = 0;
    for (let row = 0; row < f.M; row++) for (let col = 0; col < f.N; col++) {
      let expected = 0;
      for (let k = 0; k < f.K; k++) expected += (f.input[(row * f.K + k) * 2] + f.input[(row * f.K + k) * 2 + 1]) * f.values[col * f.K + k];
      expected *= f.alpha;
      const index = row * f.N + col, actual = f.scalar ? values[index] : values[index * 2] + values[index * 2 + 1];
      const error = Math.abs(actual - expected) / Math.max(1, Math.abs(expected)); maxError = Math.max(maxError, error);
      if (f.scalar ? !Object.is(actual, Math.fround(expected)) : (!Number.isFinite(error) || error > receipt.tolerance)) failures++;
    }
    receipt.cases.push({ dtype: f.dtype, scalar: f.scalar, maxError, failures });
  }
  receipt.passed = receipt.cases.every(row => row.failures === 0) && receipt.errors.length === 0;
} catch (error) { receipt.error = error.stack; throw error; }
finally { await browser?.close(); server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); await writeFile(destination, JSON.stringify(receipt, null, 2) + '\n'); }
console.log(JSON.stringify(receipt)); assert(receipt.passed, 'Paired matmul controls failed.');
