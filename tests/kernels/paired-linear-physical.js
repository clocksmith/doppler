import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';
const destination = process.argv[2];
if (!destination) throw new Error('Provide receipt destination.');
await mkdir(dirname(destination), { recursive: true });
const sources = await Promise.all(['gated_delta_conv_paired', 'qwen_linear_attention_prepare_paired', 'qwen_linear_attention_scale_paired', 'qwen_linear_attention_log_decay_paired', 'qwen_linear_attention_decay_paired', 'gated_delta_recurrent_paired'].map(name =>
  readFile(new URL(`../../src/gpu/kernels/${name}.wgsl`, import.meta.url), 'utf8')));
const hash = value => createHash('sha256').update(value).digest('hex');
let seed = 624901;
function random() { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 2 ** 32; }
const pairValues = (count, scale) => Array.from({ length: count }, () => [Math.fround((random() - .5) * scale), Math.fround((random() - .5) * scale * 2 ** -26)]).flat();
const scalarValues = (count, scale) => Array.from({ length: count }, () => Math.fround((random() - .5) * scale));
const cases = [false, true].map(packed => {
  const tokens = 3, nk = 1, nv = 2, kd = 128, vd = 128, kernel = 4, convDim = (2 * nk * kd + nv * vd), valueDim = nv * vd;
  return { tokens, nk, nv, kd, vd, kernel, convDim, valueDim, packed, eps: 1e-6, qeps: 1e-6, offset: packed,
    qkv: pairValues(tokens * (convDim + (packed ? valueDim : 0)), .8), z: pairValues(tokens * valueDim, 1.4),
    a: pairValues(tokens * nv, 1.2), b: pairValues(tokens * nv, 1.2),
    weight: scalarValues(convDim * kernel, .8), bias: scalarValues(nv, .3), log: scalarValues(nv, .8),
    norm: scalarValues(vd, .5), convState: pairValues(convDim * kernel, .2), state: pairValues(nv * kd * vd, .01) };
});
function reference(f) {
  const unpack = values => Array.from({ length: values.length / 2 }, (_, i) => values[i * 2] + values[i * 2 + 1]);
  const qkv = unpack(f.qkv), z = unpack(f.z), a = unpack(f.a), b = unpack(f.b), convState = unpack(f.convState), state = unpack(f.state);
  const conv = new Array(f.tokens * f.convDim), output = [];
  const sigmoid = value => 1 / (1 + Math.exp(-value));
  for (let t = 0; t < f.tokens; t++) {
    for (let c = 0; c < f.convDim; c++) {
      const base = c * f.kernel;
      for (let k = 0; k < f.kernel - 1; k++) convState[base + k] = convState[base + k + 1];
      convState[base + f.kernel - 1] = qkv[t * (f.convDim + (f.packed ? f.valueDim : 0)) + c];
      let sum = 0; for (let k = 0; k < f.kernel; k++) sum += convState[base + k] * f.weight[base + k];
      conv[t * f.convDim + c] = sum * sigmoid(sum);
    }
    for (let h = 0; h < f.nv; h++) {
      const row = t * f.convDim, kh = Math.floor(h / (f.nv / f.nk)), qb = kh * f.kd, kb = f.nk * f.kd + qb;
      const q = conv.slice(row + qb, row + qb + f.kd), k = conv.slice(row + kb, row + kb + f.kd);
      const qi = 1 / Math.sqrt(q.reduce((s, v) => s + v * v, 0) + f.qeps) / Math.sqrt(f.kd);
      const ki = 1 / Math.sqrt(k.reduce((s, v) => s + v * v, 0) + f.qeps);
      const beta = sigmoid(b[t * f.nv + h]);
      const decay = Math.exp(-Math.exp(f.log[h]) * Math.log1p(Math.exp(a[t * f.nv + h] + f.bias[h])));
      const raw = [];
      for (let v = 0; v < f.vd; v++) {
        let pred = 0;
        for (let d = 0; d < f.kd; d++) { const index = (h * f.kd + d) * f.vd + v; state[index] *= decay; pred += state[index] * k[d] * ki; }
        const delta = (conv[row + 2 * f.nk * f.kd + h * f.vd + v] - pred) * beta;
        let value = 0;
        for (let d = 0; d < f.kd; d++) { const index = (h * f.kd + d) * f.vd + v; state[index] += k[d] * ki * delta; value += state[index] * q[d] * qi; }
        raw.push(value);
      }
      const inv = 1 / Math.sqrt(raw.reduce((s, v) => s + v * v, 0) / f.vd + f.eps);
      for (let v = 0; v < f.vd; v++) {
        const gate = f.packed ? qkv[t * (f.convDim + f.valueDim) + f.convDim + h * f.vd + v] : z[t * f.valueDim + h * f.vd + v];
        output.push(raw[v] * inv * (f.norm[v] + Number(f.offset)) * gate * sigmoid(gate));
      }
    }
  }
  return { conv, convState, state, output };
}
const receipt = { schema: 'doppler-paired-linear-physical-v1', shaderSha256: sources.map(hash), fixtureSha256: hash(JSON.stringify(cases)), seed: 624901,
  tolerance: 2e-11, scope: 'Independent F64 convolution and gated-delta equations, initial state, source epsilon, paired persistent state, QKV/Z and A/B packing. No model qualification.', cases: [] };
const server = createServer((_req, res) => res.end('<!doctype html>'));
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', ...(process.platform === 'darwin' ? ['--use-angle=metal'] : ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'])] });
  const page = await browser.newPage(); await page.goto(`http://127.0.0.1:${server.address().port}`);
  const physical = await page.evaluate(async ({ sources, cases }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter || adapter.limits.maxStorageBuffersPerShaderStage < 9) throw new Error('Physical adapter with nine storage bindings required.');
    const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 9 } });
    const errors = [], results = []; device.addEventListener('uncapturederror', e => errors.push(e.error.message));
    try {
      const pipelines = [];
      for (const code of sources) {
        const module = device.createShaderModule({ code }); const bad = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
        if (bad.length) throw new Error(bad.map(row => row.message).join('\n'));
        pipelines.push(await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } }));
      }
      for (const f of cases) {
        const owned = [], mapped = [];
        const alloc = (size, usage) => { const b = device.createBuffer({ size, usage }); owned.push(b); return b; };
        const storage = data => { const b = alloc(data.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC); device.queue.writeBuffer(b, 0, new Float32Array(data)); return b; };
        try {
          const u = new ArrayBuffer(80), dv = new DataView(u);
          [f.tokens, f.convDim, f.kernel, f.nv, f.nk, f.kd, f.vd, f.nk * f.kd, f.nk * f.kd, f.valueDim, f.nv / f.nk, 0].forEach((v, i) => dv.setUint32(i * 4, v, true));
          [Math.fround(f.eps), f.eps - Math.fround(f.eps), Math.fround(f.qeps), f.qeps - Math.fround(f.qeps)].forEach((v, i) => dv.setFloat32(48 + i * 4, v, true));
          dv.setUint32(64, f.packed ? 3 : 0, true); dv.setUint32(68, f.tokens * f.nv, true); dv.setUint32(72, Number(f.offset), true);
          const uniform = alloc(80, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST); device.queue.writeBuffer(uniform, 0, u);
          const qkv = storage(f.qkv), z = f.packed ? qkv : storage(f.z), a = storage(f.packed ? [...f.a, ...f.b] : f.a), b = f.packed ? a : storage(f.b);
          const cw = storage(f.weight), cs = storage(f.convState), state = storage(f.state), dt = storage(f.bias), log = storage(f.log), norm = storage(f.norm);
          const conv = storage(new Array(f.tokens * f.convDim * 2).fill(0)), output = storage(new Array(f.tokens * f.valueDim * 2).fill(0));
          const gates = storage(new Array(f.tokens * f.nv * 4).fill(0));
          const bindings = [[uniform, qkv, cw, cs, conv], [uniform, b, gates], [uniform, log, gates], [uniform, a, dt, gates], [uniform, gates], [uniform, conv, z, gates, norm, state, output]];
          const encoder = device.createCommandEncoder();
          for (let i = 0; i < pipelines.length; i++) {
            const group = device.createBindGroup({ layout: pipelines[i].getBindGroupLayout(0), entries: bindings[i].map((buffer, binding) => ({ binding, resource: { buffer } })) });
            const pass = encoder.beginComputePass(); pass.setPipeline(pipelines[i]); pass.setBindGroup(0, group); pass.dispatchWorkgroups(i === 0 ? Math.ceil(f.convDim / 64) : i < pipelines.length - 1 ? Math.ceil(f.tokens * f.nv / 64) : f.nv); pass.end();
          }
          const readbacks = [conv, cs, state, output, gates].map(buffer => { const b = alloc(buffer.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ); encoder.copyBufferToBuffer(buffer, 0, b, 0, buffer.size); return b; });
          device.queue.submit([encoder.finish()]);
          const values = [];
          for (const buffer of readbacks) { await buffer.mapAsync(GPUMapMode.READ); mapped.push(buffer); values.push(Array.from(new Float32Array(buffer.getMappedRange().slice(0)))); }
          results.push(values);
        } finally { for (const b of mapped) b.unmap(); await device.queue.onSubmittedWorkDone(); for (const b of owned) b.destroy(); }
      }
      return { results, errors, adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture }, storageBindingLimit: device.limits.maxStorageBuffersPerShaderStage };
    } finally { await device.queue.onSubmittedWorkDone(); device.destroy(); }
  }, { sources, cases });
  receipt.adapter = physical.adapter; receipt.browser = browser.version(); receipt.errors = physical.errors; receipt.storageBindingLimit = physical.storageBindingLimit;
  for (let c = 0; c < cases.length; c++) {
    const expected = reference(cases[c]), rows = {};
    const f = cases[c];
    expected.gates = Array.from({ length: f.tokens * f.nv }, (_, index) => {
      const b = f.b[index * 2] + f.b[index * 2 + 1];
      const a = f.a[index * 2] + f.a[index * 2 + 1];
      return [1 / (1 + Math.exp(-b)), Math.exp(-Math.exp(f.log[index % f.nv]) * Math.log1p(Math.exp(a + f.bias[index % f.nv])))];
    }).flat();
    for (const [i, name] of ['conv', 'convState', 'state', 'output', 'gates'].entries()) {
      let maxError = 0, failures = 0; const actual = physical.results[c][i];
      for (let j = 0; j < expected[name].length; j++) {
        const error = Math.abs(actual[j * 2] + actual[j * 2 + 1] - expected[name][j]) / Math.max(1, Math.abs(expected[name][j]));
        maxError = Math.max(maxError, error); if (!Number.isFinite(error) || error > receipt.tolerance) failures++;
      }
      rows[name] = { count: expected[name].length, maxError, failures };
    }
    receipt.cases.push({ packed: cases[c].packed, rows });
  }
  receipt.passed = receipt.errors.length === 0 && receipt.cases.every(c => Object.values(c.rows).every(row => row.failures === 0));
} catch (error) { receipt.error = error.stack; throw error; }
finally { await browser?.close(); server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); await writeFile(destination, JSON.stringify(receipt, null, 2) + '\n'); }
console.log(JSON.stringify(receipt)); assert(receipt.passed, 'Paired linear controls failed.');
