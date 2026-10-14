// Node/Dawn port of paired-matmul-bench.js, for the native stage hosts. Diagnostic only:
// is matmul_paired.wgsl bound by its compensated arithmetic or by weight reads/decode?
// usage: node paired-matmul-bench-node.js <webgpu module dir> <matmul_paired.wgsl> <out.json> [tokens] [N] [K]
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';

const [moduleDir, kernelPath, destination] = process.argv.slice(2, 5);
if (!moduleDir || !kernelPath || !destination) throw new Error('Provide webgpu module dir, kernel path and result destination.');
const tokens = (process.argv[5] || '1,4,16').split(',').map(Number);
const N = Number(process.argv[6] || 3584), K = Number(process.argv[7] || 1024);
const { create, globals } = createRequire(import.meta.url)(moduleDir);
Object.assign(globalThis, globals);
const shipped = await readFile(kernelPath, 'utf8');
const loop = 'sum = pair_add(sum, pair_mul_scalar(input[row * u.K + k], weight_value(column, k)));';
const reduce = 'partials[lane.x] = pair_add(partials[lane.x], partials[lane.x + stride]);';
if (!shipped.includes(loop) || !shipped.includes(reduce)) throw new Error('Kernel text changed; update the controls.');
const variants = {
  shipped,
  'plain-f32': shipped.replace(loop, 'sum = vec2<f32>(sum.x + input[row * u.K + k].x * weight_value(column, k), 0.0);')
    .replace(reduce, 'partials[lane.x] = partials[lane.x] + partials[lane.x + stride];'),
  'paired-no-weight': shipped.replace(loop, 'sum = pair_add(sum, pair_mul_scalar(input[row * u.K + k], f32(k & 7u) * 0.125 + 0.5));'),
};
let seed = 91731;
const random = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 2 ** 32; };
const weight = new Uint8Array(N * (K / 256) * 144);
{ const view = new DataView(weight.buffer);
  for (let block = 0; block < N * K / 256; block++) {
    const base = block * 144; view.setUint16(base, 0x2abc, true); view.setUint16(base + 2, 0x21ab, true);
    for (let i = 4; i < 144; i++) weight[base + i] = Math.floor(random() * 256);
  } }
const maxM = Math.max(...tokens), input = new Float32Array(maxM * K * 2);
for (let i = 0; i < maxM * K; i++) { input[i * 2] = Math.fround((random() - .5) * 2); input[i * 2 + 1] = Math.fround((random() - .5) * 2 ** -26); }

const result = { schema: 'scratch.paired-matmul-bench/v1', surface: 'node-dawn', N, K, tokens, runs: [], errors: [] };
try {
  const gpu = create([]);
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter || adapter.info?.isFallbackAdapter) throw new Error('Physical adapter required.');
  result.adapter = { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description };
  const device = await adapter.requestDevice();
  device.addEventListener?.('uncapturederror', event => result.errors.push(event.error.message));
  const buffer = (data, usage) => { const b = device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, data); return b; };
  const weights = buffer(weight, GPUBufferUsage.STORAGE), inputs = buffer(input, GPUBufferUsage.STORAGE);
  const output = device.createBuffer({ size: maxM * N * 8, usage: GPUBufferUsage.STORAGE });
  const gridLimit = device.limits.maxComputeWorkgroupsPerDimension;
  // Explicit layout: an 'auto' layout drops the weight binding from the no-weight control.
  const bindGroupLayout = device.createBindGroupLayout({ entries: ['uniform', 'read-only-storage', 'read-only-storage', 'storage']
    .map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  for (const [name, code] of Object.entries(variants)) {
    const module = device.createShaderModule({ code });
    const bad = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
    if (bad.length) throw new Error(name + ': ' + bad.map(row => row.message).join('\n'));
    // Synchronous creation: the Oct 13 Intel crash was inside async pipeline creation.
    const pipeline = device.createComputePipeline({ layout, compute: { module, entryPoint: 'main',
      constants: { WORKGROUP_SIZE: 64, WEIGHT_F16: 0, WEIGHT_Q4K: 1, TRANSPOSE_B: 1, SCALAR_OUTPUT: 0 } } });
    for (const M of tokens) {
      const width = Math.min(M * N, gridLimit), raw = new ArrayBuffer(32), view = new DataView(raw);
      [M, N, K, width].forEach((n, i) => view.setUint32(i * 4, n, true)); view.setFloat32(16, 1, true);
      const uniforms = buffer(new Uint8Array(raw), GPUBufferUsage.UNIFORM);
      const group = device.createBindGroup({ layout: bindGroupLayout, entries: [uniforms, inputs, weights, output].map((b, binding) => ({ binding, resource: { buffer: b } })) });
      const once = async () => {
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(width, Math.ceil(M * N / width)); pass.end();
        const start = performance.now(); device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
        return performance.now() - start;
      };
      await once(); await once();
      const samples = []; for (let i = 0; i < 5; i++) samples.push(await once());
      samples.sort((a, b) => a - b);
      result.runs.push({ variant: name, M, medianMs: samples[2], minMs: samples[0], maxMs: samples[4], errorsSoFar: result.errors.length });
      uniforms.destroy();
    }
  }
  await device.queue.onSubmittedWorkDone(); device.destroy();
} finally { await writeFile(destination, JSON.stringify(result, null, 2) + '\n'); }
for (const row of result.runs) console.log(row.variant.padEnd(18), 'M=' + String(row.M).padEnd(3), row.medianMs.toFixed(2), 'ms', `(${row.minMs.toFixed(2)}-${row.maxMs.toFixed(2)})`);
if (result.errors.length) console.log('errors', result.errors);
process.exit(0);
