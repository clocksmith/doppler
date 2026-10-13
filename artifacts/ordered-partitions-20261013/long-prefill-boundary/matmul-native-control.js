// Bounded diagnostic: installed paired matrix multiplication, synthetic inputs.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const config = JSON.parse(await readFile(process.argv[2]));
const app = config.applicationDirectory;
const deliveryBytes = await readFile(app + '/native-delivery.json');
assert.equal(sha(deliveryBytes), config.nativeDeliverySha256);
const delivery = JSON.parse(deliveryBytes);
for (const entry of delivery.entries) {
  assert(!entry.path.startsWith('/') && !entry.path.split('/').includes('..'));
  assert.equal(sha(await readFile(app + '/' + entry.path)), entry.sha256);
}
const requireInstalled = createRequire(app + '/package.json');
const runtime = app + '/node_modules/doppler-gpu/src/';
const { verifyNumericalTestDelivery } = await import(pathToFileURL(
  '/var/tmp/reploid-recurrent-memory-20261013/cold-reference/numerical-test-delivery.js'));
const verifiedRuntime = await verifyNumericalTestDelivery({ ...config.testDelivery,
  runtimeDirectory: app + '/node_modules/doppler-gpu' });
const providerPath = requireInstalled.resolve('webgpu');
const { createNativeGpuHost } = await import(pathToFileURL(app + '/server/native-participant.js'));
const { initDevice, getDevice } = await import(pathToFileURL(runtime + 'gpu/device.js'));
const { snapshotTensor } = await import(pathToFileURL(runtime + 'debug/tensor.js'));
const { setGPUDevice } = await import(pathToFileURL(runtime + 'debug/config.js'));
const host = createNativeGpuHost(() => import(pathToFileURL(providerPath)));
const hash = sha;
const source = await readFile(runtime + 'gpu/kernels/matmul_paired.wgsl','utf8');
let seed = 624901;
function random() { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 2 ** 32; }
function halfValue(bits) { const e = bits >>> 10 & 31, m = bits & 1023; return (bits & 32768 ? -1 : 1) * (e ? (1024 + m) * 2 ** (e - 25) : m * 2 ** -24); }
const cases = [];
for (const dtype of ['f32', 'q4k']) for (const scalar of [false]) {
  const M = 8, N = 6144, K = 1024, input = [], values = [];
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
    bytes = new Uint8Array(N * (K / 256) * 144); const view = new DataView(bytes.buffer);
    for (let block = 0; block < N * (K / 256); block++) {
      const base = block * 144, scale = [], minimum = [], quant = [];
      const dBits = 0x2abc, mBits = 0x21ab, d = halfValue(dBits), dm = halfValue(mBits);
      view.setUint16(base, dBits, true); view.setUint16(base + 2, mBits, true);
      for (let j = 0; j < 8; j++) { scale.push(Math.floor(random() * 64)); minimum.push(Math.floor(random() * 64)); }
      for (let j = 0; j < 4; j++) {
        bytes[base + 4 + j] = scale[j] | ((scale[j + 4] >> 4) << 6);
        bytes[base + 8 + j] = minimum[j] | ((minimum[j + 4] >> 4) << 6);
        bytes[base + 12 + j] = (scale[j + 4] & 15) | ((minimum[j + 4] & 15) << 4);
      }
      for (let k = 0; k < 256; k++) { quant.push(Math.floor(random() * 16)); values.push(Math.fround(d * scale[k >>> 5] * quant[k] - dm * minimum[k >>> 5])); }
      for (let group = 0; group < 4; group++) for (let k = 0; k < 32; k++) bytes[base + 16 + group * 32 + k] = quant[group * 64 + k] | (quant[group * 64 + 32 + k] << 4);
    }
  }
  cases.push({ dtype, scalar, M, N, K, input, bytes, values, alpha: 1.1 });
}
const receipt = {schema:'doppler.paired-matmul-dispatch-control/v1', verifiedRuntime,
 shaderSha256:hash(source),fixtureSha256:hash(JSON.stringify(cases.map(f => ({dtype:f.dtype,scalar:f.scalar,M:f.M,N:f.N,K:f.K,alpha:f.alpha,input:hash(new Uint8Array(new Float32Array(f.input).buffer)),weights:hash(f.bytes)})))),seed:624901,tolerance:2e-12,
 scope:'Eight synthetic rows at original projection width; raw installed shader, independent F64 dots; not model qualification',cases:[],closed:false};
try { console.log('fixture-ready'); await host.open(); await initDevice(); console.log('device-ready');
const physical = await (async ({source,cases}) => {
    const device = getDevice(), adapter = { info: device.adapterInfo }, errors = [], results = [], timings = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    try {
      const module = device.createShaderModule({ code: source }); const bad = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
      if (bad.length) throw new Error(bad.map(row => row.message).join('\n'));
      for (const f of cases) {
        const owned = []; let staging;
        const alloc = (size, usage) => { const b = device.createBuffer({ size, usage }); owned.push(b); return b; };
        try {
          console.log('compile-start:'+f.dtype); const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main', constants: { WORKGROUP_SIZE: 64, WEIGHT_F16: Number(f.dtype === 'f16'), WEIGHT_Q4K: Number(f.dtype === 'q4k'), TRANSPOSE_B: 1, SCALAR_OUTPUT: Number(f.scalar) } } });
          console.log('compile-complete:'+f.dtype); const uniforms = alloc(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST), raw = new ArrayBuffer(32), view = new DataView(raw);
          [f.M, f.N, f.K, f.M * f.N].forEach((n, i) => view.setUint32(i * 4, n, true)); view.setFloat32(16, Math.fround(f.alpha), true); view.setFloat32(20, f.alpha - Math.fround(f.alpha), true); device.queue.writeBuffer(uniforms, 0, raw);
          const input = alloc(f.input.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST), weight = alloc(f.bytes.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
          device.queue.writeBuffer(input, 0, new Float32Array(f.input)); device.queue.writeBuffer(weight, 0, new Uint8Array(f.bytes));
          const bytes = f.M * f.N * (f.scalar ? 4 : 8), output = alloc(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC); staging = alloc(bytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
          const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [uniforms, input, weight, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
          const start = performance.now(); const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(f.M * f.N); pass.end(); encoder.copyBufferToBuffer(output, 0, staging, 0, bytes); device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ); timings.push({dtype:f.dtype, M:f.M,N:f.N,K:f.K,milliseconds:performance.now()-start}); console.log(JSON.stringify(timings.at(-1))); results.push(Array.from(new Float32Array(staging.getMappedRange().slice(0))));
        } finally { if (staging?.mapState === 'mapped') staging.unmap(); await device.queue.onSubmittedWorkDone(); for (const b of owned) b.destroy(); }
      }
      return { results, errors, timings, adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture } };
    } finally { await device.queue.onSubmittedWorkDone(); }
  })({ source, cases });
receipt.adapter=physical.adapter;receipt.errors=physical.errors;receipt.timings=physical.timings;
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
} catch (error) { receipt.error = error.stack; process.exitCode = 1; }
finally { try { await host.close(); receipt.closed = true; } catch(error) {receipt.cleanupFailure=error.stack;process.exitCode=1;}
await writeFile(process.argv[3], JSON.stringify(receipt,null,2)+'\n', {flag:'wx'}); }
if (!receipt.passed) process.exitCode=1;
