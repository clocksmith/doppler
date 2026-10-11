// Physical replay of retained operands; independent of model acceptance.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';
import { createF16ProjectionCases } from '../helpers/f16-projection-cases.js';

const [operandsPath, baselinePath, candidatePath, destination, geometry = 'base'] = process.argv.slice(2);
assert(operandsPath && baselinePath && candidatePath && destination);
assert(['base', 'tiled', 'q4-widetile'].includes(geometry));
const bytes = await readFile(operandsPath);
const fixture = JSON.parse(bytes);
assert.equal(fixture.schema, geometry === 'q4-widetile'
  ? 'doppler.q4-projection-actual-operands/v1' : 'doppler.f16-projection-actual-operands/v1');
if (geometry !== 'q4-widetile') fixture.cases = [...fixture.cases.map(item => ({ ...item, kind: 'actual', tileSize: 16, transpose: 1, alpha: 1 })),
  ...createF16ProjectionCases({ transpose: geometry === 'tiled' ? 1 : null })];
const hash = value => createHash('sha256').update(value).digest('hex');
const sources = await Promise.all([baselinePath, candidatePath].map(path => readFile(path, 'utf8')));
const server = createServer((_request, response) => response.end('<!doctype html>'));
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu',
    '--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'] });
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const physical = await page.evaluate(async ({ fixture, sources, geometry }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || !adapter.features.has('shader-f16')) throw new Error('Physical F16 WebGPU adapter required');
    const info = { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description };
    if (/software|swiftshader|llvmpipe/i.test(JSON.stringify(info))) throw new Error('Software adapter rejected');
    const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16'] });
    const results = [];
    const decode = value => Uint8Array.from(atob(value), character => character.charCodeAt(0));
    const encode = value => {
      let text = '';
      for (let i = 0; i < value.length; i += 16384) text += String.fromCharCode(...value.subarray(i, i + 16384));
      return btoa(text);
    };
    try {
      for (const [variant, source] of sources.entries()) {
        const module = device.createShaderModule({ code: source });
        const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
        if (errors.length) throw new Error(errors.map(message => message.message).join('\n'));
        for (const item of fixture.cases) {
          const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: {
            module, entryPoint: 'main', constants: geometry === 'q4-widetile' ? { TILE_M: item.tileM ?? 4, TILE_N: item.tileN ?? 256 }
              : geometry === 'tiled' ? {} : { TILE_SIZE: item.tileSize },
          } });
          const owned = [];
          const allocate = (size, usage) => { const buffer = device.createBuffer({ size, usage }); owned.push(buffer); return buffer; };
          device.pushErrorScope('validation');
          try {
            const uniform = allocate(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
            const input = allocate(item.M * item.K * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
            const weightBytes = geometry === 'q4-widetile' ? decode(item.weightQ4)
              : new Uint8Array(Math.ceil(item.N * item.K * 2 / 4) * 4);
            if (geometry !== 'q4-widetile') weightBytes.set(decode(item.weightF16));
            const weight = allocate(weightBytes.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
            const output = allocate(item.M * item.N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            const staging = allocate(item.M * item.N * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
            const data = new ArrayBuffer(32), view = new DataView(data);
            view.setUint32(0, item.M, true); view.setUint32(4, item.N, true); view.setUint32(8, item.K, true);
            view.setFloat32(12, item.alpha, true); view.setUint32(16, geometry === 'q4-widetile' ? Math.ceil(item.K / 256) : item.transpose, true);
            device.queue.writeBuffer(uniform, 0, data);
            device.queue.writeBuffer(input, 0, decode(item.inputF32));
            device.queue.writeBuffer(weight, 0, weightBytes);
            const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
              entries: [uniform, input, weight, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
            const submit = (count, readback) => {
              const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
              pass.setPipeline(pipeline); pass.setBindGroup(0, group);
              const tile = geometry === 'tiled' ? 64 : item.tileSize;
              for (let i = 0; i < count; i++) {
                if (geometry === 'q4-widetile') pass.dispatchWorkgroups(Math.ceil(item.N / (item.tileN ?? 256)), Math.ceil(item.M / (item.tileM ?? 4)));
                else pass.dispatchWorkgroups(Math.ceil(item.M / tile), Math.ceil(item.N / tile));
              }
              pass.end();
              if (readback) encoder.copyBufferToBuffer(output, 0, staging, 0, item.M * item.N * 4);
              device.queue.submit([encoder.finish()]);
            };
            submit(1, true);
            await staging.mapAsync(GPUMapMode.READ);
            const result = encode(new Uint8Array(staging.getMappedRange().slice(0))); staging.unmap();
            const batchWallMs = [];
            for (let repeat = 0; repeat < (item.kind === 'actual' ? 5 : 0); repeat++) {
              const start = performance.now(); submit(32, false);
              await device.queue.onSubmittedWorkDone(); batchWallMs.push(performance.now() - start);
            }
            results.push({ variant: variant === 0 ? 'baseline' : 'candidate', name: item.name,
              outputF32: result, batchWallMs, dispatchesPerBatch: 32 });
          } finally {
            for (const buffer of owned) buffer.destroy();
            const error = await device.popErrorScope();
            if (error) throw new Error(error.message);
          }
        }
      }
      return { adapter: info, results };
    } finally { device.destroy(); }
  }, { fixture, sources, geometry });
  const values = (encoded, Constructor) => {
    const data = Buffer.from(encoded, 'base64');
    return new Constructor(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
  };
  for (const row of physical.results) {
    const input = fixture.cases.find(item => item.name === row.name);
    const actual = values(row.outputF32, Float32Array), expected = values(input.expectedF64, Float64Array);
    const observed = input.kind === 'actual' ? values(input.observedF32, Float32Array) : null;
    const bounds = input.kind === 'synthetic' ? values(input.errorBoundF64, Float64Array) : null;
    let outsideErrorBound = 0;
    let maxAbsError = 0, squaredError = 0, differentObserved = 0;
    for (let i = 0; i < actual.length; i++) {
      assert(Number.isFinite(actual[i]) && Number.isFinite(expected[i]));
      const error = Math.abs(actual[i] - expected[i]);
      maxAbsError = Math.max(maxAbsError, error); squaredError += error * error;
      if (observed && actual[i] !== observed[i]) differentObserved++;
      if (bounds && error > bounds[i]) outsideErrorBound++;
    }
    Object.assign(row, { maxAbsError, rmsError: Math.sqrt(squaredError / actual.length),
      differentObserved: observed ? differentObserved : null, outsideErrorBound, kind: input.kind,
      shape: { M: input.M, N: input.N, K: input.K, transpose: input.transpose, alpha: input.alpha, tileSize: input.tileSize, tileM: input.tileM, tileN: input.tileN } });
  }
  const receipt = { scope: 'Exact retained operands; F32 accumulation diagnostic, not model acceptance or performance claim',
    browser: browser.version(), geometry, operandsSha256: hash(bytes), baselineSha256: hash(sources[0]), candidateSha256: hash(sources[1]), ...physical };
  await writeFile(destination, JSON.stringify(receipt, null, 2) + '\n');
  console.log(JSON.stringify({ ...receipt, results: receipt.results.map(({ outputF32, ...row }) => row) }));
  for (const name of fixture.cases.filter(item => item.kind === 'actual').map(item => item.name)) {
    const baseline = receipt.results.find(row => row.name === name && row.variant === 'baseline');
    const candidate = receipt.results.find(row => row.name === name && row.variant === 'candidate');
    assert.equal(baseline.differentObserved, 0, `${name}: baseline reproduction`);
    assert(candidate.maxAbsError < baseline.maxAbsError / (geometry === 'q4-widetile' ? 1.5 : 5), `${name}: partial sums must materially reduce local error`);
  }
  for (const row of receipt.results.filter(item => item.kind === 'synthetic')) {
    assert.equal(row.outsideErrorBound, 0, `${row.variant}/${row.name}: general shape precision`);
  }
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
