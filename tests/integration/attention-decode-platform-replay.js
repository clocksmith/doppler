// Replay captured decode operands against a Float64 oracle. No model acceptance.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { gunzipSync } from 'node:zlib';
import { chromium } from 'playwright';
import { float16ToFloat32 } from '../../src/converter/quantizer.js';

const [fixturePath, shaderPath, destination] = process.argv.slice(2);
assert(fixturePath && shaderPath && destination, 'Supply capture, shader and receipt paths');
const fixtureBytes = gunzipSync(await readFile(fixturePath));
const fixture = JSON.parse(fixtureBytes);
const hash = value => createHash('sha256').update(value).digest('hex');
const original = await readFile(shaderPath, 'utf8');
assert.equal(hash(original), fixture.contract.shaderSha256, 'Replay requires the captured shader');
const records = fixture.data.map(host => host.captures.captures.flatMap(capture => capture.records));
const inputs = records.map(rows => rows.find(row => row.boundary === 'inputs'));
const outputs = records.map(rows => rows.find(row => row.boundary === 'outputs'));
assert(inputs.every(row => row?.numTokens === 1 && row.plan.kv.layout === 'contiguous'));
assert.equal(inputs[0].plan.id, inputs[1].plan.id);
const tensor = (row, role) => row.tensors.find(item => item.role === role);
for (const role of ['q', 'cachedK', 'cachedV']) {
  assert.equal(tensor(inputs[0], role).data, tensor(inputs[1], role).data,
    `Local-error replay requires identical ${role} operands`);
}
const values = item => {
  const bytes = Buffer.from(item.data, 'base64');
  assert.equal(bytes.length, item.bytes);
  return Array.from({ length: item.elements }, (_, index) => item.dtype === 'f16'
    ? float16ToFloat32(bytes.readUInt16LE(index * 2)) : bytes.readFloatLE(index * 4));
};
const geometry = inputs[0], kvLen = geometry.plan.kv.length;
assert.deepEqual(fixture.uniforms, [geometry.numHeads, geometry.numKVHeads, geometry.headDim,
  kvLen, 1, geometry.scale, 1, kvLen - 1, 0, 0, 0, 0, geometry.plan.kv.pageSize, 0, 0],
  'This retained reproduction covers causal contiguous attention without softcap or a window');
const q = values(tensor(geometry, 'q'));
const k = values(tensor(geometry, 'cachedK')), v = values(tensor(geometry, 'cachedV'));
const expected = [];
for (let head = 0; head < geometry.numHeads; head++) {
  const kvHead = Math.floor(head / (geometry.numHeads / geometry.numKVHeads));
  const scores = Array.from({ length: kvLen }, (_, position) => {
    let sum = 0;
    for (let d = 0; d < geometry.headDim; d++) {
      sum += q[head * geometry.headDim + d]
        * k[(position * geometry.numKVHeads + kvHead) * geometry.headDim + d];
    }
    return sum * geometry.scale;
  });
  const maximum = Math.max(...scores), weights = scores.map(score => Math.exp(score - maximum));
  const denominator = weights.reduce((sum, weight) => sum + weight, 0);
  for (let d = 0; d < geometry.headDim; d++) {
    let sum = 0;
    for (let position = 0; position < kvLen; position++) {
      sum += weights[position] * v[(position * geometry.numKVHeads + kvHead) * geometry.headDim + d];
    }
    expected.push(sum / denominator);
  }
}
const candidate = process.env.DOPPLER_ATTENTION_DIAGNOSTIC ?? null;
assert(candidate === null || (candidate === 'compensated-qk'
  && process.env.DOPPLER_TEST_ONLY_ARITHMETIC === '1'), 'Arithmetic intervention requires test-only authorization');
let source = original;
if (candidate) {
  assert.equal(source.split('                var dot: f32 = 0.0;').length, 2);
  source = source.replace('                var dot: f32 = 0.0;',
    '                var dot: f32 = 0.0;\n                var correction: f32 = 0.0;');
  for (const suffix of ['0', '1']) {
    const line = `dot = dot + q${suffix} * k${suffix};`;
    assert.equal(source.split(line).length, 2);
    source = source.replace(line, `let product${suffix} = fma(q${suffix}, k${suffix}, 0.0);
                    let total${suffix} = fma(1.0, dot, product${suffix});
                    let error${suffix} = select(fma(1.0, dot, fma(-1.0, total${suffix}, product${suffix})),
                        fma(1.0, product${suffix}, fma(-1.0, total${suffix}, dot)), abs(dot) >= abs(product${suffix}));
                    correction = fma(1.0, correction, error${suffix});
                    dot = total${suffix};`);
  }
  source = source.replace('score = dot * u.scale;', 'score = fma(1.0, dot, correction) * u.scale;');
}
const backends = { darwin: ['--use-angle=metal'],
  linux: ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'] };
assert(Object.hasOwn(backends, process.platform));
const host = process.platform === 'darwin' ? 'mac' : 'linux';
const captured = values(tensor(outputs[fixture.data.findIndex(row => row.host === host)], 'core'));
const receipt = { scope: 'Identical captured operands; isolated decode attention, not model acceptance',
  host, candidate, sourceSubstitution: candidate !== null, fixtureSha256: hash(fixtureBytes),
  originalShaderSha256: hash(original), shaderSha256: hash(source), contract: fixture.contract };
const compare = (actual, reference) => {
  assert.equal(actual.length, reference.length);
  const errors = actual.map((value, index) => Math.abs(value - reference[index]));
  assert(errors.every(Number.isFinite));
  return { maxError: Math.max(...errors), rmsError: Math.sqrt(errors.reduce((s, e) => s + e * e, 0) / errors.length) };
};
const server = createServer((_request, response) => response.end('<!doctype html>'));
await new Promise(done => server.listen(0, '127.0.0.1', done));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true,
    args: ['--enable-unsafe-webgpu', ...backends[process.platform]] });
  receipt.browser = browser.version();
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result = await page.evaluate(async ({ source, fixture, geometry }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw Error('Physical GPU required');
    const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16', 'subgroups'] });
    const owned = [], errors = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const buffer = (size, usage, data) => {
      const resource = device.createBuffer({ size, usage }); owned.push(resource);
      if (data) device.queue.writeBuffer(resource, 0, data);
      return resource;
    };
    let staging;
    try {
      const module = device.createShaderModule({ code: source });
      const diagnostics = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
      if (diagnostics.length) throw Error(diagnostics.map(row => row.message).join('; '));
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: {
        module, entryPoint: fixture.contract.entryPoint, constants: fixture.contract.constants } });
      const bytes = new ArrayBuffer(64), view = new DataView(bytes);
      const u = fixture.uniforms;
      for (const [index, value] of u.entries()) {
        if (index === 5 || index === 8) view.setFloat32(index * 4, value, true);
        else view.setUint32(index * 4, value, true);
      }
      const input = role => {
        const tensor = geometry.tensors.find(item => item.role === role);
        return buffer(tensor.bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          Uint8Array.from(atob(tensor.data), char => char.charCodeAt(0)));
      };
      const output = buffer(geometry.numHeads * geometry.headDim * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const resources = [buffer(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, bytes),
        input('q'), input('cachedK'), input('cachedV'), output,
        buffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, new Uint32Array([u[3]])),
        buffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, new Uint32Array(1))];
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: resources.map((resource, binding) => ({ binding, resource: { buffer: resource } })) });
      staging = buffer(output.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(geometry.numHeads); pass.end();
      encoder.copyBufferToBuffer(output, 0, staging, 0, output.size);
      device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
      const values = Array.from(new Float32Array(staging.getMappedRange().slice(0)));
      return { values, errors, adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture } };
    } finally {
      if (staging?.mapState === 'mapped') staging.unmap();
      await device.queue.onSubmittedWorkDone().catch(() => {});
      for (const resource of owned) resource.destroy(); device.destroy();
    }
  }, { source, fixture, geometry });
  Object.assign(receipt, result);
  receipt.comparison = compare(result.values, expected);
  receipt.capturedReference = compare(captured, expected);
  receipt.capturedReplay = compare(result.values, captured);
  receipt.valuesSha256 = hash(Buffer.from(Float32Array.from(result.values).buffer));
  assert.deepEqual(result.errors, []);
  if (!candidate) assert.equal(receipt.capturedReplay.maxError, 0, 'Baseline must reproduce actual captured output');
  else {
    receipt.regressionMaxError = 4e-7;
    assert(receipt.comparison.maxError <= receipt.regressionMaxError,
      'Captured decode correction exceeds the independent regression bound');
    assert(receipt.comparison.maxError < receipt.capturedReference.maxError / 2,
      'Correction must materially improve independent accuracy');
  }
  receipt.passed = true;
  console.log(JSON.stringify({ host, candidate, comparison: receipt.comparison, capturedReplay: receipt.capturedReplay,
    valuesSha256: receipt.valuesSha256 }));
} catch (error) { receipt.passed = false; receipt.failure = error.message; throw error; }
finally {
  await writeFile(destination, JSON.stringify(receipt));
  await browser?.close(); await new Promise(done => server.close(done));
}
