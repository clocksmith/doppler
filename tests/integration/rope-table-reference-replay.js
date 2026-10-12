// Test-only table decomposition: exact declared shader versus independent F64 equations.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';
const [shaderPath, destination, parametersPath] = process.argv.slice(2);
assert(shaderPath && destination);
const source = await readFile(shaderPath, 'utf8');
const hash = value => createHash('sha256').update(value).digest('hex');
const parameters = parametersPath ? JSON.parse(await readFile(parametersPath, 'utf8'))
  : { theta: 10000000, rotaryDim: 64, frequencyBaseDim: 64, scale: 1, positions: 256 };
assert(parameters.theta > 0 && Math.fround(parameters.theta) === parameters.theta);
assert(Number.isSafeInteger(parameters.rotaryDim) && parameters.rotaryDim > 0 && parameters.rotaryDim % 2 === 0);
assert(Number.isSafeInteger(parameters.frequencyBaseDim) && parameters.frequencyBaseDim >= parameters.rotaryDim);
assert(Number.isSafeInteger(parameters.positions) && parameters.positions > 0 && parameters.positions <= 16385);
assert.equal(parameters.scale, 1);
const halfDim = parameters.rotaryDim / 2, count = parameters.positions * halfDim;
const thetaLiteral = /[.e]/i.test(String(parameters.theta)) ? String(parameters.theta) : `${parameters.theta}.0`;
const helpers = source.slice(source.indexOf('fn exp_refined'), source.indexOf('@compute'));
const phaseExpression = source.slice(source.indexOf('    let angle ='), source.indexOf('    cos_values[index]'))
  .replaceAll('USE_INVERSE_FREQUENCIES', 'false').replaceAll('u.scaling_type', '0u')
  .replaceAll('u.theta', thetaLiteral).replaceAll('u.frequency_base_dim', `${parameters.frequencyBaseDim}u`);
const observed = helpers + `
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= ${count}u) { return; }
  let dimension = id.x % ${halfDim}u; let position = id.x / ${halfDim}u;
  let logarithm = log_refined(${thetaLiteral});
  let exponent = f32(dimension * 2u) / ${parameters.frequencyBaseDim}.0;
  let power = -exponent * logarithm;
  let frequency = exp_refined(power);
  let rope_position = f32(position); let scale = 1.0;
  ${phaseExpression}
  output[id.x * 6u] = logarithm; output[id.x * 6u + 1u] = power;
  output[id.x * 6u + 2u] = frequency; output[id.x * 6u + 3u] = angle;
  output[id.x * 6u + 4u] = phase.x; output[id.x * 6u + 5u] = phase.y;
}`;
const receipt = { scope: 'Unscaled declared table and local F64 decomposition; no whole-model acceptance', parameters,
  decompositionScope: 'Intermediate log/power/frequency/angle columns describe canonical rounded stages; a compensated phase may bypass their materialization. Total cosine/sine errors use independent equations.',
  shaderSha256: hash(source), harnessSha256: hash(await readFile(import.meta.filename)) };
const server = createServer((_q, response) => response.end('<!doctype html>'));
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu',
    ...(process.platform === 'darwin' ? ['--use-angle=metal'] : ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'])] });
  const page = await browser.newPage(); await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result = await page.evaluate(async ({ source, observed, parameters, count }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw Error('Physical adapter required');
    const device = await adapter.requestDevice(); const owned = []; const errors = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const buffer = (size, usage) => { const value = device.createBuffer({ size, usage }); owned.push(value); return value; };
    const compile = async code => {
      const module = device.createShaderModule({ code });
      const errors = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
      if (errors.length) throw Error(errors.map(row => row.message).join('\n'));
      return device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    };
    let staging;
    try {
      const original = await compile(source), diagnostic = await compile(observed);
      const uniform = buffer(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const data = new ArrayBuffer(64), u = new Uint32Array(data), f = new Float32Array(data);
      u[0] = parameters.positions; u[1] = parameters.rotaryDim; u[2] = parameters.frequencyBaseDim; f[4] = parameters.theta; f[5] = 1; f[6] = 1;
      u[10] = count; u[14] = 1; u[15] = parameters.positions; device.queue.writeBuffer(uniform, 0, data);
      const dummy = buffer(4, GPUBufferUsage.STORAGE);
      const cos = buffer(count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const sin = buffer(count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const stages = buffer(count * 24, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      staging = buffer(count * 32, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const encoder = device.createCommandEncoder();
      for (const [pipeline, resources, groups] of [[original, [uniform, dummy, cos, sin], Math.ceil(count / 256)], [diagnostic, [stages], Math.ceil(count / 64)]]) {
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: resources.map((resource, binding) => ({ binding, resource: { buffer: resource } })) });
        const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(groups); pass.end();
      }
      encoder.copyBufferToBuffer(cos, 0, staging, 0, count * 4);
      encoder.copyBufferToBuffer(sin, 0, staging, count * 4, count * 4);
      encoder.copyBufferToBuffer(stages, 0, staging, count * 8, count * 24);
      device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
      return { values: Array.from(new Float32Array(staging.getMappedRange().slice(0))), errors,
        adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture } };
    } finally {
      if (staging?.mapState === 'mapped') staging.unmap();
      await device.queue.onSubmittedWorkDone(); for (const item of owned) item.destroy(); device.destroy();
    }
  }, { source, observed, parameters, count });
  receipt.adapter = result.adapter; receipt.errors = result.errors; receipt.browser = browser.version();
  receipt.rows = [];
  let observationMismatches = 0;
  const maxima = {};
  const note = (name, error, index) => { if (!maxima[name] || Math.abs(error) > maxima[name].absoluteError) maxima[name] = { absoluteError: Math.abs(error), error, index }; };
  for (let index = 0; index < count; index++) {
    const position = Math.floor(index / halfDim), dimension = index % halfDim;
    const [logarithm, power, frequency, angle, cosine, sine] = result.values.slice(count * 2 + index * 6, count * 2 + index * 6 + 6);
    if (cosine !== result.values[index] || sine !== result.values[count + index]) observationMismatches++;
    const exactFrequency = parameters.theta ** (-dimension * 2 / parameters.frequencyBaseDim), exactAngle = position * exactFrequency;
    const errors = { logarithm: logarithm - Math.log(parameters.theta), powerLocal: power + dimension * 2 / parameters.frequencyBaseDim * logarithm,
      frequencyLocal: frequency - Math.exp(power), frequencyTotal: frequency - exactFrequency,
      angleLocal: angle - position * frequency, angleTotal: angle - exactAngle,
      cosLocal: cosine - Math.cos(angle), sinLocal: sine - Math.sin(angle),
      cosTotal: result.values[index] - Math.cos(exactAngle), sinTotal: result.values[count + index] - Math.sin(exactAngle) };
    for (const [name, value] of Object.entries(errors)) note(name, value, index);
    receipt.rows.push({ position, dimension, logarithm, power, frequency, angle, cosine: result.values[index], sine: result.values[count + index], errors });
  }
  receipt.maxima = maxima; receipt.observationMismatches = observationMismatches;
  console.log(JSON.stringify({ adapter: receipt.adapter, maxima, observationMismatches }));
  assert.equal(result.errors.length, 0); assert.equal(observationMismatches, 0);
} finally {
  try { await writeFile(destination, JSON.stringify(receipt)); }
  finally { await browser?.close(); server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
}
