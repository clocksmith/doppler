/** Replay captured linear-attention operands. Diagnostic only; no model acceptance. */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { buildLinearActivationDiagnostic } from '../kernels/linear-activation-diagnostic.js';

const [capturePath, installedPackage, reploidRoot, output, operation] = process.argv.slice(2);
assert(capturePath && installedPackage && reploidRoot && output && process.env.REPLOID_EXECUTOR_WS);
assert(['conv', 'recurrent'].includes(operation));
const bytes = await readFile(capturePath), capture = JSON.parse(bytes);
const records = capture.captures.flatMap(c => c.records);
const input = records.find(r => r.boundary === 'linear-inputs');
const expected = records.find(r => r.boundary === 'linear-outputs');
assert(input && expected);
const originalShader = await readFile(resolve(installedPackage, `src/gpu/kernels/gated_delta_${operation}.wgsl`), 'utf8');
const observeActivation = process.env.DOPPLER_LINEAR_OBSERVE_ACTIVATION === '1';
const activationCandidate = process.env.DOPPLER_LINEAR_ACTIVATION_DIAGNOSTIC === '1';
assert(!observeActivation || operation === 'conv');
let shader = observeActivation ? originalShader.replace(
  '    conv_out[token_idx * params.conv_dim + channel] = silu(mixed);',
  `    let z = ${activationCandidate ? 'diagnostic_exp' : 'exp'}(-abs(mixed));
    observed_activation[token_idx * params.conv_dim + channel] = vec4<f32>(mixed, z, 1.0 + z, ${activationCandidate ? 'diagnostic_reciprocal(1.0 + z)' : '1.0 / (1.0 + z)'});
    conv_out[token_idx * params.conv_dim + channel] = silu(mixed);`)
  + '\n@group(0) @binding(5) var<storage, read_write> observed_activation: array<vec4<f32>>;'
  : originalShader;
if (activationCandidate) shader = buildLinearActivationDiagnostic(shader);
const hash = b => createHash('sha256').update(b).digest('hex');
const { chromium } = createRequire(resolve(reploidRoot, 'package.json'))('playwright');
const receipt = { scope: 'Identical captured operands through one linear-attention operation', operation,
  captureSha256: hash(bytes), shaderSha256: hash(shader), archiveSha256: capture.archiveSha256,
  sourceSubstitution: observeActivation || activationCandidate, observeActivation, activationCandidate,
  originalShaderSha256: hash(originalShader),
  upstreamSourceSubstitution: capture.sourceSubstitution, results: [] };
for (const platform of ['mac', 'linux']) {
  const browser = platform === 'mac'
    ? await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] })
    : await chromium.connect(process.env.REPLOID_EXECUTOR_WS);
  const context = await browser.newContext();
  try {
    const page = await context.newPage(); await page.goto('http://localhost:8000/config/chat-files.json');
    const result = await page.evaluate(async ({ input, expected, shader, operation, observeActivation }) => {
      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter.requestDevice({ requiredLimits:
        operation === 'recurrent' ? { maxStorageBuffersPerShaderStage: 9 } : {} });
      const owned = [], p = input.params;
      const buffer = (size, usage) => { const b = device.createBuffer({ size, usage }); owned.push(b); return b; };
      try {
        const module = device.createShaderModule({ code: shader });
        const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
        if (errors.length) throw Error(errors.map(m => m.message).join('\n'));
        const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: {
          module, entryPoint: 'main', constants: { WORKGROUP_SIZE: operation === 'conv' ? 256 : 128 } } });
        const uniforms = new ArrayBuffer(64), view = new DataView(uniforms);
        ['numTokens', 'convDim', 'convKernelSize', 'numVHeads', 'numKHeads', 'headKDim',
          'headVDim', 'qSize', 'kSize', 'valueDim', 'qRep'].forEach((key, i) => view.setUint32(i * 4, p[key], true));
        view.setUint32(44, p.normMode === 'per_head' ? 1 : 0, true);
        view.setFloat32(48, p.rmsNormEps, true); view.setFloat32(52, p.qkL2NormEps, true);
        view.setUint32(56, Number(p.abPacked) | Number(p.qkvzPacked) << 1, true);
        view.setUint32(60, p.bProjOffsetElements, true);
        const uniform = buffer(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        device.queue.writeBuffer(uniform, 0, uniforms);
        const outputRole = operation === 'conv' ? 'convOutput' : 'coreOutput';
        const roles = operation === 'conv' ? ['qkv', 'convWeight', 'convState', outputRole]
          : ['convOutput', 'z', 'a', 'b', 'dtBias', 'aLog', 'normWeight', 'recurrentState', outputRole];
        const buffers = roles.map(role => {
          const tensor = (role === 'convOutput' || role === 'coreOutput' ? expected : input).tensors.find(t => t.role === role);
          if (!tensor) throw Error(`Missing captured ${role}`);
          const b = buffer(tensor.bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
          if (role !== outputRole) device.queue.writeBuffer(b, 0, Uint8Array.from(atob(tensor.data), c => c.charCodeAt(0)));
          return b;
        });
        const size = p.numTokens * (operation === 'conv' ? p.convDim : p.valueDim) * 4;
        const trace = observeActivation ? buffer(size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) : null;
        const traceReadback = observeActivation ? buffer(size * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ) : null;
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
          [uniform, ...buffers, ...(trace ? [trace] : [])]
            .map((buffer, binding) => ({ binding, resource: { buffer } })) });
        const staging = buffer(size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(operation === 'conv' ? Math.ceil(p.convDim / 256) : p.numVHeads); pass.end();
        encoder.copyBufferToBuffer(buffers.at(-1), 0, staging, 0, size);
        if (trace) encoder.copyBufferToBuffer(trace, 0, traceReadback, 0, size * 4);
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Uint8Array(staging.getMappedRange().slice(0)); staging.unmap();
        let text = '';
        for (let i = 0; i < data.length; i += 16384) text += String.fromCharCode(...data.subarray(i, i + 16384));
        let activationData = null;
        if (traceReadback) {
          await traceReadback.mapAsync(GPUMapMode.READ);
          const data = new Uint8Array(traceReadback.getMappedRange().slice(0)); traceReadback.unmap();
          let text = '';
          for (let i = 0; i < data.length; i += 16384) text += String.fromCharCode(...data.subarray(i, i + 16384));
          activationData = btoa(text);
        }
        return { data: btoa(text), activationData, vendor: adapter.info.vendor, architecture: adapter.info.architecture };
      } finally { for (const b of owned) b.destroy(); device.destroy(); }
    }, { input, expected, shader, operation, observeActivation });
    receipt.results.push({ platform, browser: browser.version(), ...result });
    console.log(JSON.stringify({ platform, operation, completed: true }));
  } finally { await context.close(); await browser.close(); }
}
const values = text => { const b = Buffer.from(text, 'base64'); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const compare = (a, b) => {
  assert.equal(a.length, b.length); let maxDifference = 0, differentElements = 0;
  for (let i = 0; i < a.length; i++) {
    assert(Number.isFinite(a[i]) && Number.isFinite(b[i]));
    maxDifference = Math.max(maxDifference, Math.abs(a[i] - b[i])); differentElements += a[i] !== b[i];
  }
  return { elements: a.length, differentElements, maxDifference };
};
const actual = receipt.results.map(r => values(r.data));
receipt.crossPlatform = compare(...actual);
if (observeActivation) {
  const traces = receipt.results.map(r => values(r.activationData));
  receipt.activationComparison = ['mixed', 'exponential', 'denominator', 'reciprocal'].map((stage, lane) => ({ stage,
    ...compare(...traces.map(t => t.filter((_, i) => i % 4 === lane))) }));
}
const expectedValues = values(expected.tensors.find(t => t.role === (operation === 'conv' ? 'convOutput' : 'coreOutput')).data);
receipt.macCaptureAgreement = compare(actual[0], expectedValues.subarray(0, actual[0].length));
if (operation === 'conv') {
  const p = input.params, read = role => values(input.tensors.find(t => t.role === role).data);
  const qkv = read('qkv'), weights = read('convWeight'), state = Float64Array.from(read('convState'));
  const reference = new Float64Array(p.numTokens * p.convDim);
  for (let t = 0; t < p.numTokens; t++) for (let c = 0; c < p.convDim; c++) {
    const base = c * p.convKernelSize;
    for (let k = 0; k < p.convKernelSize - 1; k++) state[base + k] = state[base + k + 1];
    state[base + p.convKernelSize - 1] = qkv[t * (p.convDim + (p.qkvzPacked ? p.valueDim : 0)) + c];
    let mixed = 0;
    for (let k = 0; k < p.convKernelSize; k++) mixed += state[base + k] * weights[base + k];
    reference[t * p.convDim + c] = mixed / (1 + Math.exp(-mixed));
  }
  receipt.reference = { scope: 'Independent Float64 convolution and SiLU; never fed into inference',
    sha256: hash(Buffer.from(reference.buffer)), comparisons: actual.map((v, i) => ({ platform: receipt.results[i].platform,
      ...compare(v, reference), rmsError: Math.sqrt(v.reduce((sum, x, j) => sum + (x - reference[j]) ** 2, 0) / v.length) })) };
}
await writeFile(output, JSON.stringify(receipt));
console.log(JSON.stringify({ crossPlatform: receipt.crossPlatform, macCaptureAgreement: receipt.macCaptureAgreement,
  reference: receipt.reference, activationComparison: receipt.activationComparison }));
