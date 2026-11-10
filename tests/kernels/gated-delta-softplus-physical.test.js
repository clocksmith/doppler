import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';

// Exercise the functions actually compiled into the recurrent and decode paths.
// The independent oracle is binary64 log1p(exp(x)), rounded once to binary32.
const shaderFiles = [
  'gated_delta_recurrent.wgsl',
  'gated_delta_recurrent_f16.wgsl',
  'gated_delta_fused_decode.wgsl',
  'gated_delta_fused_decode_f16.wgsl',
  'gated_delta_fused_decode_paired.wgsl',
  'gated_delta_fused_decode_paired_f16.wgsl',
  'gated_delta_recurrent_sigmoid_gate.wgsl',
  'gated_delta_recurrent_sigmoid_gate_f16.wgsl',
  'gated_delta_fused_decode_sigmoid_gate.wgsl',
  'gated_delta_fused_decode_sigmoid_gate_f16.wgsl',
  'gated_delta_fused_decode_paired_sigmoid_gate.wgsl',
  'gated_delta_fused_decode_paired_sigmoid_gate_f16.wgsl',
  'gated_delta_fused_decode_register.wgsl',
  'gated_delta_recurrent_register_prefill.wgsl',
  'gated_delta_recurrent_gguf.wgsl',
  'gated_delta_recurrent_gguf_register_prefill.wgsl',
  'gated_delta_prefill_norm_gate.wgsl',
];
const helperNames = ['softplus', 'exp_refined', 'log_refined', 'reciprocal_refined'];
function functionSource(source, name) {
  const start = source.indexOf(`fn ${name}(`);
  assert(start >= 0, `Missing ${name}`);
  const body = source.indexOf('{', start);
  let depth = 1, end = body + 1;
  while (depth && end < source.length) {
    if (source[end] === '{') depth++;
    if (source[end] === '}') depth--;
    end++;
  }
  assert.equal(depth, 0, `Unclosed ${name}`);
  return source.slice(start, end);
}
const bits = new DataView(new ArrayBuffer(4));
function floatBits(value) { bits.setFloat32(0, value, true); return bits.getUint32(0, true); }
function fromBits(value) { bits.setUint32(0, value, true); return bits.getFloat32(0, true); }
const inputValues = [-80, -40, -30, -20, -18, -16, -12, -10, -8, -4, -1, 0, 1, 4, 10, 20, 40, 80];
for (const boundary of [-20, 20, Math.fround(Math.log(1 / 64))]) {
  for (let delta = -3; delta <= 3; delta++) inputValues.push(fromBits(floatBits(boundary) + delta));
}
for (let i = 0; i <= 8192; i++) inputValues.push(Math.fround(-80 + 120 * i / 8192));
const input = new Float32Array([...new Set(inputValues)].sort((a, b) => a - b));
const reference = Float32Array.from(input, value => Math.log1p(Math.exp(value)));
const maxUlp = 6; // Includes rounded exp, log and compensation; no cancellation-sized absolute allowance.
const errors = [];
const receipt = { schema: 'doppler-gated-delta-softplus-physical-v1',
  inputSha256: createHash('sha256').update(new Uint8Array(input.buffer)).digest('hex'),
  inputCount: input.length, acceptance: { maxUlp, positiveForFiniteInput: true, monotonic: true },
  scope: 'Scalar softplus on finite normal-range inputs; not whole-model qualification.', variants: [] };
const probe = await probeNodeGPU();
if (!probe.ready) throw new Error(`Physical softplus test requires WebGPU: ${probe.reason}`);
const device = getDevice();
receipt.adapter = getKernelCapabilities().adapterInfo;
assert.doesNotMatch(JSON.stringify(receipt.adapter), /swiftshader|llvmpipe|software/i);
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
try {
  for (const file of shaderFiles) {
    const source = await readFile(new URL(`../../src/gpu/kernels/${file}`, import.meta.url), 'utf8');
    const helpers = helperNames.map(name => functionSource(source, name)).join('\n');
    const code = `${helpers}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < arrayLength(&input)) { output[id.x] = softplus(input[id.x]); }
}`;
    const owned = [];
    const allocate = (size, usage) => { const buffer = device.createBuffer({ size, usage }); owned.push(buffer); return buffer; };
    let staging;
    device.pushErrorScope('validation');
    try {
      const module = device.createShaderModule({ code });
      const messages = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
      assert.deepEqual(messages, [], file);
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const gpuInput = allocate(input.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const output = allocate(input.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      staging = allocate(input.byteLength, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      device.queue.writeBuffer(gpuInput, 0, input);
      const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [gpuInput, output]
        .map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(input.length / 64)); pass.end();
      encoder.copyBufferToBuffer(output, 0, staging, 0, input.byteLength);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(staging.getMappedRange().slice(0));
      let worst = { ulp: 0 }, violations = 0, monotonic = true;
      for (let i = 0; i < input.length; i++) {
        const ulp = Math.abs(floatBits(actual[i]) - floatBits(reference[i]));
        if (ulp > worst.ulp) worst = { ulp, input: input[i], actual: actual[i], expected: reference[i] };
        if (!(actual[i] > 0) || !Number.isFinite(actual[i]) || ulp > maxUlp) violations++;
        if (i && actual[i] < actual[i - 1]) monotonic = false;
      }
      receipt.variants.push({ file, shaderSha256: createHash('sha256').update(source).digest('hex'),
        testedShaderSha256: createHash('sha256').update(code).digest('hex'), worst, violations, monotonic,
        minusTwenty: { actual: actual[input.indexOf(-20)], expected: reference[input.indexOf(-20)] } });
    } finally {
      if (staging?.mapState === 'mapped') staging.unmap();
      await device.queue.onSubmittedWorkDone();
      for (const buffer of owned) buffer.destroy();
      const validation = await device.popErrorScope();
      if (validation) errors.push(validation.message);
    }
  }
} finally {
  destroyDevice();
  receipt.errors = errors;
  receipt.passed = errors.length === 0 && receipt.variants.length === shaderFiles.length
    && receipt.variants.every(row => row.violations === 0 && row.monotonic);
  if (process.argv[2]) await writeFile(process.argv[2], `${JSON.stringify(receipt, null, 2)}\n`);
}
console.log(JSON.stringify(receipt));
assert(receipt.passed, 'Gated-delta softplus violates the independent scalar oracle.');
