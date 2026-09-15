import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { WgslScanner } from 'wgsl_reflect/wgsl_reflect.module.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { KERNEL_CONFIGS } from '../../src/config/kernel-registry-contract.js';
import { getUniformByteLength, writeUniformsFromObject } from '../../src/gpu/kernels/uniform-utils.js';
import { reflectWgslInterface } from '../../tools/lib/wgsl-interface.js';

const probe = await probeNodeGPU();
if (!probe.ready) throw new Error(`Uniform echo requires WebGPU: ${probe.reason}`);
const device = getDevice();
const caps = getKernelCapabilities();
assert.doesNotMatch(JSON.stringify(caps.adapterInfo), /swiftshader|llvmpipe|software/i);

function originalStruct(source, name) {
  const tokens = new WgslScanner(source).scanTokens();
  const index = tokens.findIndex((token, i) => token.lexeme === 'struct' && tokens[i + 1]?.lexeme === name);
  assert.ok(index >= 0, `struct ${name}`);
  let end = index + 2;
  while (tokens[end].lexeme !== '}') end++;
  return source.slice(tokens[index].start, tokens[end].end);
}

async function echo(struct, name, updates, expressions, label) {
  const buffers = [];
  device.pushErrorScope('validation');
  try {
    const code = `${struct}\n@group(0) @binding(0) var<uniform> params: ${name};
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1) fn echo() {\n${expressions.map((expression, i) => `output[${i}] = ${expression};`).join('\n')}\n}`;
    const module = device.createShaderModule({ code, label });
    const messages = (await module.getCompilationInfo()).messages.filter((message) => message.type === 'error');
    assert.deepEqual(messages, [], label);
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'echo' } });
    const make = (size, usage) => { const buffer = device.createBuffer({ size, usage }); buffers.push(buffer); return buffer; };
    const { data, expected } = updates[0];
    const uniform = make(data.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const output = make(expected.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const staging = make(expected.length * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: { buffer: output } },
    ] });
    for (const update of updates) {
      device.queue.writeBuffer(uniform, 0, update.data);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(output, 0, staging, 0, expected.length * 4);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      try { assert.deepEqual([...new Uint32Array(staging.getMappedRange())], update.expected, label); }
      finally { staging.unmap(); }
    }
  } finally {
    for (const buffer of buffers) buffer.destroy();
    assert.equal(await device.popErrorScope(), null, `${label} validation`);
  }
}

const seen = new Set();
let fields = 0;
try {
  for (const configs of Object.values(KERNEL_CONFIGS)) for (const config of Object.values(configs)) {
    if (!config.uniforms) continue;
    const identity = JSON.stringify(config.uniforms);
    if (seen.has(identity)) continue;
    seen.add(identity);
    const source = await fs.readFile(new URL(`../../src/gpu/kernels/${config.shaderFile}`, import.meta.url), 'utf8');
    const reflected = reflectWgslInterface(source).uniforms[0];
    const data = new ArrayBuffer(getUniformByteLength(config));
    const view = new DataView(data);
    new Uint8Array(data).fill(0xAA);
    const updates = [];
    const expressions = reflected.fields.map((field) => `bitcast<u32>(params.${field.name})`);
    for (const request of [0, 1]) {
      const values = {};
      for (const field of config.uniforms.fields) if (!field.padding) {
        values[field.name] = field.type === 'f32'
          ? (request === 0 ? -1 : 1) * (field.offset + 0.125)
          : (request === 0 ? 0xF0000000 : 0xA0000000) + field.offset;
      }
      writeUniformsFromObject(view, config, values);
      // Expected values come from the logical request, independently of the
      // generated writer's bytes and offsets. The GPU interprets the original struct.
      const expected = reflected.fields.map((field) => {
        const bits = new DataView(new ArrayBuffer(4));
        const padding = config.uniforms.fields.find((entry) => entry.name === field.name).padding;
        const value = padding ? 0 : values[field.name];
        if (field.type === 'f32') bits.setFloat32(0, value, true);
        else bits.setUint32(0, value, true);
        return bits.getUint32(0, true);
      });
      updates.push({ data: data.slice(0), expected });
    }
    await echo(originalStruct(source, reflected.structName), reflected.structName, updates, expressions, `${config.operation}/${config.variant}`);
    fields += reflected.fields.length;
  }
  const guide = await fs.readFile(new URL('../../docs/style/wgsl-style-guide.md', import.meta.url), 'utf8');
  const example = guide.match(/```wgsl\n\/\/ uniform-layout-example\n([\s\S]*?)```/)[1];
  const data = new ArrayBuffer(32);
  const view = new DataView(data);
  view.setUint32(0, 1, true);
  [11, 22, 33].forEach((value, i) => view.setUint32(16 + i * 4, value, true));
  await echo(example, 'Uniforms', [{ data, expected: [1, 11, 22, 33] }], ['params.has_residual', 'params._pad.x', 'params._pad.y', 'params._pad.z'], 'WGSL guide vec3 alignment');
  console.log(JSON.stringify({ test: 'kernel-uniform-echo-physical', passed: true, layouts: seen.size, fields,
    requestsPerLayout: 2, reusedGpuBuffers: true, guideExamples: 1, adapter: caps.adapterInfo, evidence: 'Generated writers through original uniform structs on physical WebGPU; interface verification, not numerical model qualification.' }));
} finally { destroyDevice(); }
