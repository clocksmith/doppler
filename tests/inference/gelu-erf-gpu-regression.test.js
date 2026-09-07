import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { float32ToFloat16 } from '../../src/converter/quantizer.js';
import { f16ToF32 } from '../../src/loader/dtype-utils.js';
import { runGeLU, recordGeLU } from '../../src/gpu/kernels/gelu.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';

const reference = JSON.parse(await fs.readFile(new URL('../fixtures/gelu-erf-reference.json', import.meta.url)));
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) {
  console.log(`gelu-erf-gpu-regression.test: skipped (${probe.reason})`);
  process.exit(0);
}
const device = getDevice();
try {
  for (const dtype of ['f32', 'f16']) {
    const code = await fs.readFile(new URL(`../../src/gpu/kernels/gelu${dtype === 'f16' ? '_f16' : ''}.wgsl`, import.meta.url), 'utf8');
    const module = device.createShaderModule({ code });
    const input = dtype === 'f16'
      ? Uint16Array.from(reference.inputs, float32ToFloat16) : Float32Array.from(reference.inputs);
    for (const erf of [false, true]) {
      for (const gated of [false, true]) {
        const buffers = [];
        const make = (size, usage) => {
          const buffer = device.createBuffer({ size, usage }); buffers.push(buffer); return buffer;
        };
        try {
          const size = Math.ceil(input.byteLength / 4) * 4;
          const source = make(size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
          const padded = new Uint8Array(size); padded.set(new Uint8Array(input.buffer));
          device.queue.writeBuffer(source, 0, padded);
          const output = make(size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const uniform = make(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
          device.queue.writeBuffer(uniform, 0, new Uint32Array([input.length, 0, 0, 0]));
          const staging = make(size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
          // Omitting GELU_ERF checks that existing tanh execution remains stable.
          const constants = { HAS_GATE: Number(gated), ...(erf ? { GELU_ERF: 1 } : {}) };
          const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main', constants } });
          const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
            entries: [uniform, source, output, source].map((buffer, binding) => ({ binding, resource: { buffer } })) });
          const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(Math.ceil(input.length / 256)); pass.end();
          encoder.copyBufferToBuffer(output, 0, staging, 0, size); device.queue.submit([encoder.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          const result = dtype === 'f16' ? new Uint16Array(staging.getMappedRange()) : new Float32Array(staging.getMappedRange());
          for (let index = 0; index < input.length; index++) {
            const actual = dtype === 'f16' ? f16ToF32(result[index]) : result[index];
            const expected = reference[erf ? 'erf' : 'tanh'][index] * (gated ? reference.inputs[index] : 1);
            const tolerance = (dtype === 'f16' ? 0.01 : 0.000002) * Math.max(1, Math.abs(expected));
            assert(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
              `${dtype}/${erf}/${gated}/${index}: ${actual} != ${expected} (tolerance ${tolerance})`);
          }
          staging.unmap();
        } finally { for (const buffer of buffers) buffer.destroy(); }
      }
    }
  }
  const inputValues = Float32Array.from(reference.inputs);
  const inputBuffer = acquireBuffer(inputValues.byteLength);
  uploadData(inputBuffer, inputValues);
  try {
    const input = createTensor(inputBuffer, 'f32', [inputValues.length], 'source_gelu');
    const step = { op: 'activation', kernel: 'gelu.wgsl', entry: 'main', constants: { GELU_ERF: true, HAS_GATE: false } };
    const options = { size: inputValues.length, phase: 'prefill', layerIdx: 0,
      kernelPath: { prefill: { steps: [step] }, decode: { steps: [step] } } };
    for (const recorded of [false, true]) {
      const recorder = recorded ? new CommandRecorder(device) : null;
      const output = recorded ? await recordGeLU(recorder, input, options) : await runGeLU(input, options);
      try {
        if (recorder) recorder.submit();
        const values = new Float32Array(await readBuffer(output.buffer, inputValues.byteLength));
        for (let index = 0; index < values.length; index++) {
          assert(Math.abs(values[index] - reference.erf[index]) <= 0.000002 * Math.max(1, Math.abs(reference.erf[index])),
            `Declared erf constant was lost in ${recorded ? 'recorded' : 'immediate'} dispatch at ${index}.`);
        }
      } finally { releaseBuffer(output.buffer); }
    }
    await assert.rejects(() => runGeLU(input, { ...options, gate: input }), /HAS_GATE disagrees/);
    await assert.rejects(() => runGeLU(input, { ...options, kernelPath: { decode: { steps: [{ ...step, kernel: 'gelu_f16.wgsl' }] } } }), /does not match declared/);
  } finally { releaseBuffer(inputBuffer); }
  console.log('gelu-erf-gpu-regression.test: ok (f32/f16, erf/tanh, plain/gated)');
} finally { destroyDevice(); releaseNodeWebGPU(); }
