import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { getMatmulConfig } from '../../src/gpu/kernels/matmul-selection.js';
import { calculateMatmulDispatch } from '../../src/gpu/kernels/matmul-dispatch.js';
import { float32ToFloat16 } from './reference/dequant.js';
import { f16ToF32 } from '../../src/loader/dtype-utils.js';

const argument = (name) => {
  const index = process.argv.indexOf(name);
  return index < 0 ? null : process.argv[index + 1];
};
const sourcePath = argument('--shader')
  ?? new URL('../../src/gpu/kernels/matmul_f16w_f32a_tiled.wgsl', import.meta.url);
const source = await readFile(sourcePath, 'utf8');
// --reference-shader <path> additionally requires byte-identical output to a prior shader.
const referencePath = argument('--reference-shader');
const referenceSource = referencePath ? await readFile(referencePath, 'utf8') : null;
const probe = await probeNodeGPU();
if (!probe.ready) {
  assert(!process.argv.includes('--require-gpu'), probe.reason);
  console.log(`matmul-f16w-f32a-tiled-parity: skipped (${probe.reason})`);
} else {
  const device = getDevice();
  const errors = [];
  device.addEventListener('uncapturederror', (event) => errors.push(event.error.message));
  // Partial M/N tiles, K not a multiple of TILE_K, and full-size FFN shapes.
  const shapes = [
    [1, 1, 1],
    [3, 5, 7],
    [64, 64, 16],
    [65, 63, 17],
    [130, 151, 200],
    [102, 3072, 1024],
    [37, 1024, 3072],
  ];
  const rows = [];
  try {
    for (const [M, N, K] of shapes) {
      const buffers = [];
      const makeBuffer = (size, usage, data) => {
        const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
        buffers.push(buffer);
        if (data) device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      };
      try {
        const input = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 0.31) * 0.7);
        const weightBits = Uint16Array.from({ length: N * K }, (_, i) => float32ToFloat16(Math.cos(i * 0.73) * 0.1));
        const expected = new Float64Array(M * N);
        for (let m = 0; m < M; m++) {
          for (let n = 0; n < N; n++) {
            let sum = 0;
            for (let k = 0; k < K; k++) sum += input[m * K + k] * f16ToF32(weightBits[n * K + k]);
            expected[m * N + n] = sum;
          }
        }
        const data = new ArrayBuffer(32);
        const view = new DataView(data);
        [M, N, K].forEach((value, i) => view.setUint32(i * 4, value, true));
        view.setFloat32(12, 1, true);
        view.setUint32(16, 1, true);
        const uniform = makeBuffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, data);
        const a = makeBuffer(input.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, input);
        const paddedWeights = new Uint16Array(Math.ceil(weightBits.length / 2) * 2);
        paddedWeights.set(weightBits);
        const b = makeBuffer(paddedWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, paddedWeights);
        const c = makeBuffer(M * N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const staging = makeBuffer(M * N * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        const config = getMatmulConfig('f16w_f32a_tiled', {});
        const { workgroups } = calculateMatmulDispatch('f16w_f32a_tiled', true, false, false, M, N, config);
        const execute = async (code) => {
          const module = device.createShaderModule({ code });
          const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
          const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [uniform, a, b, c].map((buffer, binding) => ({ binding, resource: { buffer } })),
          });
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(...workgroups);
          pass.end();
          encoder.copyBufferToBuffer(c, 0, staging, 0, M * N * 4);
          device.queue.submit([encoder.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          try { return staging.getMappedRange(0, M * N * 4).slice(0); } finally { staging.unmap(); }
        };
        const result = await execute(source);
        const values = new Float32Array(result);
        let maxAbsError = 0;
        for (let i = 0; i < values.length; i++) {
          const error = Math.abs(values[i] - expected[i]);
          assert(Number.isFinite(values[i]) && error <= 1e-4 + 2e-5 * Math.abs(expected[i]),
            `M=${M} N=${N} K=${K} output ${i}: ${values[i]} vs ${expected[i]}`);
          maxAbsError = Math.max(maxAbsError, error);
        }
        if (referenceSource) assert.deepEqual(new Uint8Array(result), new Uint8Array(await execute(referenceSource)));
        rows.push({ M, N, K, workgroups, maxAbsError, byteExactReference: Boolean(referenceSource) });
      } finally {
        await device.queue.onSubmittedWorkDone().catch(() => {});
        for (const buffer of buffers) buffer.destroy();
      }
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ passed: true, shaderSha256: createHash('sha256').update(source).digest('hex'),
      provider: probe.providerReceipt, rows }, null, 2));
  } finally {
    destroyDevice();
  }
}
