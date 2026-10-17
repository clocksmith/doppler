import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { quantizeQ4_KRef } from './reference/dequant.js';
import { dequantQ4KRowsRef } from './reference/fused-ffn-q4k.js';
import { getMatmulConfig } from '../../src/gpu/kernels/matmul-selection.js';
import { calculateMatmulDispatch } from '../../src/gpu/kernels/matmul-dispatch.js';

const argument = (name) => {
  const index = process.argv.indexOf(name);
  return index < 0 ? null : process.argv[index + 1];
};
const sourcePath = argument('--shader')
  ?? new URL('../../src/gpu/kernels/fused_matmul_q4_widetile_direct.wgsl', import.meta.url);
const source = await readFile(sourcePath, 'utf8');
const referencePath = argument('--reference-shader');
const referenceSource = referencePath ? await readFile(referencePath, 'utf8') : null;
const probe = await probeNodeGPU();
if (!probe.ready) {
  assert(!process.argv.includes('--require-gpu'), probe.reason);
  console.log(`fused-matmul-q4-widetile-parity: skipped (${probe.reason})`);
} else {
  const device = getDevice();
  const errors = [];
  device.addEventListener('uncapturederror', (event) => errors.push(event.error.message));
  const shapes = [
    [4, 257, 256, 4, 256],
    [7, 259, 257, 4, 128],
    [9, 65, 513, 8, 64],
    [1, 3, 1, 1, 64],
    [3, 9, 1024, 2, 128],
    [64, 1024, 1024, 4, 256],
  ];
  const rows = [];
  try {
    for (const [M, N, K, TILE_M, TILE_N] of shapes) {
      const buffers = [];
      const makeBuffer = (size, usage, data) => {
        const buffer = device.createBuffer({ size, usage });
        buffers.push(buffer);
        if (data) device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      };
      try {
        const blocks = Math.ceil(K / 256);
        const input = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 0.31) * 0.7);
        const weights = new Uint8Array(N * blocks * 144);
        for (let n = 0; n < N; n++) {
          const row = new Float32Array(blocks * 256);
          for (let k = 0; k < K; k++) row[k] = Math.cos((n * K + k) * 0.73) * 0.1;
          weights.set(quantizeQ4_KRef(row, blocks), n * blocks * 144);
        }
        const dense = dequantQ4KRowsRef(weights, N, K);
        const expected = new Float32Array(M * N);
        for (let m = 0; m < M; m++) {
          for (let n = 0; n < N; n++) {
            let sum = 0;
            for (let k = 0; k < K; k++) sum += input[m * K + k] * dense[n * K + k];
            expected[m * N + n] = sum;
          }
        }
        const data = new ArrayBuffer(32);
        const view = new DataView(data);
        [M, N, K].forEach((value, i) => view.setUint32(i * 4, value, true));
        view.setFloat32(12, 1, true);
        view.setUint32(16, blocks, true);
        const uniform = makeBuffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, data);
        const a = makeBuffer(input.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, input);
        const b = makeBuffer(weights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, weights);
        const c = makeBuffer(expected.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const staging = makeBuffer(expected.byteLength, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        const constants = { TILE_M, TILE_N };
        const config = getMatmulConfig('q4_fused_widetile_direct', constants);
        const { workgroups } = calculateMatmulDispatch('q4_fused_widetile_direct', true, false, false, M, N, config);
        const execute = async (code) => {
          const module = device.createShaderModule({ code });
          const pipeline = await device.createComputePipelineAsync({
            layout: 'auto', compute: { module, entryPoint: 'main', constants },
          });
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
          encoder.copyBufferToBuffer(c, 0, staging, 0, c.size);
          device.queue.submit([encoder.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          try { return staging.getMappedRange().slice(0); } finally { staging.unmap(); }
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
        rows.push({ M, N, K, TILE_M, TILE_N, maxAbsError, byteExactReference: Boolean(referenceSource) });
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
