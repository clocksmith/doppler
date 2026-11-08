import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { readbackBuffer } from '../../src/gpu/readback-buffer.js';
import { f32ToF16Array } from '../../src/inference/kv-cache/types.js';
import { getMatmulConfig } from '../../src/gpu/kernels/matmul-selection.js';
import { calculateMatmulDispatch, createMatmulUniformBuffer } from '../../src/gpu/kernels/matmul-dispatch.js';

const gpu = await probeNodeGPU();
if (!gpu.ready) {
  assert(!process.argv.includes('--require-gpu'), gpu.reason);
  console.log('q4k-f16acc-wide-grid: skipped: ' + gpu.reason); process.exit(0);
}
const device = getDevice(), variant = 'q4_fused_batched_f16acc_f16a';
const config = getMatmulConfig(variant, null);
const code = await fs.readFile(new URL('../../src/gpu/kernels/fused_matmul_q4_batched_f16acc_f16a.wgsl', import.meta.url), 'utf8');
const pipeline = await device.createComputePipelineAsync({ layout: 'auto',
  compute: { module: device.createShaderModule({ code }), entryPoint: config.entryPoint } });
const rows = [];
try {
  for (const [M, N] of [[1, 248320], [5, 65539], [7, 17]]) {
    const owned = [];
    const buffer = (bytes, usage, data) => {
      const b = device.createBuffer({ size: bytes, usage }); owned.push(b);
      if (data) device.queue.writeBuffer(b, 0, data); return b;
    };
    try {
      const K = 256, blockBytes = 144, scales = f32ToF16Array(new Float32Array([1/32, 2/32, 3/32, 4/32]));
      const weights = new Uint8Array(N * blockBytes), view = new DataView(weights.buffer);
      // Every block contains constant exactly representable weights, but columns
      // differ. F64 dot products and all F16 intermediate sums are exact here.
      for (let n = 0; n < N; n++) {
        const start = n * blockBytes; view.setUint16(start, scales[n % 4], true);
        weights.fill(1, start + 4, start + 8); weights.fill(1, start + 12, start + 16);
        weights.fill(0x11, start + 16, start + blockBytes);
      }
      const input = new Float32Array(M * K);
      for (let m = 0; m < M; m++) input.fill((m + 1) / K, m * K, (m + 1) * K);
      const inputBuffer = buffer(M * K * 2, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, f32ToF16Array(input));
      const weightBuffer = buffer(weights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, weights);
      const outputBytes = Math.ceil(M * N * 2 / 4) * 4 + 16;
      const sentinel = new Uint16Array(outputBytes / 2).fill(0x7e00);
      const output = buffer(outputBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, sentinel);
      const plan = calculateMatmulDispatch(variant, true, false, false, M, N, config);
      const uniform = createMatmulUniformBuffer('wide-grid', config, M, N, K, 1, true, true, plan.uniformWorkgroupsX, null, device);
      owned.push(uniform);
      const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: weightBuffer } }, { binding: 4, resource: { buffer: output } },
      ] });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(...plan.workgroups); pass.end();
      device.queue.submit([encoder.finish()]);
      const actual = new Uint16Array(await readbackBuffer(device, output, outputBytes));
      const expected = Array.from({ length: M }, (_, m) => f32ToF16Array(Float32Array.from([1,2,3,4], d => (m + 1) * d / 32)));
      for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
        assert.equal(actual[m * N + n], expected[m][n % 4], `Unwritten/wrong output at row ${m}, column ${n}`);
      }
      assert(actual.subarray(M * N).every(v => v === 0x7e00), 'Partial column planes overwrote the output guard');
      rows.push({ M, N, K, workgroups: plan.workgroups, outputsChecked: M * N, bitExact: true, guardPreserved: true });
    } finally { for (const b of owned) b.destroy(); }
  }
  console.log(JSON.stringify({ schema: 'doppler.q4k-f16acc-wide-grid/v1', provider: gpu.providerReceipt, rows }));
} finally { destroyDevice(); await releaseNodeWebGPU(); }
