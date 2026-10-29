import assert from 'node:assert/strict';
import { quantizeToQ4KMRowWise, dequantizeQ4KMRowWise } from '../../src/converter/quantizer.js';
import { destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { runFusedFFN } from '../../src/gpu/kernels/fused_ffn.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';

const probe = await probeNodeGPU();
if (!probe.ready || !getKernelCapabilities().hasSubgroups) {
  console.log(`fused-ffn-subgroup8-gpu-regression: skipped (${probe.reason ?? 'subgroups unavailable'})`);
  if (probe.ready) await destroyDevice();
  process.exit(0);
}
function upload(values) {
  const buffer = acquireBuffer(values.byteLength);
  uploadData(buffer, values);
  return buffer;
}
async function check(K, N, activation, clampMax) {
  const input = Float32Array.from({ length: K }, (_, k) => Math.sin(k * 0.037) * 0.2);
  const gate = Float32Array.from({ length: N * K }, (_, k) => Math.cos(k * 0.013) * 0.2);
  const up = Float32Array.from({ length: N * K }, (_, k) => Math.sin(k * 0.007) * 0.15);
  const packedGate = quantizeToQ4KMRowWise(gate, [N, K]).quantized;
  const packedUp = quantizeToQ4KMRowWise(up, [N, K]).quantized;
  const decodedGate = dequantizeQ4KMRowWise(packedGate, [N, K]);
  const decodedUp = dequantizeQ4KMRowWise(packedUp, [N, K]);
  const buffers = [upload(input), upload(packedGate), upload(packedUp)];
  const tensor = createTensor(buffers[0], 'f32', [1, K]);
  const weights = buffers.slice(1).map(buffer => createWeightBuffer(buffer, 'q4k', 'row', [N, K]));
  let result = null;
  try {
    result = await runFusedFFN(tensor, ...weights, K, N, {
      activation, alpha: 0.75, swigluLimit: clampMax, variant: 'q4k_subgroup8',
      pipelineConstants: { WORKGROUP_SIZE: 256, COLS_PER_WG: 32, THREADS_PER_COL: 8,
        USE_FULL_BLOCK_FAST_PATH: K % 256 === 0 },
    });
    const actual = new Float32Array(await readBuffer(result.buffer, N * 4));
    for (let n = 0; n < N; n++) {
      let g = 0, u = 0;
      for (let k = 0; k < K; k++) {
        g += input[k] * decodedGate[n * K + k];
        u += input[k] * decodedUp[n * K + k];
      }
      const activated = activation === 'silu' ? g / (1 + Math.exp(-g))
        : 0.5 * g * (1 + Math.tanh(Math.max(-15, Math.min(15, 0.7978845608 * (g + 0.044715 * g ** 3)))));
      let expected = activated * u * 0.75;
      if (clampMax != null && activation === 'silu') expected = Math.max(-clampMax, Math.min(clampMax, expected));
      assert.ok(Math.abs(actual[n] - expected) <= 0.002, `${K}/${N}/${activation}/${clampMax} row ${n}: ${actual[n]} != ${expected}`);
    }
  } finally {
    if (result) releaseBuffer(result.buffer);
    for (const buffer of buffers) releaseBuffer(buffer);
  }
}
try {
  for (const K of [288, 512, 1024, 3584]) {
    for (const [activation, clamp] of [['silu', null], ['silu', 0.05], ['gelu', null]]) {
      await check(K, 37, activation, clamp);
    }
  }
  console.log('fused-ffn-subgroup8-gpu-regression: passed 12 CPU-reference cases');
} finally { await destroyDevice(); }
