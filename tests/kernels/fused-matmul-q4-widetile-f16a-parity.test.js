import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { destroyDevice, getDevice } from '../../src/gpu/device.js';
import { quantizeQ4_KRef } from './reference/dequant.js';
import { dequantQ4KRowsRef } from './reference/fused-ffn-q4k.js';
import { f32ToF16Array, f16ToF32Bits } from '../../src/inference/kv-cache/types.js';

const rootDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const kernelSrc = readFileSync(join(rootDir, '../src/gpu/kernels/fused_matmul_q4_widetile_f16a.wgsl'), 'utf8');

const M = 4;
const K = 1536;
const N = 6144;
const numBlocksPerRow = Math.ceil(K / 256);
const blockBytes = 144;

const gpu = await probeNodeGPU();
if (!gpu.ready) {
  console.log(`fused-matmul-q4-widetile-f16a-parity.test: skipped (${gpu.reason})`);
  process.exit(0);
}

const device = getDevice();

function makePrng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

const rng = makePrng(7);

const inputF32 = new Float32Array(M * K);
for (let i = 0; i < inputF32.length; i++) inputF32[i] = (rng() - 0.5) * 1.0;
const inputF16 = f32ToF16Array(inputF32);

const wF32 = new Float32Array(N * K);
for (let i = 0; i < wF32.length; i++) wF32[i] = (rng() - 0.5) * 0.1;

function packRows(weights, numRows, kPerRow) {
  const numBlocks = Math.ceil(kPerRow / 256);
  const packed = new Uint8Array(numRows * numBlocks * blockBytes);
  for (let row = 0; row < numRows; row++) {
    const rowSlice = weights.subarray(row * kPerRow, (row + 1) * kPerRow);
    const rowPacked = quantizeQ4_KRef(rowSlice, numBlocks);
    packed.set(rowPacked, row * numBlocks * blockBytes);
  }
  return packed;
}

const Wq = packRows(wF32, N, K);

const wDeq = dequantQ4KRowsRef(Wq, N, K);
const expectedF32 = new Float32Array(M * N);
const inputF32FromF16 = new Float32Array(M * K);
for (let i = 0; i < inputF32FromF16.length; i++) inputF32FromF16[i] = f16ToF32Bits(inputF16[i]);
const alpha = 1.0;
for (let m = 0; m < M; m++) {
  for (let n = 0; n < N; n++) {
    let s = 0;
    const rowBase = n * K;
    const inBase = m * K;
    for (let k = 0; k < K; k++) {
      s += inputF32FromF16[inBase + k] * wDeq[rowBase + k];
    }
    expectedF32[m * N + n] = s * alpha;
  }
}

const module_ = device.createShaderModule({ code: kernelSrc, label: 'widetile_f16a_parity' });
const pipeline = await device.createComputePipelineAsync({
  label: 'widetile_f16a_parity_pipeline',
  layout: 'auto',
  compute: { module: module_, entryPoint: 'main' },
});

function makeStorageBuffer(data, label, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
  const size = Math.ceil(data.byteLength / 4) * 4;
  const buf = device.createBuffer({ label, size, usage });
  device.queue.writeBuffer(buf, 0, data);
  return buf;
}

const inputBuf = makeStorageBuffer(inputF16, 'parity_input_f16');
const wBuf = makeStorageBuffer(Wq, 'parity_w_q4k');

const outputBuf = device.createBuffer({
  label: 'parity_output_f16',
  size: M * N * 2,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
device.queue.writeBuffer(outputBuf, 0, new Uint16Array(M * N));

const uniformBytes = new ArrayBuffer(32);
const dv = new DataView(uniformBytes);
dv.setUint32(0, M, true);
dv.setUint32(4, N, true);
dv.setUint32(8, K, true);
dv.setFloat32(12, alpha, true);
dv.setUint32(16, numBlocksPerRow, true);

const uniformBuf = device.createBuffer({
  label: 'parity_uniforms',
  size: 32,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uniformBuf, 0, new Uint8Array(uniformBytes));

const bindGroup = device.createBindGroup({
  label: 'parity_bind_group',
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: uniformBuf } },
    { binding: 1, resource: { buffer: inputBuf } },
    { binding: 2, resource: { buffer: wBuf } },
    { binding: 4, resource: { buffer: outputBuf } },
  ],
});

const TILE_N = 256;
const TILE_M = 4;
const wgX = Math.ceil(N / TILE_N);
const wgY = Math.ceil(M / TILE_M);

const encoder = device.createCommandEncoder({ label: 'parity_encoder' });
const pass = encoder.beginComputePass({ label: 'parity_pass' });
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(wgX, wgY, 1);
pass.end();

const stagingBuf = device.createBuffer({
  label: 'parity_staging',
  size: M * N * 2,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});
encoder.copyBufferToBuffer(outputBuf, 0, stagingBuf, 0, M * N * 2);
device.queue.submit([encoder.finish()]);

await stagingBuf.mapAsync(GPUMapMode.READ);
const actualU16 = new Uint16Array(stagingBuf.getMappedRange().slice(0));
stagingBuf.unmap();

const actualF32 = new Float32Array(M * N);
for (let i = 0; i < actualF32.length; i++) actualF32[i] = f16ToF32Bits(actualU16[i]);

let maxAbsDiff = 0;
let maxRelDiff = 0;
let nanCount = 0;
let infCount = 0;
let firstDiff = -1;
let firstNan = -1;
for (let i = 0; i < expectedF32.length; i++) {
  if (Number.isNaN(actualF32[i])) {
    nanCount++;
    if (firstNan < 0) firstNan = i;
    continue;
  }
  if (!Number.isFinite(actualF32[i])) {
    infCount++;
    if (firstDiff < 0) firstDiff = i;
    continue;
  }
  const d = Math.abs(actualF32[i] - expectedF32[i]);
  if (d > maxAbsDiff) maxAbsDiff = d;
  const denom = Math.max(Math.abs(expectedF32[i]), 1e-3);
  const rd = d / denom;
  if (rd > maxRelDiff) maxRelDiff = rd;
  if (d > 0.5 && firstDiff < 0) firstDiff = i;
}

console.log(`fused-matmul-q4-widetile-f16a-parity:`);
console.log(`  shape M=${M} K=${K} N=${N} TILE_M=${TILE_M} TILE_N=${TILE_N}`);
console.log(`  max_abs_diff=${maxAbsDiff.toExponential(3)} max_rel_diff=${maxRelDiff.toExponential(3)}`);
console.log(`  nan_count=${nanCount} inf_count=${infCount} first_nan_idx=${firstNan} first_diff_idx=${firstDiff}`);
console.log(`  actual[0..6]=${Array.from(actualF32.subarray(0, 6)).map(x => x.toFixed(4))}`);
console.log(`  expected[0..6]=${Array.from(expectedF32.subarray(0, 6)).map(x => x.toFixed(4))}`);

inputBuf.destroy();
wBuf.destroy();
outputBuf.destroy();
uniformBuf.destroy();
stagingBuf.destroy();
destroyDevice();

assert.equal(nanCount, 0, `widetile_f16a produced ${nanCount} NaN outputs at first idx=${firstNan}`);
assert.equal(infCount, 0, `widetile_f16a produced ${infCount} Inf outputs at first idx=${firstDiff}`);
assert.ok(maxAbsDiff < 1.0, `widetile_f16a max_abs_diff=${maxAbsDiff} exceeds f16 matmul tolerance`);

console.log('fused-matmul-q4-widetile-f16a-parity.test: ok');
