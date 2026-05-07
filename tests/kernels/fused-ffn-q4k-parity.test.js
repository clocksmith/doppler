import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { destroyDevice, getDevice } from '../../src/gpu/device.js';
import { quantizeQ4_KRef } from './reference/dequant.js';
import { fusedFfnQ4KRef } from './reference/fused-ffn-q4k.js';

const rootDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const kernelSrc = readFileSync(join(rootDir, '../src/gpu/kernels/fused_ffn_q4k.wgsl'), 'utf8');

const M = 1;
const K = 1536;
const N = 6144;
const numBlocksPerRow = Math.ceil(K / 256);
const blockBytes = 144;

const gpu = await probeNodeGPU();
if (!gpu.ready) {
  console.log(`fused-ffn-q4k-parity.test: skipped (${gpu.reason})`);
  process.exit(0);
}

const device = getDevice();
const adapter = device.adapterInfo || {};
console.log(`fused-ffn-q4k-parity: vendor=${adapter.vendor || 'unknown'} arch=${adapter.architecture || 'unknown'}`);

function makePrng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

const rng = makePrng(42);
const input = new Float32Array(M * K);
for (let i = 0; i < input.length; i++) input[i] = (rng() - 0.5) * 0.5;

const gateF32 = new Float32Array(N * K);
const upF32 = new Float32Array(N * K);
for (let i = 0; i < gateF32.length; i++) gateF32[i] = (rng() - 0.5) * 0.1;
for (let i = 0; i < upF32.length; i++) upF32[i] = (rng() - 0.5) * 0.1;

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

const Wgate = packRows(gateF32, N, K);
const Wup = packRows(upF32, N, K);

const expected = fusedFfnQ4KRef({
  input,
  Wgate,
  Wup,
  M,
  hiddenSize: K,
  intermediateSize: N,
  alpha: 1.0,
  activation: 'silu',
  swigluLimit: null,
});

const module_ = device.createShaderModule({ code: kernelSrc, label: 'fused_ffn_q4k_parity' });
const pipeline = await device.createComputePipelineAsync({
  label: 'fused_ffn_q4k_parity_pipeline',
  layout: 'auto',
  compute: { module: module_, entryPoint: 'main' },
});

function makeStorageBuffer(data, label) {
  const size = Math.ceil(data.byteLength / 4) * 4;
  const buf = device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buf, 0, data);
  return buf;
}

const inputBuf = makeStorageBuffer(input, 'parity_input');
const gateBuf = makeStorageBuffer(Wgate, 'parity_gate_q4k');
const upBuf = makeStorageBuffer(Wup, 'parity_up_q4k');

const outputBuf = device.createBuffer({
  label: 'parity_output',
  size: M * N * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
device.queue.writeBuffer(outputBuf, 0, new Float32Array(M * N));

const uniformBytes = new ArrayBuffer(32);
const dv = new DataView(uniformBytes);
dv.setUint32(0, M, true);
dv.setUint32(4, K, true);
dv.setUint32(8, N, true);
dv.setFloat32(12, 1.0, true);
dv.setUint32(16, 0, true);
dv.setUint32(20, numBlocksPerRow, true);
dv.setFloat32(24, 0, true);
dv.setUint32(28, 0, true);

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
    { binding: 2, resource: { buffer: gateBuf } },
    { binding: 3, resource: { buffer: upBuf } },
    { binding: 4, resource: { buffer: outputBuf } },
  ],
});

const stagingBuf = device.createBuffer({
  label: 'parity_staging',
  size: M * N * 4,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const gateProbeStaging = device.createBuffer({
  label: 'parity_gate_probe_staging',
  size: 256,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const encoder = device.createCommandEncoder({ label: 'parity_encoder' });
const pass = encoder.beginComputePass({ label: 'parity_pass' });
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(Math.ceil(N / 32), 1, 1);
pass.end();
encoder.copyBufferToBuffer(outputBuf, 0, stagingBuf, 0, M * N * 4);
encoder.copyBufferToBuffer(gateBuf, 0, gateProbeStaging, 0, 256);
device.queue.submit([encoder.finish()]);

await stagingBuf.mapAsync(GPUMapMode.READ);
const actual = new Float32Array(stagingBuf.getMappedRange().slice(0));
stagingBuf.unmap();

await gateProbeStaging.mapAsync(GPUMapMode.READ);
const gateProbeAfter = new Uint8Array(gateProbeStaging.getMappedRange().slice(0));
gateProbeStaging.unmap();

let firstDiff = -1;
let maxAbsDiff = 0;
let nanCount = 0;
let infCount = 0;
for (let i = 0; i < expected.length; i++) {
  if (Number.isNaN(actual[i])) { nanCount++; if (firstDiff < 0) firstDiff = i; continue; }
  if (!Number.isFinite(actual[i])) { infCount++; if (firstDiff < 0) firstDiff = i; continue; }
  const d = Math.abs(actual[i] - expected[i]);
  if (d > maxAbsDiff) maxAbsDiff = d;
  if (d > 5e-2 && firstDiff < 0) firstDiff = i;
}

let gateModified = 0;
for (let i = 0; i < 256; i++) {
  if (gateProbeAfter[i] !== Wgate[i]) gateModified++;
}

console.log(`fused-ffn-q4k-parity:`);
console.log(`  shape M=${M} K=${K} N=${N} numBlocksPerRow=${numBlocksPerRow}`);
console.log(`  max_abs_diff=${maxAbsDiff.toExponential(3)} nan_count=${nanCount} inf_count=${infCount} first_diff_idx=${firstDiff}`);
console.log(`  actual[0..6]=${Array.from(actual.subarray(0, 6)).map(x => x.toFixed(5))}`);
console.log(`  expected[0..6]=${Array.from(expected.subarray(0, 6)).map(x => x.toFixed(5))}`);
console.log(`  gate_buffer_modified_bytes=${gateModified} (of 256 probed)`);

inputBuf.destroy();
gateBuf.destroy();
upBuf.destroy();
outputBuf.destroy();
uniformBuf.destroy();
stagingBuf.destroy();
gateProbeStaging.destroy();
destroyDevice();

assert.equal(nanCount, 0, `fused_ffn_q4k produced ${nanCount} NaN outputs at first idx=${firstDiff}`);
assert.equal(infCount, 0, `fused_ffn_q4k produced ${infCount} Inf outputs`);
assert.equal(gateModified, 0, `gate buffer (read-only binding) was modified by ${gateModified} bytes — kernel writes outside its output binding`);
assert.ok(maxAbsDiff < 5e-2, `fused_ffn_q4k diverges from CPU reference: max_abs_diff=${maxAbsDiff}`);

console.log('fused-ffn-q4k-parity.test: ok');
