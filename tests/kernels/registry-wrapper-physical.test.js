import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { readBufferSlice, releaseBuffer, destroyBufferPool, getBufferPool } from '../../src/memory/buffer-pool.js';
import { runMatmul } from '../../src/gpu/kernels/matmul.js';
import { runRMSNorm } from '../../src/gpu/kernels/rmsnorm.js';
import { runLayerNorm } from '../../src/gpu/kernels/layernorm.js';
import { runSandwichRMSNormPair } from '../../src/gpu/kernels/rmsnorm-pair.js';
import { runScale } from '../../src/gpu/kernels/scale.js';
import { runSplitQKV } from '../../src/gpu/kernels/split_qkv.js';
import { rmsNormRef } from './reference/rmsnorm.js';
import { layerNormRef } from './reference/layernorm.js';
import { f32ToF16Array, f16ToF32Bits } from '../../src/inference/kv-cache/types.js';

const probe = await probeNodeGPU();
if (!probe.ready) throw new Error(`Registry wrapper verification requires WebGPU: ${probe.reason}`);
const device = getDevice();
const caps = getKernelCapabilities();
assert.doesNotMatch(JSON.stringify(caps.adapterInfo), /swiftshader|llvmpipe|software/i);
const inputs = [];
const outputs = new Set();
let cases = 0;
function upload(data) {
  const buffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  inputs.push(buffer); device.queue.writeBuffer(buffer, 0, data); return buffer;
}
function tensor(data, shape) { return createTensor(upload(data), 'f32', shape, 'registry-test'); }
function retain(output) { outputs.add(output.buffer); return output; }
async function check(output, expected, label, tolerance = 2e-5) {
  retain(output);
  const actual = new Float32Array(await readBufferSlice(output.buffer, 0, expected.length * 4));
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) <= tolerance * Math.max(1, Math.abs(expected[i])), `${label}[${i}]: ${actual[i]} != ${expected[i]}`);
  }
  cases++;
}

try {
  for (const hiddenSize of [17, 257]) {
    const batchSize = 3, eps = 1e-5;
    const data = Float32Array.from({ length: hiddenSize * batchSize }, (_, i) => Math.sin(i * 0.17));
    const weights = Float32Array.from({ length: hiddenSize }, (_, i) => 0.5 + (i % 7) / 10);
    const bias = Float32Array.from({ length: hiddenSize }, (_, i) => (i % 3) / 10);
    const input = tensor(data, [batchSize, hiddenSize]);
    const weight = upload(weights), biasBuffer = upload(bias);
    const options = { batchSize, hiddenSize };
    await check(await runRMSNorm(input, weight, eps, options), rmsNormRef(data, weights, batchSize, hiddenSize, eps), `rmsnorm-${hiddenSize}`);
    await check(await runLayerNorm(input, weight, biasBuffer, eps, { ...options, normWeightDtype: 'f32' }), layerNormRef(data, weights, bias, batchSize, hiddenSize, eps), `layernorm-${hiddenSize}`);
    await check(await runScale(input, 0.25, { count: data.length }), Float32Array.from(data, (value) => value * 0.25), `scale-${hiddenSize}`);
    const pair = await runSandwichRMSNormPair(input, null, weight, weight, eps, options);
    retain(pair.postAttn); retain(pair.ffnInput);
    const post = rmsNormRef(data, weights, batchSize, hiddenSize, eps);
    await check(pair.postAttn, post, `pair-post-${hiddenSize}`);
    await check(pair.ffnInput, rmsNormRef(post, weights, batchSize, hiddenSize, eps), `pair-pre-${hiddenSize}`);
  }
  const qkvData = Float32Array.from({ length: 3 * 12 }, (_, i) => i / 4);
  const split = await runSplitQKV(tensor(qkvData, [3, 12]), { numTokens: 3, qSize: 6, kSize: 3, vSize: 3 });
  for (const output of Object.values(split)) retain(output);
  for (const [name, offset, width] of [['Q', 0, 6], ['K', 6, 3], ['V', 9, 3]]) {
    const expected = Float32Array.from({ length: 3 * width }, (_, i) => qkvData[Math.floor(i / width) * 12 + offset + i % width]);
    await check(split[name], expected, `split-${name}`);
  }
  for (const M of [1, 3]) for (const dtype of ['f32', 'f16']) {
    const N = 17, K = 32;
    const a = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 0.13));
    const original = Float32Array.from({ length: N * K }, (_, i) => Math.cos(i * 0.23));
    const encoded = dtype === 'f16' ? f32ToF16Array(original) : original;
    const b = dtype === 'f16' ? Float32Array.from(encoded, f16ToF32Bits) : original;
    const weights = createWeightBuffer(upload(encoded), dtype, 'row', [N, K], 'registry-matmul');
    const expected = new Float32Array(M * N);
    for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K; k++) sum += a[m * K + k] * b[n * K + k];
      expected[m * N + n] = sum;
    }
    const output = await runMatmul(tensor(a, [M, K]), weights, M, N, K, { alpha: 1, transposeB: true, outputDtype: 'f32' });
    await check(output, expected, `matmul-${M}-${dtype}`, 1e-4);
  }
  await device.queue.onSubmittedWorkDone();
  console.log(JSON.stringify({ test: 'registry-wrapper-physical', passed: true, cases, adapter: caps.adapterInfo,
    evidence: 'Actual wrapper dispatch and CPU-reference operator parity; not model qualification or cross-hardware coverage.' }));
} finally {
  for (const buffer of outputs) releaseBuffer(buffer);
  for (const buffer of inputs) buffer.destroy();
  assert.equal(getBufferPool().getStats().activeBuffers, 0);
  destroyBufferPool(); destroyDevice();
}
