import assert from 'node:assert/strict';

import { destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { runGather } from '../../src/gpu/kernels/gather.js';
import { runMatmul } from '../../src/gpu/kernels/matmul.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';

const gpuProbe = await probeNodeGPU();
if (!gpuProbe.ready) {
  console.log(`bf16-weight-kernels-gpu-regression.test: skipped (${gpuProbe.reason})`);
  process.exit(0);
}

function f32ToBF16(values) {
  const source = new Float32Array(1);
  const bits = new Uint32Array(source.buffer);
  const output = new Uint16Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    source[0] = values[index];
    output[index] = bits[0] >>> 16;
  }
  return output;
}

function bf16ToF32(bits) {
  const raw = new Uint32Array(1);
  const value = new Float32Array(raw.buffer);
  raw[0] = bits << 16;
  return value[0];
}

function createGpuBuffer(values, label) {
  const buffer = acquireBuffer(values.byteLength, undefined, label);
  uploadData(buffer, values);
  return buffer;
}

function assertClose(actual, expected, label, tolerance = 1e-5) {
  assert.equal(actual.length, expected.length, `${label} length`);
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(
      Math.abs(actual[index] - expected[index]) <= tolerance,
      `${label}[${index}]: actual=${actual[index]} expected=${expected[index]}`
    );
  }
}

const owned = [];
try {
  const vocabSize = 4;
  const hiddenSize = 8;
  const embeddingSource = Float32Array.from(
    { length: vocabSize * hiddenSize },
    (_, index) => Math.sin(index * 0.19) * 0.4 + Math.cos(index * 0.07) * 0.2
  );
  const embeddingBF16 = f32ToBF16(embeddingSource);
  const indices = new Uint32Array([2, 0]);
  const indicesBuffer = createGpuBuffer(indices, 'bf16_gather_indices');
  const embeddingBuffer = createGpuBuffer(embeddingBF16, 'bf16_gather_weights');
  owned.push(indicesBuffer, embeddingBuffer);

  const gathered = await runGather(
    indicesBuffer,
    embeddingBuffer,
    indices.length,
    hiddenSize,
    vocabSize,
    { embeddingDtype: 'bf16', outputDtype: 'f32' }
  );
  owned.push(gathered.buffer);
  const gatheredValues = new Float32Array(await readBuffer(gathered.buffer, indices.length * hiddenSize * 4));
  const expectedGather = new Float32Array(indices.length * hiddenSize);
  for (let row = 0; row < indices.length; row += 1) {
    for (let column = 0; column < hiddenSize; column += 1) {
      expectedGather[row * hiddenSize + column] = bf16ToF32(
        embeddingBF16[indices[row] * hiddenSize + column]
      );
    }
  }
  assertClose(gatheredValues, expectedGather, 'gather', 0);

  const M = 3;
  const N = 5;
  const K = 8;
  const inputValues = Float32Array.from(
    { length: M * K },
    (_, index) => Math.sin(index * 0.11) * 0.3 - Math.cos(index * 0.17) * 0.1
  );
  const weightSource = Float32Array.from(
    { length: N * K },
    (_, index) => Math.cos(index * 0.13) * 0.25 + Math.sin(index * 0.05) * 0.15
  );
  const weightBF16 = f32ToBF16(weightSource);
  const inputBuffer = createGpuBuffer(inputValues, 'bf16_matmul_input');
  const weightBuffer = createGpuBuffer(weightBF16, 'bf16_matmul_weights');
  owned.push(inputBuffer, weightBuffer);
  const input = createTensor(inputBuffer, 'f32', [M, K], 'bf16_matmul_input');
  const weights = createWeightBuffer(weightBuffer, 'bf16', 'row', [N, K], 'bf16_matmul_weights');
  const kernelPath = {
    id: 'bf16-weight-regression',
    prefill: {
      steps: [
        { op: 'q_proj', kernel: 'matmul_bf16w_f32a.wgsl', entry: 'main' },
      ],
    },
    decode: {
      steps: [
        { op: 'q_proj', kernel: 'matmul_gemv_subgroup_bf16w.wgsl', entry: 'main_vec4' },
      ],
    },
  };
  const expectedMatmul = new Float32Array(M * N);
  for (let row = 0; row < M; row += 1) {
    for (let column = 0; column < N; column += 1) {
      let sum = 0;
      for (let inner = 0; inner < K; inner += 1) {
        sum += inputValues[row * K + inner] * bf16ToF32(weightBF16[column * K + inner]);
      }
      expectedMatmul[row * N + column] = sum;
    }
  }

  const prefill = await runMatmul(input, weights, M, N, K, {
    role: 'q_proj',
    layerIdx: 0,
    phaseOverride: 'prefill',
    outputDtype: 'f32',
    kernelPath,
  });
  owned.push(prefill.buffer);
  const prefillValues = new Float32Array(await readBuffer(prefill.buffer, M * N * 4));
  assertClose(prefillValues, expectedMatmul, 'prefill matmul');

  if (getKernelCapabilities().hasSubgroups === true) {
    const decodeInput = createTensor(inputBuffer, 'f32', [1, K], 'bf16_gemv_input');
    const decode = await runMatmul(decodeInput, weights, 1, N, K, {
      role: 'q_proj',
      layerIdx: 0,
      phaseOverride: 'decode',
      outputDtype: 'f32',
      kernelPath,
    });
    owned.push(decode.buffer);
    const decodeValues = new Float32Array(await readBuffer(decode.buffer, N * 4));
    assertClose(decodeValues, expectedMatmul.subarray(0, N), 'decode matmul');
  }

  console.log('bf16-weight-kernels-gpu-regression.test: ok');
} finally {
  for (const buffer of new Set(owned)) {
    releaseBuffer(buffer);
  }
  await destroyDevice();
}
