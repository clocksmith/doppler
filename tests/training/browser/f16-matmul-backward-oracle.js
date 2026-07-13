import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { getKernelCapabilities, initDevice } from '../../../src/gpu/device.js';
import { createTensor } from '../../../src/gpu/tensor.js';
import { runMatmulBackwardDx } from '../../../src/gpu/kernels/backward/utils.js';
import { f16ToF32Array, f32ToF16Array } from '../../../src/inference/kv-cache/types.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../../src/memory/buffer-pool.js';

function makeTensor(values, dtype, shape, label) {
  const byteLength = Math.ceil(values.byteLength / 4) * 4;
  const upload = byteLength === values.byteLength
    ? values
    : (() => {
        const padded = new Uint8Array(byteLength);
        padded.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
        return padded;
      })();
  const buffer = acquireBuffer(byteLength, undefined, label);
  uploadData(buffer, upload);
  return createTensor(buffer, dtype, shape, label);
}

function cpuInputGradient(gradOutput, weight, M, K, N, transposeB) {
  const output = new Float32Array(M * K);
  for (let row = 0; row < M; row += 1) {
    for (let col = 0; col < K; col += 1) {
      let sum = 0;
      for (let inner = 0; inner < N; inner += 1) {
        const weightIndex = transposeB
          ? (inner * K) + col
          : (col * N) + inner;
        sum += gradOutput[(row * N) + inner] * weight[weightIndex];
      }
      output[(row * K) + col] = sum;
    }
  }
  return output;
}

function compare(actual, expected) {
  let maxAbsError = 0;
  let squaredError = 0;
  let allFinite = true;
  for (let index = 0; index < expected.length; index += 1) {
    const error = Math.abs(actual[index] - expected[index]);
    maxAbsError = Math.max(maxAbsError, error);
    squaredError += error * error;
    allFinite = allFinite && Number.isFinite(actual[index]);
  }
  return {
    elementCount: expected.length,
    allFinite,
    maxAbsError,
    rmse: Math.sqrt(squaredError / expected.length),
  };
}

async function executeCase(gradOutput, weightValues, M, K, N, transposeB, label) {
  const packedWeight = f32ToF16Array(weightValues);
  const roundedWeight = f16ToF32Array(packedWeight);
  const gradTensor = makeTensor(gradOutput, 'f32', [M, N], `${label}_grad_output`);
  const weightTensor = makeTensor(
    packedWeight,
    'f16',
    transposeB ? [N, K] : [K, N],
    `${label}_weight`
  );
  let output = null;
  try {
    output = await runMatmulBackwardDx(gradTensor, weightTensor, M, K, N, { transposeB });
    const bytes = await readBuffer(output.buffer, M * K * Float32Array.BYTES_PER_ELEMENT);
    const actual = new Float32Array(bytes);
    const expected = cpuInputGradient(gradOutput, roundedWeight, M, K, N, transposeB);
    return { actual, expected, comparison: compare(actual, expected) };
  } finally {
    if (output?.buffer) releaseBuffer(output.buffer);
    releaseBuffer(gradTensor.buffer);
    releaseBuffer(weightTensor.buffer);
  }
}

export async function runF16MatmulBackwardOracle() {
  const baseUrl = new URL('../../../src/config/', import.meta.url);
  setPlatformsBaseUrl(new URL('platforms/', baseUrl).toString());
  setRegistryUrl(new URL('kernels/registry.json', baseUrl).toString());
  await initDevice();

  const M = 3;
  const K = 5;
  const N = 7;
  const tolerance = 2e-6;
  const gradOutput = new Float32Array(
    Array.from({ length: M * N }, (_, index) => Math.sin(index * 0.37) * 0.75)
  );
  const standardWeight = new Float32Array(
    Array.from({ length: K * N }, (_, index) => Math.cos(index * 0.23) * 0.5)
  );
  const transposedWeight = new Float32Array(N * K);
  for (let row = 0; row < K; row += 1) {
    for (let col = 0; col < N; col += 1) {
      transposedWeight[(col * K) + row] = standardWeight[(row * N) + col];
    }
  }

  const standard = await executeCase(
    gradOutput,
    standardWeight,
    M,
    K,
    N,
    false,
    'f16_backward_standard'
  );
  const transposed = await executeCase(
    gradOutput,
    transposedWeight,
    M,
    K,
    N,
    true,
    'f16_backward_transposed'
  );
  const perturbedWeight = new Float32Array(standardWeight);
  perturbedWeight[9] += 0.125;
  const perturbed = await executeCase(
    gradOutput,
    perturbedWeight,
    M,
    K,
    N,
    false,
    'f16_backward_perturbed'
  );
  const perturbation = compare(perturbed.actual, standard.actual);
  const capabilities = getKernelCapabilities();
  const passed = standard.comparison.allFinite
    && transposed.comparison.allFinite
    && standard.comparison.maxAbsError <= tolerance
    && transposed.comparison.maxAbsError <= tolerance
    && perturbation.maxAbsError > 1e-4;
  return {
    artifactType: 'f16_frozen_weight_matmul_backward_oracle',
    schemaVersion: 1,
    passed,
    dimensions: { M, K, N },
    tolerance: { maxAbsError: tolerance },
    standard: standard.comparison,
    transposed: transposed.comparison,
    negativeControl: {
      perturbation: 'weight_index_9_plus_0.125_before_f16_rounding',
      outputDifference: perturbation,
      passed: perturbation.maxAbsError > 1e-4,
    },
    adapterInfo: capabilities.adapterInfo || null,
    claimBoundary: 'Kernel-level F16 frozen-weight input-gradient mechanics only; not Qwen block, optimizer, adapter, or capability parity.',
  };
}
