import assert from 'node:assert/strict';
import { assertSubgroupPartition } from '../../src/gpu/kernels/subgroup-partition.js';

import {
  dequantizeQ4KMRowWise,
  quantizeToQ4KMRowWise,
} from '../../src/converter/quantizer.js';
import { destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { runMatmul } from '../../src/gpu/kernels/matmul.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';

const gpuProbe = await probeNodeGPU();
if (!gpuProbe.ready) {
  console.log(`q4-subgroup-gpu-regression.test: skipped (${gpuProbe.reason})`);
  process.exit(0);
}

const capabilities = getKernelCapabilities();
if (capabilities.hasSubgroups !== true) {
  console.log('q4-subgroup-gpu-regression.test: skipped (requires subgroups)');
  await destroyDevice();
  process.exit(0);
}

function createGpuBuffer(values, label) {
  const buffer = acquireBuffer(values.byteLength, undefined, label);
  uploadData(buffer, values);
  return buffer;
}

function createKernelPath(id, kernel, entry, constants) {
  return {
    id,
    name: id,
    activationDtype: 'f32',
    decode: {
      steps: [
        {
          op: 'q_proj',
          kernel,
          entry,
          constants,
        },
      ],
    },
  };
}

const baselinePath = createKernelPath(
  'q4-metal-baseline',
  'fused_matmul_q4.wgsl',
  'main_gemv',
  {
    WORKGROUP_SIZE: 256,
    COLS_PER_WG: 64,
    THREADS_PER_COL_GEMV: 4,
    SHARED_A_MAX: 1,
    USE_FULL_BLOCK_FAST_PATH: true,
  }
);

async function runCase(K, lanes, M = 1) {
  const N = 37;
  const inputValues = new Float32Array(M * K);
  const weightValues = new Float32Array(N * K);
  for (let k = 0; k < M * K; k++) {
    inputValues[k] = Math.sin(k * 0.037) * 0.25 + Math.cos(k * 0.011) * 0.125;
  }
  for (let index = 0; index < weightValues.length; index++) {
    weightValues[index] = Math.sin(index * 0.013) * 0.4 + Math.cos(index * 0.007) * 0.2;
  }

  const { quantized } = quantizeToQ4KMRowWise(weightValues, [N, K]);
  const dequantized = dequantizeQ4KMRowWise(quantized, [N, K]);
  const inputBuffer = createGpuBuffer(inputValues, `q4_simd16_input_${K}`);
  const weightBuffer = createGpuBuffer(quantized, `q4_simd16_weights_${K}`);
  const input = createTensor(inputBuffer, 'f32', [M, K], `q4_simd16_input_${K}`);
  const weights = createWeightBuffer(
    weightBuffer,
    'q4k',
    'row',
    [N, K],
    `q4_simd16_weights_${K}`
  );
  const fullBlock = K % 256 === 0;
  const candidatePath = createKernelPath(
    `q4-subgroup-${lanes}-${K}`,
    'fused_matmul_q4_subgroup.wgsl',
    'main',
    {
      WORKGROUP_SIZE: 256,
      COLS_PER_WG: 256 / lanes,
      THREADS_PER_COL_GEMV: lanes,
      USE_FULL_BLOCK_FAST_PATH: fullBlock,
    }
  );

  let baseline = null;
  let candidate = null;
  try {
    if (M === 1) baseline = await runMatmul(input, weights, 1, N, K, {
      role: 'q_proj',
      layerIdx: 0,
      outputDtype: 'f32',
      kernelPath: fullBlock
        ? baselinePath
        : createKernelPath(
            'q4-metal-baseline-tail',
            'fused_matmul_q4.wgsl',
            'main_gemv',
            {
              WORKGROUP_SIZE: 256,
              COLS_PER_WG: 64,
              THREADS_PER_COL_GEMV: 4,
              SHARED_A_MAX: 1,
              USE_FULL_BLOCK_FAST_PATH: false,
            }
          ),
    });
    candidate = await runMatmul(input, weights, M, N, K, {
      role: 'q_proj',
      layerIdx: 0,
      outputDtype: 'f32',
      kernelPath: candidatePath,
    });

    const baselineValues = baseline ? new Float32Array(await readBuffer(baseline.buffer, N * 4)) : null;
    const candidateValues = new Float32Array(await readBuffer(candidate.buffer, M * N * 4));
    for (let row = 0; row < M * N; row++) {
      let expected = 0;
      const weightOffset = (row % N) * K;
      const inputOffset = Math.floor(row / N) * K;
      for (let k = 0; k < K; k++) {
        expected += inputValues[inputOffset + k] * dequantized[weightOffset + k];
      }
      assert.equal(Number.isFinite(candidateValues[row]), true, `K=${K} row=${row} is not finite`);
      assert.ok(
        Math.abs(candidateValues[row] - expected) < 0.002,
        `K=${K} row=${row} CPU mismatch: actual=${candidateValues[row]} expected=${expected}`
      );
      if (baselineValues) assert.ok(
        Math.abs(candidateValues[row] - baselineValues[row]) < 0.002,
        `K=${K} row=${row} baseline mismatch: candidate=${candidateValues[row]} baseline=${baselineValues[row]}`
      );
    }
  } finally {
    for (const buffer of [inputBuffer, weightBuffer, baseline?.buffer, candidate?.buffer]) {
      if (buffer) releaseBuffer(buffer);
    }
  }
}

try {
  for (const lanes of [1, 2, 4, 8, 16, 32]) {
    try { assertSubgroupPartition(lanes, capabilities.adapterInfo); }
    catch (error) { console.log('q4-subgroup-gpu-regression.test: skipped lanes=' + lanes + ' (' + error.message + ')'); continue; }
    await runCase(512, lanes);
    await runCase(300, lanes);
    await runCase(1024, lanes);
    await runCase(3584, lanes);
    await runCase(300, lanes, 3);
    await runCase(1024, lanes, 3);
  }
  console.log('q4-subgroup-gpu-regression.test: ok');
} finally {
  await destroyDevice();
}
