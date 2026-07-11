import assert from 'node:assert/strict';

import {
  dequantizeQ4KMRowWise,
  quantizeToQ4KMRowWise,
} from '../../src/converter/quantizer.js';
import { destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { runFusedFFN } from '../../src/gpu/kernels/fused_ffn.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';

const gpuProbe = await probeNodeGPU();
if (!gpuProbe.ready) {
  console.log(`fused-ffn-q4-metal-simd16-gpu-regression.test: skipped (${gpuProbe.reason})`);
  process.exit(0);
}

const capabilities = getKernelCapabilities();
if (capabilities?.adapterInfo?.vendor !== 'apple' || capabilities.hasSubgroups !== true) {
  console.log('fused-ffn-q4-metal-simd16-gpu-regression.test: skipped (requires Apple Metal subgroups)');
  await destroyDevice();
  process.exit(0);
}

function createGpuBuffer(values, label) {
  const buffer = acquireBuffer(values.byteLength, undefined, label);
  uploadData(buffer, values);
  return buffer;
}

async function runCase(hiddenSize) {
  const intermediateSize = 37;
  const inputValues = new Float32Array(hiddenSize);
  const gateValues = new Float32Array(intermediateSize * hiddenSize);
  const upValues = new Float32Array(intermediateSize * hiddenSize);
  for (let k = 0; k < hiddenSize; k++) {
    inputValues[k] = Math.sin(k * 0.037) * 0.1 + Math.cos(k * 0.011) * 0.05;
  }
  for (let index = 0; index < gateValues.length; index++) {
    gateValues[index] = Math.sin(index * 0.013) * 0.15 + Math.cos(index * 0.007) * 0.075;
    upValues[index] = Math.cos(index * 0.017) * 0.125 - Math.sin(index * 0.005) * 0.05;
  }

  const gateQuantized = quantizeToQ4KMRowWise(gateValues, [intermediateSize, hiddenSize]).quantized;
  const upQuantized = quantizeToQ4KMRowWise(upValues, [intermediateSize, hiddenSize]).quantized;
  const gateDequantized = dequantizeQ4KMRowWise(gateQuantized, [intermediateSize, hiddenSize]);
  const upDequantized = dequantizeQ4KMRowWise(upQuantized, [intermediateSize, hiddenSize]);
  const inputBuffer = createGpuBuffer(inputValues, `fused_ffn_simd16_input_${hiddenSize}`);
  const gateBuffer = createGpuBuffer(gateQuantized, `fused_ffn_simd16_gate_${hiddenSize}`);
  const upBuffer = createGpuBuffer(upQuantized, `fused_ffn_simd16_up_${hiddenSize}`);
  const input = createTensor(inputBuffer, 'f32', [1, hiddenSize], `fused_ffn_simd16_input_${hiddenSize}`);
  const gate = createWeightBuffer(
    gateBuffer,
    'q4k',
    'row',
    [intermediateSize, hiddenSize],
    `fused_ffn_simd16_gate_${hiddenSize}`
  );
  const up = createWeightBuffer(
    upBuffer,
    'q4k',
    'row',
    [intermediateSize, hiddenSize],
    `fused_ffn_simd16_up_${hiddenSize}`
  );
  const fullBlock = hiddenSize % 256 === 0;
  let baseline = null;
  let candidate = null;
  try {
    baseline = await runFusedFFN(input, gate, up, hiddenSize, intermediateSize, {
      batchSize: 1,
      activation: 'silu',
      swigluLimit: null,
      pipelineConstants: {
        WORKGROUP_SIZE: 256,
        COLS_PER_WG: 32,
        THREADS_PER_COL: 8,
        USE_FULL_BLOCK_FAST_PATH: fullBlock,
      },
    });
    candidate = await runFusedFFN(input, gate, up, hiddenSize, intermediateSize, {
      batchSize: 1,
      activation: 'silu',
      swigluLimit: null,
      variant: 'q4k_metal_simd16',
      pipelineConstants: {
        WORKGROUP_SIZE: 256,
        COLS_PER_WG: 16,
        THREADS_PER_COL: 16,
        USE_FULL_BLOCK_FAST_PATH: fullBlock,
      },
    });

    const baselineValues = new Float32Array(await readBuffer(baseline.buffer, intermediateSize * 4));
    const candidateValues = new Float32Array(await readBuffer(candidate.buffer, intermediateSize * 4));
    for (let row = 0; row < intermediateSize; row++) {
      let expectedGate = 0;
      let expectedUp = 0;
      const weightOffset = row * hiddenSize;
      for (let k = 0; k < hiddenSize; k++) {
        expectedGate += inputValues[k] * gateDequantized[weightOffset + k];
        expectedUp += inputValues[k] * upDequantized[weightOffset + k];
      }
      const expected = expectedGate / (1 + Math.exp(-expectedGate)) * expectedUp;
      assert.equal(Number.isFinite(candidateValues[row]), true, `K=${hiddenSize} row=${row} is not finite`);
      assert.ok(
        Math.abs(candidateValues[row] - expected) < 0.002,
        `K=${hiddenSize} row=${row} CPU mismatch: actual=${candidateValues[row]} expected=${expected}`
      );
      assert.ok(
        Math.abs(candidateValues[row] - baselineValues[row]) < 0.002,
        `K=${hiddenSize} row=${row} baseline mismatch: candidate=${candidateValues[row]} baseline=${baselineValues[row]}`
      );
    }
  } finally {
    for (const buffer of [inputBuffer, gateBuffer, upBuffer, baseline?.buffer, candidate?.buffer]) {
      if (buffer) releaseBuffer(buffer);
    }
  }
}

try {
  await runCase(512);
  await runCase(288);
  console.log('fused-ffn-q4-metal-simd16-gpu-regression.test: ok');
} finally {
  await destroyDevice();
}
