import assert from 'node:assert/strict';

const { probeNodeGPU } = await import('../helpers/gpu-probe.js');
const { acquireBuffer, uploadData, readBuffer, releaseBuffer } = await import('../../src/memory/buffer-pool.js');
const { createTensor } = await import('../../src/gpu/tensor.js');
const { destroyDevice } = await import('../../src/gpu/device.js');
const { createWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
const { runRMSNorm } = await import('../../src/gpu/kernels/rmsnorm.js');
const { runRMSNormStats } = await import('../../src/gpu/kernels/rmsnorm-stats.js');
const { runFusedFFN, runFusedFFNFromRMSNormStats } = await import('../../src/gpu/kernels/fused_ffn.js');
const { float32ToFloat16 } = await import('../../src/converter/quantizer.js');

const Q4K_BLOCK_SIZE = 144;
const Q4K_WORD_COUNT = 32;

function createGpuBuffer(values, label) {
  const buffer = acquireBuffer(values.byteLength, undefined, label);
  uploadData(buffer, values);
  return buffer;
}

function createUnitQ4KWeight(intermediateSize, label) {
  const data = new ArrayBuffer(intermediateSize * Q4K_BLOCK_SIZE);
  const view = new DataView(data);
  const d = float32ToFloat16(1);
  const dmin = float32ToFloat16(0);
  const dDmin = d | (dmin << 16);
  for (let col = 0; col < intermediateSize; col += 1) {
    const base = col * Q4K_BLOCK_SIZE;
    view.setUint32(base, dDmin, true);
    view.setUint32(base + 4, 0x01010101, true);
    view.setUint32(base + 8, 0, true);
    view.setUint32(base + 12, 0x01010101, true);
    for (let word = 0; word < Q4K_WORD_COUNT; word += 1) {
      view.setUint32(base + 16 + word * 4, 0x11111111, true);
    }
  }
  const buffer = acquireBuffer(data.byteLength, undefined, label);
  uploadData(buffer, new Uint8Array(data));
  return createWeightBuffer(buffer, 'q4k', 'row', [intermediateSize, 256], label);
}

function assertClose(actual, expected, tolerance, label) {
  assert.equal(actual.length, expected.length, `${label} length mismatch`);
  for (let i = 0; i < actual.length; i += 1) {
    const delta = Math.abs(actual[i] - expected[i]);
    assert.ok(
      delta <= tolerance,
      `${label}[${i}] mismatch actual=${actual[i]} expected=${expected[i]} delta=${delta}`
    );
  }
}

const gpuProbe = await probeNodeGPU();
if (!gpuProbe.ready) {
  console.log(`fused-normed-ffn-gpu-regression.test: skipped (${gpuProbe.reason})`);
  process.exit(0);
}

const hiddenSize = 256;
const intermediateSize = 32;
const eps = 1e-6;
const postAttnValues = new Float32Array(hiddenSize);
const residualValues = new Float32Array(hiddenSize);
for (let i = 0; i < hiddenSize; i += 1) {
  postAttnValues[i] = ((i % 17) - 8) / 500;
  residualValues[i] = ((i % 13) - 6) / 700;
}
const normWeights = new Float32Array(hiddenSize);

const postAttnBuffer = createGpuBuffer(postAttnValues, 'normed_ffn_post_attn');
const residualBuffer = createGpuBuffer(residualValues, 'normed_ffn_residual');
const normWeightBuffer = createGpuBuffer(normWeights, 'normed_ffn_norm_weight');
const normWeight = createWeightBuffer(normWeightBuffer, 'f32', 'row', [hiddenSize], 'normed_ffn_norm_weight');
const gateWeight = createUnitQ4KWeight(intermediateSize, 'normed_ffn_gate_q4k');
const upWeight = createUnitQ4KWeight(intermediateSize, 'normed_ffn_up_q4k');
const residualSumBuffer = acquireBuffer(hiddenSize * 4, undefined, 'normed_ffn_residual_sum');

let baseline = null;
let fused = null;
let stats = null;
let normed = null;
try {
  const postAttn = createTensor(postAttnBuffer, 'f32', [1, hiddenSize], 'post_attn');
  const residual = createTensor(residualBuffer, 'f32', [1, hiddenSize], 'residual');
  normed = await runRMSNorm(postAttn, normWeight, eps, {
    batchSize: 1,
    hiddenSize,
    preResidual: residual,
    residualSumOutput: residualSumBuffer,
    rmsNormWeightOffset: true,
  });
  baseline = await runFusedFFN(normed, gateWeight, upWeight, hiddenSize, intermediateSize, {
    batchSize: 1,
    activation: 'silu',
    swigluLimit: null,
  });
  stats = await runRMSNormStats(postAttn, residual, eps, {
    batchSize: 1,
    hiddenSize,
  });
  fused = await runFusedFFNFromRMSNormStats(
    stats.prenormSum,
    stats.invRmsBuffer,
    normWeight,
    gateWeight,
    upWeight,
    hiddenSize,
    intermediateSize,
    {
      batchSize: 1,
      activation: 'silu',
      swigluLimit: null,
      rmsNormWeightOffset: true,
    }
  );

  const baselineValues = new Float32Array(await readBuffer(baseline.buffer, intermediateSize * 4));
  const fusedValues = new Float32Array(await readBuffer(fused.buffer, intermediateSize * 4));
  assertClose(fusedValues, baselineValues, 1e-4, 'fused normed FFN');
  console.log('fused-normed-ffn-gpu-regression.test: ok');
} finally {
  for (const buffer of [
    postAttnBuffer,
    residualBuffer,
    normWeightBuffer,
    gateWeight.buffer,
    upWeight.buffer,
    residualSumBuffer,
    baseline?.buffer,
    fused?.buffer,
    normed?.buffer,
    stats?.prenormSum?.buffer,
    stats?.invRmsBuffer,
  ]) {
    if (buffer) releaseBuffer(buffer);
  }
  await destroyDevice();
}
