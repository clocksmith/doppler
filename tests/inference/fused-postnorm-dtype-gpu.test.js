import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { destroyDevice, getDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, getBufferPool, readBuffer, releaseBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { float32ToFloat16 } from '../../src/formats/float16.js';
import { f16ToF32 } from '../../src/loader/dtype-utils.js';
import { runMatmulRMSNormFused } from '../../src/gpu/kernels/fused_matmul_rmsnorm.js';
import { runDenseFFNWithFusedPostNormGPU } from '../../src/inference/pipelines/text/ffn/dense-executor.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { quantizeQ4_KRef, dequantQ4_KRef } from '../kernels/reference/dequant.js';

const gpu = await probeNodeGPU({ installFileFetchShim: true });
if (!gpu.ready) { console.log(`fused-postnorm-dtype-gpu: skipped (${gpu.reason})`); process.exit(0); }
const H = 256, I = 256, eps = 1e-5;
const owned = [];
function buffer(values, label) {
  const b = acquireBuffer(values.byteLength, undefined, label); owned.push(b); uploadData(b, values); return b;
}
const half = value => f16ToF32(float32ToFloat16(value));
const input = Float32Array.from({ length: H }, (_, i) => half(Math.sin(i * 0.07) * 0.3));
const residual = Float32Array.from({ length: H }, (_, i) => half(Math.cos(i * 0.03) * 0.1));
const down = Float32Array.from({ length: H * I }, (_, i) => half(Math.sin(i * 0.017) * 0.02));
const gateBytes = quantizeQ4_KRef(Float32Array.from({ length: H * I }, (_, i) => Math.sin(i * 0.021) * 0.03), I);
const upBytes = quantizeQ4_KRef(Float32Array.from({ length: H * I }, (_, i) => Math.cos(i * 0.013) * 0.04), I);
const gate = dequantQ4_KRef(gateBytes, I), up = dequantQ4_KRef(upBytes, I);
const activated = new Float32Array(I), projected = new Float32Array(H);
for (let j = 0; j < I; j++) {
  let g = 0, u = 0;
  for (let k = 0; k < H; k++) { g += input[k] * gate[j * H + k]; u += input[k] * up[j * H + k]; }
  activated[j] = half(0.5 * g * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (g + 0.044715 * g ** 3))) * u);
}
for (let j = 0; j < H; j++) for (let k = 0; k < I; k++) projected[j] += activated[k] * down[j * I + k];
const inv = 1 / Math.sqrt(projected.reduce((s, x) => s + x * x, 0) / H + eps);
const expected = projected.map((x, i) => half(x * inv + residual[i]));
let result;
try {
  const inputTensor = createTensor(buffer(Uint16Array.from(input, float32ToFloat16), 'postnorm_input'), 'f16', [1, H]);
  const residualTensor = createTensor(buffer(Uint16Array.from(residual, float32ToFloat16), 'postnorm_residual'), 'f16', [1, H]);
  const weights = {
    gate: createWeightBuffer(buffer(gateBytes, 'postnorm_gate'), 'q4k', 'row', [I, H]),
    up: createWeightBuffer(buffer(upBytes, 'postnorm_up'), 'q4k', 'row', [I, H]),
    down: createWeightBuffer(buffer(Uint16Array.from(down, float32ToFloat16), 'postnorm_down'), 'f16', 'row', [H, I]),
    postFeedforwardNorm: createWeightBuffer(buffer(Uint16Array.from({ length: H }, () => float32ToFloat16(1)), 'postnorm_norm'), 'f16', 'row', [H]),
  };
  const output = buffer(new Uint16Array(H), 'postnorm_output');
  const mixedInput = createTensor(buffer(activated, 'postnorm_wrong_dtype'), 'f32', [1, I]);
  await assert.rejects(runMatmulRMSNormFused(mixedInput, weights.down, weights.postFeedforwardNorm,
    { N: H, K: I, eps, transposeB: true, rmsNormWeightOffset: false }),
  /Input dtype f32 must match weight dtype f16/);

  const context = {
    phase: 'decode', config: { hiddenSize: H, intermediateSize: I, hiddenActivation: 'gelu', swigluLimit: null,
      useDoubleWideMlp: false, gatedActivation: true, rmsNormWeightOffset: false },
    ffnStepPrecision: { inputDtype: 'f16', outputDtype: 'f16' },
    executionPolicies: { dtypeTransition: 'require_cast_step' },
  };
  const active = getBufferPool().getStats().activeBuffers;
  for (const recorded of [false, true]) {
    const recorder = recorded ? new CommandRecorder(getDevice(), 'postnorm_dtype') : null;
    try {
      result = await runDenseFFNWithFusedPostNormGPU(0, inputTensor, 1,
        { ...context, recorder }, weights, residualTensor, eps, true, output);
      if (recorder) await recorder.submitAndWait();
      assert.equal(result.dtype, 'f16', 'Fused Q4 gate/up returns F32; the declared down step requires a GPU cast to F16.');
      const actual = Float32Array.from(new Uint16Array(await readBuffer(result.buffer, H * 2)), f16ToF32);
      let maxAbs = 0;
      for (let i = 0; i < H; i++) { assert.ok(Number.isFinite(actual[i])); maxAbs = Math.max(maxAbs, Math.abs(actual[i] - expected[i])); }
      assert.ok(maxAbs < 0.004, `Independent postnorm reference maxAbs=${maxAbs}`);
      console.log(`fused-postnorm-dtype-gpu: recorded=${recorded} maxAbs=${maxAbs}`);
    } finally { if (recorder) await recorder.abort(); }
    assert.equal(getBufferPool().getStats().activeBuffers, active, 'Successful dispatch must release its cast and activation buffers.');
    const failingRecorder = recorded ? new CommandRecorder(getDevice(), 'postnorm_dtype_failure') : null;
    try {
      await assert.rejects(runDenseFFNWithFusedPostNormGPU(0, inputTensor, 1,
        { ...context, recorder: failingRecorder }, weights, residualTensor, null, true, output), /eps is required/);
    } finally { if (failingRecorder) await failingRecorder.abort(); }
    assert.equal(getBufferPool().getStats().activeBuffers, active, 'Rejected dispatch must also release its cast and activation buffers.');
  }
} finally { owned.forEach(releaseBuffer); destroyDevice(); }
