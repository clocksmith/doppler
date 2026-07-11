import assert from 'node:assert/strict';

import { float32ToFloat16 } from '../../src/converter/quantizer.js';
import { destroyDevice } from '../../src/gpu/device.js';
import { runFusedFFN } from '../../src/gpu/kernels/fused_ffn.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { f16ToF32 } from '../../src/loader/dtype-utils.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';

const gpuProbe = await probeNodeGPU();
if (!gpuProbe.ready) {
  console.log(`fused-ffn-gelu-gpu-regression.test: skipped (${gpuProbe.reason})`);
  process.exit(0);
}

const hiddenSize = 256;
const gateValue = 11.3337;
const upValue = 5.6731;
const inputValues = new Float32Array(hiddenSize);
inputValues[0] = 1;

const gateValues = new Uint16Array(hiddenSize);
const upValues = new Uint16Array(hiddenSize);
gateValues[0] = float32ToFloat16(gateValue);
upValues[0] = float32ToFloat16(upValue);

function createGpuBuffer(values, label) {
  const buffer = acquireBuffer(values.byteLength, undefined, label);
  uploadData(buffer, values);
  return buffer;
}

const inputBuffer = createGpuBuffer(inputValues, 'fused_ffn_gelu_input');
const gateBuffer = createGpuBuffer(gateValues, 'fused_ffn_gelu_gate');
const upBuffer = createGpuBuffer(upValues, 'fused_ffn_gelu_up');
const input = createTensor(inputBuffer, 'f32', [1, hiddenSize], 'fused_ffn_gelu_input');
const gate = createWeightBuffer(gateBuffer, 'f16', 'row', [1, hiddenSize], 'fused_ffn_gelu_gate');
const up = createWeightBuffer(upBuffer, 'f16', 'row', [1, hiddenSize], 'fused_ffn_gelu_up');

let output = null;
try {
  output = await runFusedFFN(input, gate, up, hiddenSize, 1, {
    batchSize: 1,
    activation: 'gelu',
    swigluLimit: null,
  });
  const values = new Float32Array(await readBuffer(output.buffer, 4));
  const roundedGate = f16ToF32(gateValues[0]);
  const roundedUp = f16ToF32(upValues[0]);
  const inner = Math.sqrt(2 / Math.PI)
    * (roundedGate + 0.044715 * roundedGate * roundedGate * roundedGate);
  const expected = 0.5 * roundedGate * (1 + Math.tanh(Math.min(15, inner))) * roundedUp;

  assert.equal(Number.isFinite(values[0]), true, `fused GELU returned ${values[0]}`);
  assert.ok(
    Math.abs(values[0] - expected) < 0.1,
    `fused GELU mismatch: actual=${values[0]} expected=${expected}`
  );
  console.log('fused-ffn-gelu-gpu-regression.test: ok');
} finally {
  for (const buffer of [inputBuffer, gateBuffer, upBuffer, output?.buffer]) {
    if (buffer) releaseBuffer(buffer);
  }
  await destroyDevice();
}
