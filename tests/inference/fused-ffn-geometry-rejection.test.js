import assert from 'node:assert/strict';
import { recordFusedFFN } from '../../src/gpu/kernels/fused_ffn.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { setDevice } from '../../src/gpu/device.js';

const buffer = { size: 65536, __dopplerFakeGPUBuffer: true };
const input = createTensor(buffer, 'f32', [1, 256], 'geometry-input');
const weight = createWeightBuffer(buffer, 'q4k', 'row', [32, 256], 'geometry-weight');
let allocations = 0;
const recorder = { device: { createBuffer() { allocations++; throw new Error('Unexpected allocation'); } } };
setDevice({ ...recorder.device, features: new Set(['subgroups']), limits: {}, createBindGroup() {} });
allocations = 0;
await assert.rejects(recordFusedFFN(recorder, input, weight, weight, 256, 32, {
  batchSize: 1, activation: 'silu', swigluLimit: null, variant: 'q4k_metal_simd16',
  pipelineConstants: { WORKGROUP_SIZE: 256, COLS_PER_WG: 64, THREADS_PER_COL: 4 },
}), /requires THREADS_PER_COL=16/);
assert.equal(allocations, 0, 'incompatible geometry must fail before allocating or compiling a zero-output shader');
setDevice(null);
console.log('fused-ffn-geometry-rejection.test: passed');
