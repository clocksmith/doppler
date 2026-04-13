import assert from 'node:assert/strict';

import { acquireBuffer, uploadData, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { initDevice, setDevice } from '../../src/gpu/device.js';
import { applyUlStage1Batch, cleanupUlPreparedBatch } from '../../src/experimental/training/ul_dataset.js';
import { bootstrapNodeWebGPU } from '../../src/tooling/node-webgpu.js';

function createInputTensor(values, shape, label) {
  const data = new Float32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label);
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape, label);
}

const ulConfig = {
  enabled: true,
  seed: 1337,
  lambda0: 5,
  noiseSchedule: {
    type: 'linear',
    minLogSNR: -1,
    maxLogSNR: 5,
    steps: 8,
  },
};

let webgpuReady = false;
try {
  await bootstrapNodeWebGPU();
  webgpuReady = typeof globalThis.navigator !== 'undefined' && !!globalThis.navigator.gpu;
  if (webgpuReady) {
    await initDevice();
  }
} catch {
  webgpuReady = false;
}

if (!webgpuReady) {
  console.log('ul-stage1-determinism.test: skipped (no WebGPU runtime)');
} else {
  const input = createInputTensor([0.5, 0.1, -0.3, 0.2], [2, 2], 'ul_stage1_determinism_input');

  const preparedA = await applyUlStage1Batch({ input }, ulConfig, {
    seed: 1337,
    stepIndex: 2,
    includeValues: true,
  });
  const preparedB = await applyUlStage1Batch({ input }, ulConfig, {
    seed: 1337,
    stepIndex: 2,
    includeValues: true,
  });
  const preparedC = await applyUlStage1Batch({ input }, ulConfig, {
    seed: 1337,
    stepIndex: 3,
    includeValues: true,
  });

  assert.deepEqual(preparedA.ul.values.noisy, preparedB.ul.values.noisy);
  assert.notDeepEqual(preparedA.ul.values.noisy, preparedC.ul.values.noisy);

  cleanupUlPreparedBatch(preparedA);
  cleanupUlPreparedBatch(preparedB);
  cleanupUlPreparedBatch(preparedC);
  releaseBuffer(input.buffer);
  setDevice(null);
}

console.log('ul-stage1-determinism.test: ok');
