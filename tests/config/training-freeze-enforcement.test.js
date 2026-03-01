import assert from 'node:assert/strict';

import { createTrainingConfig } from '../../src/config/training-defaults.js';
import { trainStep } from '../../src/training/trainer.js';
import { AdamOptimizer } from '../../src/training/optimizer.js';
import { crossEntropyLoss } from '../../src/training/loss.js';
import { clipGradients } from '../../src/training/clip.js';
import { OpType } from '../../src/training/autograd.js';
import { runMatmul } from '../../src/gpu/kernels/index.js';
import { initDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { bootstrapNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

function makeF32Tensor(values, shape, label) {
  const data = new Float32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label);
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape, label);
}

function makeU32Tensor(values, shape, label) {
  const data = new Uint32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label);
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape, label);
}

async function readTensorF32(tensor) {
  return Array.from(new Float32Array(await readBuffer(tensor.buffer)));
}

let webgpuReady = false;
try {
  await bootstrapNodeWebGPU();
  installNodeFileFetchShim();
  await initDevice();
  webgpuReady = typeof globalThis.navigator !== 'undefined' && !!globalThis.navigator.gpu;
} catch {
  webgpuReady = false;
}

if (!webgpuReady) {
  console.log('training-freeze-enforcement.test: skipped (no WebGPU runtime)');
} else {
  const config = createTrainingConfig({
    training: {
      enabled: true,
      gradient: { maxNorm: 0 },
      lossScaling: { enabled: false },
      ul: {
        enabled: false,
        freeze: {
          encoder: true,
          base: false,
          lora: false,
        },
      },
    },
  });

  const encoderWeight = makeF32Tensor(
    [0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.05, 0.07, -0.03],
    [3, 3],
    'freeze_encoder_weight'
  );
  const baseWeight = makeF32Tensor(
    [0.1, -0.2, 0.3, 0.4, 0.05, -0.1],
    [3, 2],
    'freeze_base_weight'
  );
  const input = makeF32Tensor([0.5, 0.1, -0.3, 0.2, 0.4, -0.1], [2, 3], 'freeze_input');
  const targets = makeU32Tensor([1, 0], [2], 'freeze_targets');

  const beforeEncoder = await readTensorF32(encoderWeight);
  const beforeBase = await readTensorF32(baseWeight);

  const model = {
    async forward(inputTensor, tape) {
      const hidden = await tape.record(
        OpType.MATMUL,
        (a, b) => runMatmul(a, b, 2, 3, 3, { transposeB: false }),
        [inputTensor, encoderWeight],
        { M: 2, N: 3, K: 3, transposeB: false }
      );
      return tape.record(
        OpType.MATMUL,
        (a, b) => runMatmul(a, b, 2, 2, 3, { transposeB: false }),
        [hidden, baseWeight],
        { M: 2, N: 2, K: 3, transposeB: false }
      );
    },
    paramGroups() {
      return {
        encoder: [encoderWeight],
        base: [baseWeight],
        lora: [baseWeight],
      };
    },
  };

  const result = await trainStep(model, { input, targets }, config, {
    optimizer: new AdamOptimizer(config),
    crossEntropyLoss,
    clipGradients,
  });

  const afterEncoder = await readTensorF32(encoderWeight);
  const afterBase = await readTensorF32(baseWeight);

  assert.deepEqual(afterEncoder, beforeEncoder, 'encoder group must remain unchanged when frozen');
  assert.ok(
    afterBase.some((value, index) => Math.abs(value - beforeBase[index]) > 1e-7),
    'base group should be updated when not frozen'
  );
  assert.ok(result.paramGroupMetrics?.frozenGroups?.includes('encoder'));
  assert.ok(result.paramGroupMetrics?.trainableGroups?.includes('base'));

  releaseBuffer(encoderWeight.buffer);
  releaseBuffer(baseWeight.buffer);
  releaseBuffer(input.buffer);
  releaseBuffer(targets.buffer);
}

console.log('training-freeze-enforcement.test: ok');
