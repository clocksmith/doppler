import assert from 'node:assert/strict';

import { createTrainingConfig } from '../../src/config/training-defaults.js';
import { trainStep } from '../../src/experimental/training/trainer.js';
import { AdamOptimizer } from '../../src/experimental/training/optimizer.js';
import { crossEntropyLoss } from '../../src/experimental/training/loss.js';
import { clipGradients } from '../../src/experimental/training/clip.js';
import { OpType } from '../../src/experimental/training/autograd.js';
import { runMatmul } from '../../src/gpu/kernels/index.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';

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

const gpuProbe = await probeNodeGPU({ installFileFetchShim: true });
if (!gpuProbe.ready) {
  console.log(`training-freeze-enforcement.test: skipped (${gpuProbe.reason})`);
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
