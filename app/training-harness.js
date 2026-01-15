
import { initializeDevice } from '../src/inference/test-harness.js';
import { createTrainingConfig } from '../src/config/training-defaults.js';
import { TrainingRunner, AdamOptimizer, crossEntropyLoss, clipGradients } from '../src/training/index.js';
import { OpType } from '../src/training/autograd.js';
import { runMatmul } from '../src/gpu/kernels/index.js';
import { acquireBuffer, uploadData } from '../src/gpu/buffer-pool.js';
import { createTensor } from '../src/gpu/tensor.js';

function makeTensor(values, shape, label) {
  const data = new Float32Array(values);
  const buf = acquireBuffer(data.byteLength, undefined, label);
  uploadData(buf, data);
  return createTensor(buf, 'f32', shape, label);
}

function makeTargetTensor(values, shape, label) {
  const data = new Uint32Array(values);
  const buf = acquireBuffer(data.byteLength, undefined, label);
  uploadData(buf, data);
  return createTensor(buf, 'f32', shape, label);
}

function createToyModel(weight) {
  return {
    async forward(input, tape) {
      const [tokens, features] = input.shape;
      const [, classes] = weight.shape;
      return tape.record(
        OpType.MATMUL,
        (a, b) => runMatmul(a, b, tokens, classes, features, { transposeB: false }),
        [input, weight],
        { M: tokens, N: classes, K: features, transposeB: false }
      );
    },
    loraParams() {
      return [weight];
    },
  };
}

function readParams() {
  const params = new URLSearchParams(window.location.search);
  return {
    steps: Number.parseInt(params.get('steps') || '5', 10),
    lr: Number.parseFloat(params.get('lr') || '0.1'),
  };
}

export async function runTrainingDemo() {
  await initializeDevice();

  const { steps, lr } = readParams();
  const config = createTrainingConfig({
    training: {
      enabled: true,
      optimizer: { lr },
      lossScaling: { enabled: false },
    },
  });

  const weight = makeTensor([0.1, -0.2, 0.3, 0.4, 0.05, -0.1], [3, 2], 'demo_weight');
  const model = createToyModel(weight);

  const input = makeTensor([0.5, 0.1, -0.3, 0.2, 0.4, -0.1], [2, 3], 'demo_input');
  const targets = makeTargetTensor([1, 0], [2], 'demo_targets');
  const dataset = [{ input, targets }];

  const runner = new TrainingRunner(config, {
    optimizer: new AdamOptimizer(config),
    crossEntropyLoss,
    clipGradients,
    onStep: ({ step, loss }) => {
      const row = document.createElement('div');
      row.className = 'row';
      row.textContent = `step ${step}: loss=${loss.toFixed(4)}`;
      document.getElementById('results').appendChild(row);
    },
  });

  await runner.run(model, dataset, { epochs: 1, batchSize: 1, maxSteps: steps, logEvery: 1 });
  document.getElementById('status').textContent = 'Done';
}

runTrainingDemo().catch((err) => {
  document.getElementById('status').textContent = `Error: ${err.message || err}`;
});
