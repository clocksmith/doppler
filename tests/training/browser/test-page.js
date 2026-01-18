import { initDevice } from '../../../src/gpu/device.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { createTensor } from '../../../src/gpu/tensor.js';
import { acquireBuffer, uploadData, readBuffer, getBufferPool } from '../../../src/memory/buffer-pool.js';
import {
  runSoftmax,
  runCrossEntropyLoss,
  runCrossEntropyBackward,
  runSoftmaxBackward,
  runRmsNormBackward,
  runMatmul,
  runMatmulBackward,
} from '../../../src/gpu/kernels/index.js';
import { OpType } from '../../../src/training/autograd.js';
import { trainStep } from '../../../src/training/trainer.js';
import { createTrainingConfig } from '../../../src/config/training-defaults.js';
import { AdamOptimizer } from '../../../src/training/optimizer.js';
import { crossEntropyLoss } from '../../../src/training/loss.js';
import { clipGradients } from '../../../src/training/clip.js';
import { compareArrays, KERNEL_TOLERANCES } from '../../kernels/harness/tolerance.js';

function toFloat32(arrayBuffer) {
  return new Float32Array(arrayBuffer);
}

function makeTensorFromFloat32(values, shape, label) {
  const data = new Float32Array(values);
  const buf = acquireBuffer(data.byteLength, undefined, label || 'train_tensor');
  uploadData(buf, data);
  return createTensor(buf, 'f32', shape, label || 'train_tensor');
}

function makeTensorFromUint32(values, shape, label) {
  const data = new Uint32Array(values);
  const buf = acquireBuffer(data.byteLength, undefined, label || 'train_tokens');
  uploadData(buf, data);
  return createTensor(buf, 'f32', shape, label || 'train_tokens');
}

function softmaxCpu(logits, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) {
    let max = -Infinity;
    for (let c = 0; c < cols; c += 1) {
      const v = logits[r * cols + c];
      if (v > max) max = v;
    }
    let sum = 0;
    for (let c = 0; c < cols; c += 1) {
      const exp = Math.exp(logits[r * cols + c] - max);
      out[r * cols + c] = exp;
      sum += exp;
    }
    const inv = sum > 0 ? 1 / sum : 0;
    for (let c = 0; c < cols; c += 1) {
      out[r * cols + c] *= inv;
    }
  }
  return out;
}

function softmaxBackwardCpu(softmax, gradOutput, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) {
    let rowSum = 0;
    for (let c = 0; c < cols; c += 1) {
      rowSum += softmax[r * cols + c] * gradOutput[r * cols + c];
    }
    for (let c = 0; c < cols; c += 1) {
      const s = softmax[r * cols + c];
      out[r * cols + c] = s * (gradOutput[r * cols + c] - rowSum);
    }
  }
  return out;
}

function crossEntropyCpu(softmax, targets, rows, cols) {
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r += 1) {
    const t = targets[r];
    out[r] = -Math.log(Math.max(softmax[r * cols + t], 1e-9));
  }
  return out;
}

function crossEntropyBackwardCpu(softmax, targets, gradOutput, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) {
    const t = targets[r];
    for (let c = 0; c < cols; c += 1) {
      let grad = softmax[r * cols + c];
      if (c === t) {
        grad -= 1;
      }
      out[r * cols + c] = grad * gradOutput[r];
    }
  }
  return out;
}

function rmsnormBackwardCpu(input, weight, gradOutput, rows, cols, eps) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) {
    let sumSq = 0;
    let sumGX = 0;
    for (let c = 0; c < cols; c += 1) {
      const x = input[r * cols + c];
      const g = gradOutput[r * cols + c];
      sumSq += x * x;
      sumGX += g * x;
    }
    const meanSq = sumSq / cols;
    const invRms = 1 / Math.sqrt(meanSq + eps);
    const invRms3 = invRms * invRms * invRms;
    const coeff = (sumGX / cols) * invRms3;
    for (let c = 0; c < cols; c += 1) {
      const x = input[r * cols + c];
      const g = gradOutput[r * cols + c];
      const w = weight[c];
      out[r * cols + c] = w * invRms * g - x * coeff;
    }
  }
  return out;
}

async function initGPU() {
  setPlatformsBaseUrl('/src/config/platforms/');
  setRegistryUrl('/src/config/kernels/registry.json');
  return initDevice();
}

async function readTensor(tensor) {
  const data = await readBuffer(tensor.buffer);
  return toFloat32(data);
}

async function testSoftmaxAndLoss() {
  const rows = 2;
  const cols = 3;
  const logits = new Float32Array([1, 2, 3, 0.5, -1, 2]);
  const logitsTensor = makeTensorFromFloat32(logits, [rows, cols], 'logits');
  const targets = new Uint32Array([2, 0]);
  const targetsTensor = makeTensorFromUint32(targets, [rows], 'targets');

  const softmaxTensor = await runSoftmax(logitsTensor, -1, { batchSize: rows, size: cols });
  const lossTensor = await runCrossEntropyLoss(softmaxTensor, targetsTensor, { numTokens: rows, vocabSize: cols });

  const softmaxGPU = await readTensor(softmaxTensor);
  const lossGPU = await readTensor(lossTensor);
  const softmaxCPU = softmaxCpu(logits, rows, cols);
  const lossCPU = crossEntropyCpu(softmaxCPU, targets, rows, cols);

  const softmaxCompare = compareArrays(softmaxGPU, softmaxCPU, KERNEL_TOLERANCES.softmax);
  const lossCompare = compareArrays(lossGPU, lossCPU, KERNEL_TOLERANCES.softmax);

  return softmaxCompare.pass && lossCompare.pass;
}

async function testCrossEntropyBackward() {
  const rows = 2;
  const cols = 3;
  const logits = new Float32Array([1, 2, 3, 0.5, -1, 2]);
  const softmaxCPU = softmaxCpu(logits, rows, cols);
  const softmaxTensor = makeTensorFromFloat32(softmaxCPU, [rows, cols], 'softmax');
  const targets = new Uint32Array([2, 0]);
  const targetsTensor = makeTensorFromUint32(targets, [rows], 'targets');
  const gradOutput = new Float32Array([1, 0.5]);
  const gradTensor = makeTensorFromFloat32(gradOutput, [rows], 'loss_scale');

  const gradGPU = await runCrossEntropyBackward(softmaxTensor, targetsTensor, gradTensor, { numTokens: rows, vocabSize: cols });
  const gradGPUData = await readTensor(gradGPU);
  const gradCPU = crossEntropyBackwardCpu(softmaxCPU, targets, gradOutput, rows, cols);

  return compareArrays(gradGPUData, gradCPU, KERNEL_TOLERANCES.softmax).pass;
}

async function testSoftmaxBackward() {
  const rows = 2;
  const cols = 3;
  const logits = new Float32Array([1, 2, 3, 0.5, -1, 2]);
  const softmaxCPU = softmaxCpu(logits, rows, cols);
  const softmaxTensor = makeTensorFromFloat32(softmaxCPU, [rows, cols], 'softmax');
  const gradOutput = new Float32Array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3]);
  const gradTensor = makeTensorFromFloat32(gradOutput, [rows, cols], 'grad');

  const gradGPU = await runSoftmaxBackward(softmaxTensor, gradTensor, { rows, cols });
  const gradGPUData = await readTensor(gradGPU);
  const gradCPU = softmaxBackwardCpu(softmaxCPU, gradOutput, rows, cols);

  return compareArrays(gradGPUData, gradCPU, KERNEL_TOLERANCES.softmax).pass;
}

async function testRmsNormBackward() {
  const rows = 2;
  const cols = 4;
  const input = new Float32Array([0.2, -0.1, 0.3, 0.5, -0.4, 0.2, 0.1, -0.2]);
  const weight = new Float32Array([1.0, 1.1, 0.9, 1.05]);
  const gradOutput = new Float32Array([0.4, -0.2, 0.1, 0.3, -0.1, 0.2, -0.3, 0.05]);
  const inputTensor = makeTensorFromFloat32(input, [rows, cols], 'rms_input');
  const weightTensor = makeTensorFromFloat32(weight, [cols], 'rms_weight');
  const gradTensor = makeTensorFromFloat32(gradOutput, [rows, cols], 'rms_grad');
  const eps = 1e-6;

  const gradGPU = await runRmsNormBackward(inputTensor, weightTensor, gradTensor, { numTokens: rows, hiddenSize: cols, eps });
  const gradGPUData = await readTensor(gradGPU);
  const gradCPU = rmsnormBackwardCpu(input, weight, gradOutput, rows, cols, eps);

  return compareArrays(gradGPUData, gradCPU, KERNEL_TOLERANCES.rmsnorm).pass;
}

async function testMatmulBackwardGradient() {
  const M = 2;
  const K = 3;
  const N = 4;
  const input = new Float32Array([0.2, -0.1, 0.3, 0.5, -0.4, 0.2]);
  const weight = new Float32Array([
    0.1, -0.2, 0.3, 0.4,
    -0.5, 0.2, 0.1, -0.3,
    0.25, -0.1, 0.05, 0.2,
  ]);
  const inputTensor = makeTensorFromFloat32(input, [M, K], 'matmul_input');
  const weightTensor = makeTensorFromFloat32(weight, [K, N], 'matmul_weight');

  const forward = await runMatmul(inputTensor, weightTensor, M, N, K, { transposeB: false });
  const gradOutput = new Float32Array(M * N).fill(1);
  const gradTensor = makeTensorFromFloat32(gradOutput, [M, N], 'matmul_grad');
  const grads = await runMatmulBackward(inputTensor, weightTensor, gradTensor, { M, N, K, transposeB: false });

  const gradInputGPU = await readTensor(grads.gradInput);
  const gradWeightGPU = await readTensor(grads.gradWeight);

  const gradInputCPU = new Float32Array(M * K);
  for (let m = 0; m < M; m += 1) {
    for (let k = 0; k < K; k += 1) {
      let sum = 0;
      for (let n = 0; n < N; n += 1) {
        sum += gradOutput[m * N + n] * weight[k * N + n];
      }
      gradInputCPU[m * K + k] = sum;
    }
  }

  const gradWeightCPU = new Float32Array(K * N);
  for (let k = 0; k < K; k += 1) {
    for (let n = 0; n < N; n += 1) {
      let sum = 0;
      for (let m = 0; m < M; m += 1) {
        sum += input[m * K + k] * gradOutput[m * N + n];
      }
      gradWeightCPU[k * N + n] = sum;
    }
  }

  const inputPass = compareArrays(gradInputGPU, gradInputCPU, KERNEL_TOLERANCES.matmul).pass;
  const weightPass = compareArrays(gradWeightGPU, gradWeightCPU, KERNEL_TOLERANCES.matmul).pass;
  return inputPass && weightPass;
}

async function testTrainingLoopLeakAndPerf() {
  const config = createTrainingConfig({
    training: {
      enabled: true,
      lossScaling: { enabled: false },
    },
  });
  const weight = makeTensorFromFloat32([0.1, -0.2, 0.3, 0.4, 0.05, -0.1], [3, 2], 'linear_weight');
  const model = {
    async forward(input, tape) {
      return tape.record(
        OpType.MATMUL,
        (a, b) => runMatmul(a, b, 2, 2, 3, { transposeB: false }),
        [input, weight],
        { M: 2, N: 2, K: 3, transposeB: false }
      );
    },
    loraParams() {
      return [weight];
    },
  };

  const input = makeTensorFromFloat32([0.5, 0.1, -0.3, 0.2, 0.4, -0.1], [2, 3], 'train_input');
  const targets = makeTensorFromUint32([1, 0], [2], 'train_targets');
  const batch = { input, targets };
  const optimizer = new AdamOptimizer(config);

  const pool = getBufferPool();
  const before = pool.getStats().currentBytesAllocated;

  const start = performance.now();
  for (let i = 0; i < 5; i += 1) {
    await trainStep(model, batch, config, {
      crossEntropyLoss,
      clipGradients,
      optimizer,
    });
  }
  const elapsed = performance.now() - start;
  const after = pool.getStats().currentBytesAllocated;

  const growth = Math.max(0, after - before);
  const perfOk = elapsed < 2000;
  const leakOk = growth < 10 * 1024 * 1024;

  return perfOk && leakOk;
}

async function testParityFixture() {
  const res = await fetch('/doppler/tests/fixtures/training/python-parity.json');
  const fixture = await res.json();
  const logits = new Float32Array(fixture.logits);
  const rows = fixture.rows;
  const cols = fixture.cols;
  const targets = new Uint32Array(fixture.targets);
  const softmaxCPU = softmaxCpu(logits, rows, cols);
  const lossCPU = crossEntropyCpu(softmaxCPU, targets, rows, cols);
  let sum = 0;
  for (let i = 0; i < lossCPU.length; i += 1) {
    sum += lossCPU[i];
  }
  const mean = sum / lossCPU.length;
  return Math.abs(mean - fixture.lossMean) < fixture.tolerance;
}

const TESTS = {
  'loss-forward': testSoftmaxAndLoss,
  'softmax-backward': testSoftmaxBackward,
  'cross-entropy-backward': testCrossEntropyBackward,
  'rmsnorm-backward': testRmsNormBackward,
  'matmul-backward': testMatmulBackwardGradient,
  'parity-fixture': testParityFixture,
  'training-leak-perf': testTrainingLoopLeakAndPerf,
};

export const trainingHarness = {
  async getGPU() {
    await initGPU();
    return true;
  },
  async runTest(name) {
    const fn = TESTS[name];
    if (!fn) {
      return { passed: false, error: `Unknown training test: ${name}` };
    }
    const passed = await fn();
    return { passed };
  },
  listTests() {
    return Object.keys(TESTS);
  },
};
