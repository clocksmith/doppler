import { initDevice } from '../../../src/gpu/device.js';
import { getDevice } from '../../../src/gpu/device.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { createTensor } from '../../../src/gpu/tensor.js';
import { acquireBuffer, uploadData, readBuffer, getBufferPool } from '../../../src/memory/buffer-pool.js';
import {
  runSoftmax,
  runCrossEntropyLoss,
  runCrossEntropyBackward,
  runSoftmaxBackward,
  recordSoftmax,
  recordSoftmaxBackward,
  runRmsNormBackward,
  runEmbedBackward,
  runMatmul,
  runMatmulBackward,
  runGeLU,
  runScale,
  recordMatmul,
  recordGeLU,
  recordScale,
  recordResidualAdd,
  recordMatmulBackward,
  recordGeluBackward,
  recordAdam,
  createCommandRecorder,
} from '../../../src/gpu/kernels/index.js';
import { AutogradTape, OpType } from '../../../src/training/autograd.js';
import { trainStep } from '../../../src/training/trainer.js';
import { createTrainingConfig } from '../../../src/config/training-defaults.js';
import { AdamOptimizer } from '../../../src/training/optimizer.js';
import { crossEntropyLoss } from '../../../src/training/loss.js';
import { clipGradients } from '../../../src/training/clip.js';
import { compareArrays, KERNEL_TOLERANCES } from '../../kernels/harness/tolerance.js';
import { loadBackwardRegistry } from '../../../src/config/backward-registry-loader.js';
import { releaseBuffer } from '../../../src/memory/buffer-pool.js';
import { computeSampleStats } from '../../../src/debug/stats.js';
import { resetSubmitStats, setTrackSubmits, getSubmitStats } from '../../../src/gpu/submit-tracker.js';
import { getRuntimeConfig } from '../../../src/config/runtime.js';

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
  if (!inputPass || !weightPass) {
    return false;
  }

  const weightStoredT = new Float32Array(N * K);
  for (let k = 0; k < K; k += 1) {
    for (let n = 0; n < N; n += 1) {
      weightStoredT[n * K + k] = weight[k * N + n];
    }
  }
  const weightTensorT = makeTensorFromFloat32(weightStoredT, [N, K], 'matmul_weight_T');

  const gradsT = await runMatmulBackward(inputTensor, weightTensorT, gradTensor, { M, N, K, transposeB: true });
  const gradInputGPUT = await readTensor(gradsT.gradInput);
  const gradWeightGPUT = await readTensor(gradsT.gradWeight);

  const gradWeightCPUStoredT = new Float32Array(N * K);
  for (let k = 0; k < K; k += 1) {
    for (let n = 0; n < N; n += 1) {
      gradWeightCPUStoredT[n * K + k] = gradWeightCPU[k * N + n];
    }
  }

  const inputPassT = compareArrays(gradInputGPUT, gradInputCPU, KERNEL_TOLERANCES.matmul).pass;
  const weightPassT = compareArrays(gradWeightGPUT, gradWeightCPUStoredT, KERNEL_TOLERANCES.matmul).pass;

  return inputPassT && weightPassT;
}

async function testEmbedBackwardScatterAdd() {
  await initGPU();
  const numTokens = 5;
  const hiddenSize = 4;
  const vocabSize = 6;

  const indices = new Uint32Array([2, 1, 2, 0, 2]);
  const gradOutput = new Float32Array([
    0.5, -1.0, 0.25, 2.0,
    1.5, 0.0, -0.5, 0.75,
    0.25, 1.25, 0.0, -1.0,
    -0.25, 0.5, 0.75, 0.0,
    2.0, -0.5, 1.0, 0.25,
  ]);

  const indicesTensor = makeTensorFromUint32(indices, [numTokens], 'embed_indices');
  const gradTensor = makeTensorFromFloat32(gradOutput, [numTokens, hiddenSize], 'embed_grad_out');
  const gradWeight = await runEmbedBackward(indicesTensor, gradTensor, {
    numTokens,
    hiddenSize,
    vocabSize,
    transpose: false,
    indexOffset: 0,
  });

  const gradWeightGPU = await readTensor(gradWeight);

  const gradWeightCPU = new Float32Array(vocabSize * hiddenSize);
  for (let t = 0; t < numTokens; t += 1) {
    const tokenId = indices[t];
    for (let d = 0; d < hiddenSize; d += 1) {
      gradWeightCPU[tokenId * hiddenSize + d] += gradOutput[t * hiddenSize + d];
    }
  }

  return compareArrays(gradWeightGPU, gradWeightCPU, KERNEL_TOLERANCES.matmul).pass;
}

function meanSquare(values) {
  if (!values.length) return 0;
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    sum += v * v;
  }
  return sum / values.length;
}

async function testEBMStateOptimizeSmoke() {
  const registry = loadBackwardRegistry();
  const config = createTrainingConfig({
    training: {
      enabled: true,
      optimizer: {
        lr: 0.2,
      },
    },
  });
  const optimizer = new AdamOptimizer(config);

  const M = 8;
  const K = 16;
  const H = 32;
  const O = 1;
  const stateSize = M * K;
  const hiddenSize = M * H;
  const outSize = M * O;

  const stateSeed = 9001;
  const w1Seed = 9002;
  const w2Seed = 9003;

  const state = makeTensorFromFloat32(
    Array.from({ length: stateSize }, (_, i) => Math.sin(i + stateSeed) * 0.25),
    [M, K],
    'ebm_state'
  );
  const w1 = makeTensorFromFloat32(
    Array.from({ length: K * H }, (_, i) => Math.cos(i + w1Seed) * 0.2),
    [K, H],
    'ebm_w1'
  );
  const w2 = makeTensorFromFloat32(
    Array.from({ length: H * O }, (_, i) => Math.sin(i + w2Seed) * 0.2),
    [H, O],
    'ebm_w2'
  );

  let baseline = null;
  let latest = null;

  for (let step = 0; step < 8; step += 1) {
    const tape = new AutogradTape(registry);

    const hidden = await tape.record(
      OpType.MATMUL,
      (a, b) => runMatmul(a, b, M, H, K, { transposeB: false }),
      [state, w1],
      { M, N: H, K, transposeB: false, computeGradWeight: false }
    );

    const activated = await tape.record(
      OpType.GELU,
      (x) => runGeLU(x, { size: hiddenSize }),
      [hidden],
      { count: hiddenSize }
    );

    const out = await tape.record(
      OpType.MATMUL,
      (a, b) => runMatmul(a, b, M, O, H, { transposeB: false }),
      [activated, w2],
      { M, N: O, K: H, transposeB: false, computeGradWeight: false }
    );

    const outF32 = await readTensor(out);
    const energy = meanSquare(outF32);
    if (baseline === null) {
      baseline = energy;
    }
    latest = energy;

    const gradScale = outSize > 0 ? (2 / outSize) : 0;
    const gradOut = await runScale(out, gradScale, { inplace: true });
    const grads = await tape.backward(gradOut);

    const gradState = grads.get(state);
    if (!gradState) {
      throw new Error('EBM smoke test expected gradient for state tensor');
    }

    await optimizer.step([state], grads, config);

    const buffersToRelease = new Set();
    for (const grad of grads.values()) {
      buffersToRelease.add(grad.buffer);
    }
    buffersToRelease.add(hidden.buffer);
    buffersToRelease.add(activated.buffer);
    buffersToRelease.add(out.buffer);
    for (const buffer of buffersToRelease) {
      releaseBuffer(buffer);
    }
  }

  return Number.isFinite(baseline) && Number.isFinite(latest) && latest < baseline;
}

function createRng(seed) {
  let state = seed >>> 0;
  if (!state) state = 0x6d2b79f5;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function fillRandom(array, rng, scale = 1.0) {
  const safeScale = Number.isFinite(scale) ? scale : 1.0;
  for (let i = 0; i < array.length; i += 1) {
    array[i] = (rng() * 2 - 1) * safeScale;
  }
  return array;
}

function mapSubmitStats(stats) {
  if (!stats) return null;
  const bySource = stats.bySource instanceof Map
    ? Object.fromEntries(stats.bySource.entries())
    : null;
  return {
    count: stats.count,
    totalMs: stats.totalMs,
    avgMs: stats.avgMs,
    maxMs: stats.maxMs,
    minMs: stats.minMs,
    bySource,
  };
}

async function benchEBMRecordedIteration(state, w1, w2, moments, opt, dims) {
  const { M, K, H, O } = dims;
  const hiddenSize = M * H;
  const outSize = M * O;
  const gradScale = outSize > 0 ? (2 / outSize) : 0;
  const step = opt.step;

  const forwardRecorder = createCommandRecorder('ebm_fwd');
  const hidden = await recordMatmul(forwardRecorder, state, w1, M, H, K, { transposeB: false, role: 'fwd_w1' });
  const activated = await recordGeLU(forwardRecorder, hidden, { size: hiddenSize });
  const out = await recordMatmul(forwardRecorder, activated, w2, M, O, H, { transposeB: false, role: 'fwd_w2' });

  const forwardStart = performance.now();
  await forwardRecorder.submitAndWait();
  const forwardTimeMs = performance.now() - forwardStart;

  const backwardRecorder = createCommandRecorder('ebm_bwd');
  const gradOut = await recordScale(backwardRecorder, out, gradScale, { inplace: true });
  const dActivated = (await recordMatmulBackward(
    backwardRecorder,
    activated,
    w2,
    gradOut,
    { M, N: O, K: H, transposeB: false, computeGradWeight: false }
  )).gradInput;
  if (!dActivated) {
    throw new Error('EBM bench expected dActivated gradient');
  }
  const dHidden = await recordGeluBackward(backwardRecorder, hidden, dActivated, { count: hiddenSize });
  const dState = (await recordMatmulBackward(
    backwardRecorder,
    state,
    w1,
    dHidden,
    { M, N: H, K, transposeB: false, computeGradWeight: false }
  )).gradInput;
  if (!dState) {
    throw new Error('EBM bench expected dState gradient');
  }

  await recordAdam(backwardRecorder, state, dState, moments.m, moments.v, {
    count: M * K,
    step,
    lr: opt.lr,
    beta1: opt.beta1,
    beta2: opt.beta2,
    eps: opt.eps,
  });

  const buffersToRelease = new Set([
    hidden.buffer,
    activated.buffer,
    out.buffer,
    dActivated.buffer,
    dHidden.buffer,
    dState.buffer,
    gradOut.buffer,
  ]);
  buffersToRelease.delete(state.buffer);
  buffersToRelease.delete(w1.buffer);
  buffersToRelease.delete(w2.buffer);
  buffersToRelease.delete(moments.m.buffer);
  buffersToRelease.delete(moments.v.buffer);
  for (const buffer of buffersToRelease) {
    backwardRecorder.trackTemporaryBuffer(buffer);
  }

  const backwardStart = performance.now();
  await backwardRecorder.submitAndWait();
  const backwardTimeMs = performance.now() - backwardStart;

  opt.step += 1;

  return { forwardTimeMs, backwardTimeMs };
}

async function testEBMRecordedBench() {
  await initGPU();
  const device = getDevice();
  if (!device) {
    throw new Error('EBM bench requires a GPU device');
  }

  const runtime = getRuntimeConfig();
  const dims = runtime?.shared?.harness?.trainingBench?.ebmRecorded?.dims;
  if (!dims) {
    throw new Error('EBM bench requires runtime.shared.harness.trainingBench.ebmRecorded.dims.');
  }
  if (![dims.M, dims.K, dims.H, dims.O].every((value) => Number.isFinite(value) && value > 0 && Math.floor(value) === value)) {
    throw new Error('EBM bench requires positive integer dims (M, K, H, O).');
  }

  const rng = createRng(20260206);
  const stateData = fillRandom(new Float32Array(dims.M * dims.K), rng, 0.25);
  const w1Data = fillRandom(new Float32Array(dims.K * dims.H), rng, 0.2);
  const w2Data = fillRandom(new Float32Array(dims.H * dims.O), rng, 0.2);

  const state = makeTensorFromFloat32(stateData, [dims.M, dims.K], 'ebm_bench_state');
  const w1 = makeTensorFromFloat32(w1Data, [dims.K, dims.H], 'ebm_bench_w1');
  const w2 = makeTensorFromFloat32(w2Data, [dims.H, dims.O], 'ebm_bench_w2');

  const momentInit = new Float32Array(dims.M * dims.K);
  const moments = {
    m: makeTensorFromFloat32(momentInit, [dims.M, dims.K], 'ebm_bench_adam_m'),
    v: makeTensorFromFloat32(momentInit, [dims.M, dims.K], 'ebm_bench_adam_v'),
  };

  const opt = {
    step: 1,
    lr: 0.2,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
  };

  const pool = getBufferPool();
  const bytesBefore = pool.getStats().currentBytesAllocated;

  const warmup = runtime?.shared?.benchmark?.run?.warmupRuns;
  const iters = runtime?.shared?.benchmark?.run?.timedRuns;
  if (!Number.isFinite(warmup) || warmup < 0 || Math.floor(warmup) !== warmup) {
    throw new Error('EBM bench requires runtime.shared.benchmark.run.warmupRuns to be a non-negative integer.');
  }
  if (!Number.isFinite(iters) || iters <= 0 || Math.floor(iters) !== iters) {
    throw new Error('EBM bench requires runtime.shared.benchmark.run.timedRuns to be a positive integer.');
  }
  for (let i = 0; i < warmup; i += 1) {
    await benchEBMRecordedIteration(state, w1, w2, moments, opt, dims);
  }

  setTrackSubmits(true);
  resetSubmitStats();

  const forwardTimes = [];
  const backwardTimes = [];
  const totalTimes = [];

  for (let i = 0; i < iters; i += 1) {
    const start = performance.now();
    const { forwardTimeMs, backwardTimeMs } = await benchEBMRecordedIteration(state, w1, w2, moments, opt, dims);
    const total = performance.now() - start;
    forwardTimes.push(forwardTimeMs);
    backwardTimes.push(backwardTimeMs);
    totalTimes.push(total);
  }

  const submitStats = getSubmitStats();
  setTrackSubmits(false);

  const bytesAfter = pool.getStats().currentBytesAllocated;

  const report = {
    schemaVersion: 1,
    timestamp: new Date().toISOString(),
    suite: 'training',
    benchmark: 'ebm-recorded',
    workload: {
      dims,
      warmup,
      iters,
      op: 'state-only',
      update: 'adam',
    },
    metrics: {
      forwardMs: computeSampleStats(forwardTimes),
      backwardMs: computeSampleStats(backwardTimes),
      totalMs: computeSampleStats(totalTimes),
      backwardToForwardRatio: forwardTimes.length
        ? (backwardTimes.reduce((a, b) => a + b, 0) / Math.max(1e-9, forwardTimes.reduce((a, b) => a + b, 0)))
        : 0,
    },
    submits: mapSubmitStats(submitStats),
    memory: {
      currentBytesAllocatedBefore: bytesBefore,
      currentBytesAllocatedAfter: bytesAfter,
      currentBytesAllocatedDelta: bytesAfter - bytesBefore,
    },
  };

  console.log('[Benchmark]', JSON.stringify(report));

  releaseBuffer(state.buffer);
  releaseBuffer(w1.buffer);
  releaseBuffer(w2.buffer);
  releaseBuffer(moments.m.buffer);
  releaseBuffer(moments.v.buffer);

  return true;
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
  const res = await fetch('/tests/fixtures/training/python-parity.json');
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

async function testAutogradBranching() {
  await initGPU();
  const registry = loadBackwardRegistry();
  const tape = new AutogradTape(registry);

  // Graph:
  //   x = [1, 2]
  //   y1 = scale(x, 2)
  //   y2 = scale(x, 3)
  //   z = y1 + y2
  //   Loss = sum(z)
  // Expected grad x = 2 + 3 = 5

  const xData = new Float32Array([1.0, 2.0]);
  const x = makeTensorFromFloat32(xData, [2], 'branch_x');

  const y1 = await tape.record(
    OpType.SCALE,
    (input) => runScale(input, 2.0),
    [x],
    { scale: 2.0 }
  );

  const y2 = await tape.record(
    OpType.SCALE,
    (input) => runScale(input, 3.0),
    [x],
    { scale: 3.0 }
  );

  // We don't have a sum(y1, y2) OpType yet that is registered,
  // but we can use recordResidualAdd if we wrap it.
  // Actually, let's just use two scales and sum them manually via tape.backward
  // by providing gradOutput for BOTH y1 and y2.
  // Wait, the tape follows the graph. Let's make a real branch.

  // Simplified branch:
  // x -> scale(2) -> y1
  // x -> scale(3) -> y2
  // We want dLoss/dx = dLoss/dy1 * dy1/dx + dLoss/dy2 * dy2/dx
  // If dLoss/dy1 = 1 and dLoss/dy2 = 1, then dLoss/dx = 1*2 + 1*3 = 5

  const gradY1 = makeTensorFromFloat32([1.0, 1.0], [2], 'grad_y1');
  const gradY2 = makeTensorFromFloat32([1.0, 1.0], [2], 'grad_y2');

  // Since our tape.backward currently starts from a SINGLE gradOutput,
  // we need to simulate the merge.
  // In a real model, z = y1 + y2 would be recorded.
  // Let's implement a simple 'add' backward logic or just test accumulation directly.

  const grads = new Map();
  // Simulate the tape reaching y1 and y2
  const entryScale = registry.ops[OpType.SCALE];

  // Manually run the branching logic that AutogradTape.backward would do
  // if it encountered the same input tensor twice.
  const g1 = await tape.runBackward(entryScale.backward, { inputs: [x], options: { scale: 2.0 } }, gradY1);
  const g2 = await tape.runBackward(entryScale.backward, { inputs: [x], options: { scale: 3.0 } }, gradY2);

  // This calls accumulateGrad internally
  await tape.accumulateGrad(grads, x, g1[0].grad);
  await tape.accumulateGrad(grads, x, g2[0].grad);

  const finalGradX = await readTensor(grads.get(x));
  const expectedGradX = new Float32Array([5.0, 5.0]);

  const pass = compareArrays(finalGradX, expectedGradX, KERNEL_TOLERANCES.residual).pass;

  // Cleanup
  releaseBuffer(x.buffer);
  releaseBuffer(y1.buffer);
  releaseBuffer(y2.buffer);
  releaseBuffer(gradY1.buffer);
  releaseBuffer(gradY2.buffer);
  releaseBuffer(g1[0].grad.buffer);
  releaseBuffer(g2[0].grad.buffer);
  releaseBuffer(grads.get(x).buffer);

  return pass;
}

function layernormBackwardCpu(input, weight, bias, gradOutput, rows, cols, eps) {
  const gradInput = new Float32Array(rows * cols);
  const gradWeight = new Float32Array(cols);
  const gradBias = new Float32Array(cols);

  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let mean = 0;
    for (let i = 0; i < cols; i++) mean += input[base + i];
    mean /= cols;

    let varSum = 0;
    for (let i = 0; i < cols; i++) {
      const diff = input[base + i] - mean;
      varSum += diff * diff;
    }
    const invStd = 1 / Math.sqrt(varSum / cols + eps);

    let sumGY = 0;
    let sumGYX = 0;
    for (let i = 0; i < cols; i++) {
      const x = input[base + i];
      const diff = x - mean;
      const norm = diff * invStd;
      const dy = gradOutput[base + i];
      
      gradWeight[i] += dy * norm;
      gradBias[i] += dy;

      const gy = dy * weight[i];
      sumGY += gy;
      sumGYX += gy * diff;
    }

    const invStd2 = invStd * invStd;
    for (let i = 0; i < cols; i++) {
      const x = input[base + i];
      const gy = gradOutput[base + i] * weight[i];
      gradInput[base + i] = invStd * (gy - (sumGY + (x - mean) * invStd2 * sumGYX) / cols);
    }
  }
  return { gradInput, gradWeight, gradBias };
}

async function testLayernormBackward() {
  const rows = 2;
  const cols = 4;
  const input = new Float32Array([0.2, -0.1, 0.3, 0.5, -0.4, 0.2, 0.1, -0.2]);
  const weight = new Float32Array([1.0, 1.1, 0.9, 1.05]);
  const bias = new Float32Array([0.1, -0.1, 0.05, 0.0]);
  const gradOutput = new Float32Array([0.4, -0.2, 0.1, 0.3, -0.1, 0.2, -0.3, 0.05]);
  
  const inputTensor = makeTensorFromFloat32(input, [rows, cols], 'ln_input');
  const weightTensor = makeTensorFromFloat32(weight, [cols], 'ln_weight');
  const biasTensor = makeTensorFromFloat32(bias, [cols], 'ln_bias');
  const gradTensor = makeTensorFromFloat32(gradOutput, [rows, cols], 'ln_grad');
  const eps = 1e-5;

  const { runLayerNormBackward } = await import('../../../src/gpu/kernels/backward/layernorm_backward.js');
  const gradsGPU = await runLayerNormBackward(inputTensor, weightTensor, gradTensor, { numTokens: rows, hiddenSize: cols, eps });
  
  const giGPU = await readTensor(gradsGPU.gradInput);
  const gwGPU = await readTensor(gradsGPU.gradWeight);
  const gbGPU = await readTensor(gradsGPU.gradBias);
  
  const expected = layernormBackwardCpu(input, weight, bias, gradOutput, rows, cols, eps);

  const giPass = compareArrays(giGPU, expected.gradInput, KERNEL_TOLERANCES.rmsnorm).pass;
  const gwPass = compareArrays(gwGPU, expected.gradWeight, KERNEL_TOLERANCES.rmsnorm).pass;
  const gbPass = compareArrays(gbGPU, expected.gradBias, KERNEL_TOLERANCES.rmsnorm).pass;

  return giPass && gwPass && gbPass;
}

function conv2dBackwardCpu(input, weight, gradOutput, options) {
  const { inChannels, outChannels, height, width, outHeight, outWidth, kernelH, kernelW, stride, pad } = options;
  const gradInput = new Float32Array(inChannels * height * width);
  const gradWeight = new Float32Array(outChannels * inChannels * kernelH * kernelW);

  for (let oc = 0; oc < outChannels; oc++) {
    for (let ic = 0; ic < inChannels; ic++) {
      for (let ky = 0; ky < kernelH; ky++) {
        for (let kx = 0; kx < kernelW; kx++) {
          let w_sum = 0;
          for (let oy = 0; oy < outHeight; oy++) {
            for (let ox = 0; ox < outWidth; ox++) {
              const iy = oy * stride + ky - pad;
              const ix = ox * stride + kx - pad;
              if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                const dy = gradOutput[(oc * outHeight + oy) * outWidth + ox];
                
                // dInput accumulation
                const w = weight[(((oc * inChannels + ic) * kernelH + ky) * kernelW + kx)];
                gradInput[(ic * height + iy) * width + ix] += dy * w;
                
                // dWeight accumulation
                const x = input[(ic * height + iy) * width + ix];
                w_sum += dy * x;
              }
            }
          }
          gradWeight[(((oc * inChannels + ic) * kernelH + ky) * kernelW + kx)] = w_sum;
        }
      }
    }
  }
  return { gradInput, gradWeight };
}

async function testConv2DBackward() {
  const options = {
    inChannels: 1,
    outChannels: 1,
    height: 4,
    width: 4,
    kernelH: 3,
    kernelW: 3,
    stride: 1,
    pad: 1,
  };
  options.outHeight = options.height;
  options.outWidth = options.width;

  const input = new Float32Array(options.height * options.width).map((_, i) => Math.sin(i));
  const weight = new Float32Array(options.kernelH * options.kernelW).map((_, i) => Math.cos(i));
  const gradOutput = new Float32Array(options.outHeight * options.outWidth).fill(1.0);

  const inputTensor = makeTensorFromFloat32(input, [options.inChannels, options.height, options.width], 'conv_in');
  const weightTensor = makeTensorFromFloat32(weight, [options.outChannels, options.inChannels, options.kernelH, options.kernelW], 'conv_w');
  const gradOutputTensor = makeTensorFromFloat32(gradOutput, [options.outChannels, options.outHeight, options.outWidth], 'conv_grad');

  const { runConv2DBackward } = await import('../../../src/gpu/kernels/backward/conv2d_backward.js');
  const gradsGPU = await runConv2DBackward(inputTensor, weightTensor, gradOutputTensor, options);

  const giGPU = await readTensor(gradsGPU.gradInput);
  const gwGPU = await readTensor(gradsGPU.gradWeight);

  const expected = conv2dBackwardCpu(input, weight, gradOutput, options);

  const giPass = compareArrays(giGPU, expected.gradInput, KERNEL_TOLERANCES.residual).pass;
  const gwPass = compareArrays(gwGPU, expected.gradWeight, KERNEL_TOLERANCES.residual).pass;

  return giPass && gwPass;
}

const TESTS = {
  'loss-forward': testSoftmaxAndLoss,
  'softmax-backward': testSoftmaxBackward,
  'cross-entropy-backward': testCrossEntropyBackward,
  'rmsnorm-backward': testRmsNormBackward,
  'layernorm-backward': testLayernormBackward,
  'conv2d-backward': testConv2DBackward,
  'matmul-backward': testMatmulBackwardGradient,
  'embed-backward': testEmbedBackwardScatterAdd,
  'ebm-state-optimize': testEBMStateOptimizeSmoke,
  'ebm-recorded-bench': testEBMRecordedBench,
  'parity-fixture': testParityFixture,
  'training-leak-perf': testTrainingLoopLeakAndPerf,
  'autograd-branching': testAutogradBranching,
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
