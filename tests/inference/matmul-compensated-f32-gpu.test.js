import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { destroyDevice, getDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { runMatmul, recordMatmul } from '../../src/gpu/kernels/matmul.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, releaseBuffer, readBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { observeKernelDispatches } from '../helpers/kernel-dispatch-observer.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) {
  if (process.env.DOPPLER_REQUIRE_F32_COMPENSATED_GPU === '1') throw new Error(probe.reason);
  console.log(`matmul-compensated-f32-gpu.test: skipped (${probe.reason})`);
  process.exit(0);
}

const programs = [
  { variant: 'f32_compensated', kernel: 'matmul_compensated_f32.wgsl', M: 3,
    baselines: ['matmul_f32.wgsl'] },
  { variant: 'gemv_f32_compensated', kernel: 'matmul_gemv_compensated_f32.wgsl', M: 1,
    baselines: ['matmul_f32.wgsl', 'matmul_gemv_f32.wgsl'] },
];
const report = { provider: probe.providerReceipt, capabilities: getKernelCapabilities(),
  oracle: 'Float64 compensated CPU sum of exact single-F32 input products; nearest-F32 output', cases: [] };
const pathFor = kernel => {
  const step = { op: 'q_proj', kernel, entry: 'main' };
  return { activationDtype: 'f32', decode: { steps: [step] }, prefill: { steps: [step] } };
};

function independentReference(input, weights, M, N, K, transpose, alpha) {
  const result = new Float64Array(M * N);
  for (let row = 0; row < M; row++) {
    for (let column = 0; column < N; column++) {
      let sum = 0;
      let correction = 0;
      for (let k = 0; k < K; k++) {
        const product = input[row * K + k] * weights[transpose ? column * K + k : k * N + column];
        const next = sum + product;
        correction += Math.abs(sum) >= Math.abs(product) ? (sum - next) + product : (product - next) + sum;
        sum = next;
      }
      result[row * N + column] = (sum + correction) * alpha;
    }
  }
  return result;
}

function errors(reference, values) {
  let maxAbsoluteError = 0;
  let sumAbsoluteError = 0;
  let nearestF32Mismatches = 0;
  for (let index = 0; index < reference.length; index++) {
    assert.ok(Number.isFinite(values[index]), `nonfinite output at ${index}`);
    const error = Math.abs(values[index] - reference[index]);
    maxAbsoluteError = Math.max(maxAbsoluteError, error);
    sumAbsoluteError += error;
    if (values[index] !== Math.fround(reference[index])) nearestF32Mismatches++;
  }
  return { maxAbsoluteError, sumAbsoluteError, nearestF32Mismatches };
}

function checkDispatch(observation, program) {
  assert.equal(observation.dispatches.length, 1, 'expected one matmul dispatch');
  const provenance = observation.dispatches[0].pipeline;
  assert.equal(provenance?.shaderFile, program.kernel);
  assert.equal(provenance.entryPoint, 'main');
  assert.equal(provenance.variant, program.variant);
  assert.equal(provenance.constants.COMPENSATED_F32, 1);
  assert.match(provenance.shaderHash, /^sha256:[a-f0-9]{64}$/);
}

async function runCase(program, { M = program.M, N = 35, K, transpose = true, alpha = 1, cancellation = false, expect2D = false }) {
  alpha = Math.fround(alpha);
  const inputValues = Float32Array.from({ length: M * K }, (_, index) => {
    if (cancellation) {
      const block = Math.floor((index % K) / 32) % 4;
      return block === 0 ? 2 ** 20 : block === 2 ? -(2 ** 20) : 2 ** -10;
    }
    return Math.sin(index * 0.17) * 1.25 + Math.cos(index * 0.03) * 0.625;
  });
  const weightValues = new Float32Array(N * K);
  for (let column = 0; column < N; column++) {
    for (let k = 0; k < K; k++) {
      weightValues[transpose ? column * K + k : k * N + column] = cancellation
        ? (column + 1) / 128 : Math.sin((column + 1) * 0.13 + k * 0.07) * 0.125;
    }
  }
  const reference = independentReference(inputValues, weightValues, M, N, K, transpose, alpha);
  const buffers = [];
  const own = buffer => { buffers.push(buffer); return buffer; };
  const upload = values => {
    const buffer = own(acquireBuffer(values.byteLength));
    uploadData(buffer, values);
    return buffer;
  };
  let recorder = null;
  try {
    const input = createTensor(upload(inputValues), 'f32', [M, K]);
    const weights = createWeightBuffer(upload(weightValues), 'f32', transpose ? 'row' : 'column',
      transpose ? [N, K] : [K, N]);
    const options = { role: 'q_proj', outputDtype: 'f32', transposeB: transpose,
      kernelPath: pathFor(program.kernel), alpha, phaseOverride: M > 1 ? 'prefill' : 'decode' };
    const immediate = await observeKernelDispatches(async () => {
      const output = await runMatmul(input, weights, M, N, K, options);
      own(output.buffer);
      return output;
    });
    checkDispatch(immediate, program);
    const values = new Float32Array(await readBuffer(immediate.result.buffer, M * N * 4));
    const candidate = errors(reference, values);
    assert.equal(candidate.nearestF32Mismatches, 0, `${program.variant} immediate: ${JSON.stringify(candidate)}`);
    const recorded = await observeKernelDispatches(async () => {
      recorder = new CommandRecorder(null, 'compensated_f32_reference');
      const output = await recordMatmul(recorder, input, weights, M, N, K, options);
      own(output.buffer);
      await recorder.submitAndWait();
      return output;
    });
    checkDispatch(recorded, program);
    const replay = new Float32Array(await readBuffer(recorded.result.buffer, M * N * 4));
    assert.deepEqual(replay, values, 'run/record output differs');
    assert.deepEqual(recorded.dispatches, immediate.dispatches, 'run/record dispatch differs');
    if (expect2D) assert.ok(immediate.dispatches[0].workgroups[1] > 1, 'wide case did not use 2D dispatch');
    const baselines = [];
    for (const kernel of program.baselines) {
      if (expect2D && kernel === 'matmul_f32.wgsl') continue;
      const output = await runMatmul(input, weights, M, N, K, { ...options, kernelPath: pathFor(kernel) });
      own(output.buffer);
      const baseline = errors(reference, new Float32Array(await readBuffer(output.buffer, M * N * 4)));
      assert.ok(candidate.sumAbsoluteError <= baseline.sumAbsoluteError,
        `compensation regressed Float64 error versus ${kernel}`);
      baselines.push({ kernel, ...baseline });
    }
    report.cases.push({ variant: program.variant, M, N, K, transpose, alpha, cancellation,
      candidate, baselines, outputSha256: createHash('sha256').update(new Uint8Array(values.buffer)).digest('hex'),
      immediateDispatches: immediate.dispatches, recordedDispatches: recorded.dispatches });
  } finally {
    if (recorder && !recorder.getStats().submitted) await recorder.abort();
    await getDevice().queue.onSubmittedWorkDone();
    for (const buffer of new Set(buffers)) releaseBuffer(buffer);
  }
}

try {
  for (const program of programs) {
    for (const K of [1024, 3584]) {
      for (const transpose of [true, false]) await runCase(program, { K, transpose });
    }
    await runCase(program, { N: 17, K: 1025, alpha: -0.75 });
    await runCase(program, { M: program.M > 1 ? 17 : 1, N: 19, K: 3587, transpose: false, alpha: 0.1234567 });
    await runCase(program, { N: 5, K: 3584, cancellation: true });
    assert.ok(report.cases.some(c => c.variant === program.variant && c.baselines.every(b =>
      c.candidate.sumAbsoluteError < b.sumAbsoluteError)), `${program.variant} did not improve the CPU-reference error`);
  }
  await runCase(programs[1], { N: 262151, K: 3, expect2D: true });
  console.log(`matmul-compensated-f32-gpu.test: ok ${JSON.stringify(report)}`);
} finally {
  await destroyDevice();
  await releaseNodeWebGPU();
}
