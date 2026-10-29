import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { destroyDevice, getDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { runMatmul, recordMatmul } from '../../src/gpu/kernels/matmul.js';
import { getPipelineProvenance } from '../../src/gpu/kernels/pipeline-cache.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, releaseBuffer, readBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) {
  if (process.env.DOPPLER_REQUIRE_F32_GEMV_GPU === '1') throw new Error(probe.reason);
  console.log(`matmul-f32-gemv-gpu.test: skipped (${probe.reason})`);
  process.exit(0);
}

const candidateStep = { op: 'q_proj', kernel: 'matmul_gemv_f32.wgsl', entry: 'main' };
const baselineStep = { op: 'q_proj', kernel: 'matmul_f32.wgsl', entry: 'main' };
const pathFor = step => ({ activationDtype: 'f32', decode: { steps: [step] } });
const report = { provider: probe.providerReceipt, capabilities: getKernelCapabilities(), cases: [] };

// Observe the pipeline bound at the actual dispatch boundary, rather than a
// cache entry that could exist without being submitted. Native methods retain
// their original receivers; the observation changes no dispatch parameters.
async function observeDispatches(operation) {
  const device = getDevice();
  const descriptor = Object.getOwnPropertyDescriptor(device, 'createCommandEncoder');
  const createCommandEncoder = device.createCommandEncoder;
  const dispatches = [];
  device.createCommandEncoder = (...args) => {
    const encoder = createCommandEncoder.apply(device, args);
    return new Proxy(encoder, {
      get(target, key) {
        if (key === 'beginComputePass') {
          return (...passArgs) => {
            const pass = target.beginComputePass(...passArgs);
            let pipeline = null;
            return new Proxy(pass, {
              get(passTarget, passKey) {
                if (passKey === 'setPipeline') return value => {
                  pipeline = value;
                  return passTarget.setPipeline(value);
                };
                if (passKey === 'dispatchWorkgroups') return (...workgroups) => {
                  dispatches.push({ workgroups, pipeline: getPipelineProvenance(pipeline) });
                  return passTarget.dispatchWorkgroups(...workgroups);
                };
                const value = Reflect.get(passTarget, passKey, passTarget);
                return typeof value === 'function' ? value.bind(passTarget) : value;
              },
            });
          };
        }
        const value = Reflect.get(target, key, target);
        return typeof value === 'function' ? value.bind(target) : value;
      },
    });
  };
  try {
    const result = await operation();
    assert.equal(dispatches.length, 1, 'expected exactly one GEMV dispatch');
    const provenance = dispatches[0].pipeline;
    assert.equal(provenance?.shaderFile, candidateStep.kernel, 'requested GEMV shader did not dispatch');
    assert.equal(provenance.entryPoint, candidateStep.entry);
    assert.equal(provenance.variant, 'gemv_f32');
    assert.match(provenance.shaderHash, /^sha256:[a-f0-9]{64}$/);
    return { result, dispatches };
  } finally {
    if (descriptor) Object.defineProperty(device, 'createCommandEncoder', descriptor);
    else delete device.createCommandEncoder;
  }
}

function independentReference(input, weights, N, K, transpose, alpha) {
  const result = new Float64Array(N);
  for (let column = 0; column < N; column++) {
    let sum = 0;
    for (let k = 0; k < K; k++) {
      sum += input[k] * weights[transpose ? column * K + k : k * N + column];
    }
    result[column] = sum * alpha;
  }
  return result;
}

function check(expected, actual, label) {
  let maxAbsoluteError = 0;
  for (let index = 0; index < expected.length; index++) {
    const error = Math.abs(actual[index] - expected[index]);
    const limit = 1e-6 + 1e-5 * Math.abs(expected[index]);
    assert.ok(Number.isFinite(actual[index]) && error <= limit,
      `${label}[${index}]: ${actual[index]} != ${expected[index]} (limit ${limit})`);
    maxAbsoluteError = Math.max(maxAbsoluteError, error);
  }
  return maxAbsoluteError;
}

async function runCase({ N, K, transpose = true, alpha = 1, cancellation = false, expect2D = false }) {
  const inputValues = Float32Array.from({ length: K }, (_, index) =>
    cancellation ? (index % 4 === 0 ? 1024 : index % 4 === 2 ? -1024 : 1 / 64)
      : Math.sin(index * 0.17) * 0.125 + Math.cos(index * 0.03) * 0.0625);
  const weightValues = new Float32Array(N * K);
  for (let column = 0; column < N; column++) {
    for (let k = 0; k < K; k++) {
      weightValues[transpose ? column * K + k : k * N + column] = cancellation
        ? (column + 1) / 128
        : Math.sin((column + 1) * 0.13 + k * 0.07) * 0.125;
    }
  }
  const expected = independentReference(inputValues, weightValues, N, K, transpose, alpha);
  const buffers = [];
  const upload = values => {
    const buffer = acquireBuffer(values.byteLength);
    buffers.push(buffer);
    uploadData(buffer, values);
    return buffer;
  };
  let output = null;
  let baseline = null;
  let replay = null;
  let recorder = null;
  try {
    const input = createTensor(upload(inputValues), 'f32', [1, K]);
    const weights = createWeightBuffer(upload(weightValues), 'f32', transpose ? 'row' : 'column',
      transpose ? [N, K] : [K, N]);
    const options = { role: 'q_proj', outputDtype: 'f32', transposeB: transpose,
      kernelPath: pathFor(candidateStep), alpha };
    const immediate = await observeDispatches(() => runMatmul(input, weights, 1, N, K, options));
    output = immediate.result;
    const actual = new Float32Array(await readBuffer(output.buffer, N * 4));
    const maxAbsoluteError = check(expected, actual, 'immediate');
    if (N < 1000) {
      baseline = await runMatmul(input, weights, 1, N, K,
        { ...options, kernelPath: pathFor(baselineStep) });
      check(expected, new Float32Array(await readBuffer(baseline.buffer, N * 4)), 'baseline');
    }
    const recorded = await observeDispatches(async () => {
      recorder = new CommandRecorder(null, 'f32_gemv_reference');
      const result = await recordMatmul(recorder, input, weights, 1, N, K, options);
      await recorder.submitAndWait();
      return result;
    });
    replay = recorded.result;
    const replayValues = new Float32Array(await readBuffer(replay.buffer, N * 4));
    const recordedMaxAbsoluteError = check(expected, replayValues, 'recorded');
    assert.deepEqual(replayValues, actual, 'run/record output differs');
    assert.deepEqual(recorded.dispatches, immediate.dispatches, 'run/record dispatch differs');
    if (expect2D) assert.ok(immediate.dispatches[0].workgroups[1] > 1, 'wide case did not use 2D dispatch');
    report.cases.push({ N, K, transpose, alpha, cancellation, maxAbsoluteError, recordedMaxAbsoluteError,
      outputSha256: createHash('sha256').update(new Uint8Array(actual.buffer)).digest('hex'),
      immediateDispatches: immediate.dispatches, recordedDispatches: recorded.dispatches });
  } finally {
    if (recorder && !recorder.getStats().submitted) await recorder.submitAndWait();
    for (const buffer of [...buffers, output?.buffer, baseline?.buffer, replay?.buffer]) {
      if (buffer) releaseBuffer(buffer);
    }
  }
}

try {
  await runCase({ N: 1, K: 1 });
  await runCase({ N: 7, K: 35, alpha: -0.75 });
  await runCase({ N: 11, K: 513, transpose: false });
  await runCase({ N: 13, K: 1024 });
  await runCase({ N: 5, K: 3073 });
  await runCase({ N: 9, K: 129, cancellation: true });
  await runCase({ N: 262151, K: 3, expect2D: true });
  console.log(`matmul-f32-gemv-gpu.test: ok ${JSON.stringify(report)}`);
} finally {
  await destroyDevice();
  await releaseNodeWebGPU();
}
