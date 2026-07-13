import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import { AdamOptimizer } from '../../../src/experimental/training/optimizer.js';
import { getKernelCapabilities, initDevice } from '../../../src/gpu/device.js';
import { createTensor } from '../../../src/gpu/tensor.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../../src/memory/buffer-pool.js';

function makeTensor(values, label) {
  const buffer = acquireBuffer(values.byteLength, undefined, label);
  uploadData(buffer, values);
  return createTensor(buffer, 'f32', [values.length], label);
}

async function readF32(tensor) {
  return new Float32Array(await readBuffer(tensor.buffer, tensor.shape[0] * 4));
}

function compare(actual, expected) {
  let maxAbsError = 0;
  let squaredError = 0;
  let allFinite = true;
  for (let index = 0; index < expected.length; index += 1) {
    const error = Math.abs(actual[index] - expected[index]);
    maxAbsError = Math.max(maxAbsError, error);
    squaredError += error * error;
    allFinite = allFinite && Number.isFinite(actual[index]);
  }
  return {
    elementCount: expected.length,
    allFinite,
    maxAbsError,
    rmse: Math.sqrt(squaredError / expected.length),
  };
}

function scalarStep(parameters, gradients, moment1, moment2, options, step) {
  const nextParameters = new Float32Array(parameters.length);
  const nextMoment1 = new Float32Array(parameters.length);
  const nextMoment2 = new Float32Array(parameters.length);
  for (let index = 0; index < parameters.length; index += 1) {
    const gradient = gradients[index];
    const first = options.beta1 * moment1[index] + (1 - options.beta1) * gradient;
    const second = options.beta2 * moment2[index]
      + (1 - options.beta2) * gradient * gradient;
    const correctedFirst = first / (1 - (options.beta1 ** step));
    const correctedSecond = second / (1 - (options.beta2 ** step));
    nextParameters[index] = parameters[index] - options.lr * (
      correctedFirst / (Math.sqrt(correctedSecond) + options.eps)
      + options.weightDecay * parameters[index]
    );
    nextMoment1[index] = first;
    nextMoment2[index] = second;
  }
  return {
    parameters: nextParameters,
    moment1: nextMoment1,
    moment2: nextMoment2,
  };
}

export async function runAdamwOptimizerOracle() {
  const baseUrl = new URL('../../../src/config/', import.meta.url);
  setPlatformsBaseUrl(new URL('platforms/', baseUrl).toString());
  setRegistryUrl(new URL('kernels/registry.json', baseUrl).toString());
  await initDevice();

  const initialParameters = new Float32Array([0.2, -0.4, 0.7, -0.1, 0.05, -0.8]);
  const firstGradients = new Float32Array([0.3, -0.2, 0.1, 0.5, -0.4, 0.25]);
  const secondGradients = new Float32Array([-0.15, 0.35, -0.25, 0.2, 0.45, -0.1]);
  const optimizerOptions = {
    type: 'adamw',
    lr: 0.001,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weightDecay: 0.1,
    scheduler: { enabled: false },
  };
  const config = { training: { optimizer: optimizerOptions } };
  const parameters = makeTensor(initialParameters, 'adamw_oracle_parameters');
  const gradients = makeTensor(firstGradients, 'adamw_oracle_gradients');
  const optimizer = new AdamOptimizer(config);
  let firstExpected = null;
  let secondExpected = null;
  try {
    firstExpected = scalarStep(
      initialParameters,
      firstGradients,
      new Float32Array(initialParameters.length),
      new Float32Array(initialParameters.length),
      optimizerOptions,
      1
    );
    await optimizer.step([parameters], new Map([[parameters, gradients]]), config);
    uploadData(gradients.buffer, secondGradients);
    secondExpected = scalarStep(
      firstExpected.parameters,
      secondGradients,
      firstExpected.moment1,
      firstExpected.moment2,
      optimizerOptions,
      2
    );
    await optimizer.step([parameters], new Map([[parameters, gradients]]), config);
    const state = optimizer.getState(parameters);
    const actualParameters = await readF32(parameters);
    const actualMoment1 = await readF32(state.m);
    const actualMoment2 = await readF32(state.v);
    const noDecayOptions = { ...optimizerOptions, weightDecay: 0 };
    const noDecayFirst = scalarStep(
      initialParameters,
      firstGradients,
      new Float32Array(initialParameters.length),
      new Float32Array(initialParameters.length),
      noDecayOptions,
      1
    );
    const noDecaySecond = scalarStep(
      noDecayFirst.parameters,
      secondGradients,
      noDecayFirst.moment1,
      noDecayFirst.moment2,
      noDecayOptions,
      2
    );
    const decayDifference = compare(actualParameters, noDecaySecond.parameters);
    const comparisons = {
      parameters: compare(actualParameters, secondExpected.parameters),
      moment1: compare(actualMoment1, secondExpected.moment1),
      moment2: compare(actualMoment2, secondExpected.moment2),
    };
    const tolerance = 2e-7;
    const passed = optimizer.stepCount === 2
      && Object.values(comparisons).every(
        (entry) => entry.allFinite && entry.maxAbsError <= tolerance
      )
      && decayDifference.maxAbsError > 1e-5;
    const capabilities = getKernelCapabilities();
    return {
      artifactType: 'adamw_optimizer_oracle',
      schemaVersion: 1,
      passed,
      steps: optimizer.stepCount,
      optimizer: optimizerOptions,
      tolerance: { maxAbsError: tolerance },
      comparisons,
      negativeControl: {
        control: 'same_two_steps_with_weight_decay_zero',
        parameterDifference: decayDifference,
        passed: decayDifference.maxAbsError > 1e-5,
      },
      adapterInfo: capabilities.adapterInfo || null,
      claimBoundary: 'Two deterministic F32 decoupled AdamW steps and zero-initialized optimizer state; Qwen loss integration, accumulation, resume, and sustained training remain absent.',
    };
  } finally {
    const state = optimizer.state.get(parameters);
    if (state) {
      releaseBuffer(state.m.buffer);
      releaseBuffer(state.v.buffer);
    }
    releaseBuffer(parameters.buffer);
    releaseBuffer(gradients.buffer);
  }
}
