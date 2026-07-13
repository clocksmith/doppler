import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import { QwenGradientAccumulator } from '../../../src/experimental/training/qwen-gradient-accumulator.js';
import { AdamOptimizer } from '../../../src/experimental/training/optimizer.js';
import { initDevice } from '../../../src/gpu/device.js';
import { createTensor } from '../../../src/gpu/tensor.js';
import {
  acquireBuffer,
  readBuffer,
  releaseBuffer,
  uploadData,
} from '../../../src/memory/buffer-pool.js';

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

function average(left, right) {
  return Float32Array.from(left, (value, index) => (value + right[index]) / 2);
}

function scalarAdamw(parameter, gradient, options) {
  const updated = new Float32Array(parameter.length);
  const moment1 = new Float32Array(parameter.length);
  const moment2 = new Float32Array(parameter.length);
  for (let index = 0; index < parameter.length; index += 1) {
    moment1[index] = (1 - options.beta1) * gradient[index];
    moment2[index] = (1 - options.beta2) * gradient[index] * gradient[index];
    const first = moment1[index] / (1 - options.beta1);
    const second = moment2[index] / (1 - options.beta2);
    updated[index] = parameter[index] - options.lr * (
      first / (Math.sqrt(second) + options.eps) + options.weightDecay * parameter[index]
    );
  }
  return { updated, moment1, moment2 };
}

function makeTensor(data, shape, label, ownedTensors) {
  const buffer = acquireBuffer(data.byteLength, undefined, label);
  uploadData(buffer, data);
  const tensor = createTensor(buffer, 'f32', shape, label);
  ownedTensors.push(tensor);
  return tensor;
}

async function readF32(tensor) {
  const count = tensor.shape.reduce((product, value) => product * value, 1);
  return new Float32Array(await readBuffer(tensor.buffer, count * 4));
}

export async function runQwenGradientAccumulationOracle() {
  const baseUrl = new URL('../../../src/config/', import.meta.url);
  setPlatformsBaseUrl(new URL('platforms/', baseUrl).toString());
  setRegistryUrl(new URL('kernels/registry.json', baseUrl).toString());
  await initDevice();

  const optimizerOptions = {
    type: 'adamw',
    lr: 0.001,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weightDecay: 0.01,
    scheduler: { enabled: false },
  };
  const trainingConfig = { training: { optimizer: optimizerOptions } };
  const ownedTensors = [];
  const parameters = [
    new Float32Array([0.25, -0.5, 0.75, -1]),
    new Float32Array([0.125, 0.375, -0.625, -0.875, 1.125, 1.375]),
  ];
  const firstGradients = [
    new Float32Array([0.1, -0.2, 0.3, -0.4]),
    new Float32Array([-0.15, 0.25, -0.35, 0.45, -0.55, 0.65]),
  ];
  const secondGradients = [
    new Float32Array([0.5, 0.6, -0.7, -0.8]),
    new Float32Array([0.75, -0.85, 0.95, -1.05, 1.15, -1.25]),
  ];
  const shapes = [[2, 2], [2, 3]];
  const names = [
    'layers.0.self_attn.q_proj.lora_A',
    'layers.0.mlp.down_proj.lora_B',
  ];
  const parameterTensors = parameters.map((values, index) => makeTensor(
    values,
    shapes[index],
    `qwen_accum_parameter_${index}`,
    ownedTensors
  ));
  const firstGradientTensors = firstGradients.map((values, index) => makeTensor(
    values,
    shapes[index],
    `qwen_accum_gradient_1_${index}`,
    ownedTensors
  ));
  const secondGradientTensors = secondGradients.map((values, index) => makeTensor(
    values,
    shapes[index],
    `qwen_accum_gradient_2_${index}`,
    ownedTensors
  ));
  const optimizer = new AdamOptimizer(trainingConfig);
  const accumulator = new QwenGradientAccumulator({ accumSteps: 2 });

  try {
    const first = await accumulator.accumulate(names.map((name, index) => ({
      name,
      parameter: parameterTensors[index],
      gradient: firstGradientTensors[index],
    })));
    const second = await accumulator.accumulate(names.map((name, index) => ({
      name,
      parameter: parameterTensors[index],
      gradient: secondGradientTensors[index],
    })));
    const expectedGradients = firstGradients.map((values, index) => average(
      values,
      secondGradients[index]
    ));
    const comparisons = {};
    for (let index = 0; index < accumulator.entries.length; index += 1) {
      comparisons[`${names[index]}.accumulatedGradient`] = compare(
        await readF32(accumulator.entries[index].gradient),
        expectedGradients[index]
      );
    }

    await accumulator.step(optimizer, trainingConfig);
    for (let index = 0; index < parameterTensors.length; index += 1) {
      const expected = scalarAdamw(parameters[index], expectedGradients[index], optimizerOptions);
      const state = optimizer.getState(parameterTensors[index]);
      comparisons[`${names[index]}.parameter`] = compare(
        await readF32(parameterTensors[index]),
        expected.updated
      );
      comparisons[`${names[index]}.moment1`] = compare(
        await readF32(state.m),
        expected.moment1
      );
      comparisons[`${names[index]}.moment2`] = compare(
        await readF32(state.v),
        expected.moment2
      );
    }

    const tolerance = 1e-6;
    const passed = first.microstepCount === 1
      && first.ready === false
      && second.microstepCount === 2
      && second.ready === true
      && optimizer.stepCount === 1
      && accumulator.microstepCount === 0
      && accumulator.ready === false
      && Object.values(comparisons).every(
        (entry) => entry.allFinite && entry.maxAbsError <= tolerance
      );
    return {
      artifactType: 'qwen_gradient_accumulation_oracle',
      schemaVersion: 1,
      passed,
      accumSteps: 2,
      parameterCount: names.length,
      parameterNames: names,
      optimizerStepCount: optimizer.stepCount,
      windowResetAfterStep: accumulator.microstepCount === 0 && accumulator.ready === false,
      tolerance: { maxAbsError: tolerance },
      comparisons,
      claimBoundary: 'Two distinct F32 microstep gradients averaged on GPU and applied through one decoupled AdamW update with matching parameters and moments; not completion-loss integration, production Qwen geometry, sustained 1,200-row training, resume, or throughput evidence.',
    };
  } finally {
    accumulator.dispose();
    for (const state of optimizer.state.values()) {
      releaseBuffer(state.m.buffer);
      releaseBuffer(state.v.buffer);
    }
    for (const tensor of ownedTensors) releaseBuffer(tensor.buffer);
  }
}
