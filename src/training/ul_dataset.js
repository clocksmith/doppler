import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../memory/buffer-pool.js';
import { createTensor } from '../gpu/tensor.js';
import { resolveUlScheduledLambda } from './ul_schedule.js';

function xorshift32(value) {
  let x = value >>> 0;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  return x >>> 0;
}

function uniformFromSeed(seed, index) {
  const mixed = xorshift32((seed ^ ((index + 1) * 0x9e3779b9)) >>> 0);
  return (mixed + 1) / 4294967297;
}

function gaussianFromSeed(seed, index) {
  const u1 = Math.max(uniformFromSeed(seed, index * 2), 1e-7);
  const u2 = uniformFromSeed(seed ^ 0x85ebca6b, index * 2 + 1);
  const radius = Math.sqrt(-2 * Math.log(u1));
  const theta = 2 * Math.PI * u2;
  return radius * Math.cos(theta);
}

export function resolveUlNoiseScale(lambda0) {
  const snr = Math.exp(lambda0);
  const alpha = Math.sqrt(snr / (1 + snr));
  const sigma = Math.sqrt(1 / (1 + snr));
  return { alpha, sigma };
}

export { resolveUlScheduledLambda } from './ul_schedule.js';

function summarizeArray(values) {
  if (!values.length) {
    return { mean: 0, std: 0 };
  }
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i];
  }
  const mean = sum / values.length;
  let variance = 0;
  for (let i = 0; i < values.length; i += 1) {
    const delta = values[i] - mean;
    variance += delta * delta;
  }
  variance /= values.length;
  return { mean, std: Math.sqrt(Math.max(variance, 0)) };
}

export async function buildNoisyLatentsFromInputTensor(inputTensor, ulConfig, options = {}) {
  if (!inputTensor || !inputTensor.buffer || !Array.isArray(inputTensor.shape)) {
    throw new Error('UL dataset: input tensor is required.');
  }
  if (inputTensor.dtype !== 'f32') {
    throw new Error(`UL dataset: expected f32 input tensor, got ${inputTensor.dtype}.`);
  }

  const rawInput = new Float32Array(await readBuffer(inputTensor.buffer));
  const elementCount = inputTensor.shape.reduce((acc, value) => acc * Math.max(1, Math.floor(Number(value) || 1)), 1);
  const inputData = rawInput.length === elementCount
    ? rawInput
    : rawInput.slice(0, elementCount);
  const noisy = new Float32Array(inputData.length);
  const noise = new Float32Array(inputData.length);
  const stepIndex = Math.max(0, Math.floor(Number(options.stepIndex) || 0));
  const lambda0 = Number.isFinite(options.lambda0)
    ? options.lambda0
    : resolveUlScheduledLambda(ulConfig, stepIndex);
  const seedBase = Number.isFinite(options.seed) ? options.seed : ulConfig.seed;
  const seed = (seedBase + (stepIndex * 104729)) >>> 0;
  const { alpha, sigma } = resolveUlNoiseScale(lambda0);

  for (let i = 0; i < inputData.length; i += 1) {
    const n = gaussianFromSeed(seed, i);
    noise[i] = n;
    noisy[i] = alpha * inputData[i] + sigma * n;
  }

  const noisyBuffer = acquireBuffer(noisy.byteLength, undefined, 'ul_noisy_latents');
  uploadData(noisyBuffer, noisy);
  const noisyTensor = createTensor(noisyBuffer, 'f32', [...inputTensor.shape], 'ul_noisy_latents');
  const cleanStats = summarizeArray(inputData);
  const noiseStats = summarizeArray(noise);
  const noisyStats = summarizeArray(noisy);

  return {
    noisyTensor,
    cleanStats,
    noiseStats,
    noisyStats,
    alpha,
    sigma,
    lambda0,
    stepIndex,
    cleanValues: options.includeValues === true ? inputData : null,
    noiseValues: options.includeValues === true ? noise : null,
    noisyValues: options.includeValues === true ? noisy : null,
    shape: [...inputTensor.shape],
  };
}

export async function applyUlStage1Batch(batch, ulConfig, options = {}) {
  const stepIndex = Math.max(0, Math.floor(Number(options.stepIndex) || 0));
  const lambda = Number.isFinite(options.lambda0)
    ? options.lambda0
    : resolveUlScheduledLambda(ulConfig, stepIndex);
  const prepared = await buildNoisyLatentsFromInputTensor(batch.input, ulConfig, {
    ...options,
    lambda0: lambda,
    stepIndex,
  });
  return {
    ...batch,
    input: prepared.noisyTensor,
    ul: {
      alpha: prepared.alpha,
      sigma: prepared.sigma,
      clean: prepared.cleanStats,
      noise: prepared.noiseStats,
      noisy: prepared.noisyStats,
      lambda: prepared.lambda0,
      schedule: ulConfig.noiseSchedule || null,
      stepIndex,
      shape: prepared.shape,
      values: options.includeValues === true
        ? {
          clean: prepared.cleanValues ? Array.from(prepared.cleanValues) : null,
          noise: prepared.noiseValues ? Array.from(prepared.noiseValues) : null,
          noisy: prepared.noisyValues ? Array.from(prepared.noisyValues) : null,
        }
        : null,
    },
    _ulTemporaryTensors: [prepared.noisyTensor],
  };
}

export function cleanupUlPreparedBatch(batch) {
  const tensors = Array.isArray(batch?._ulTemporaryTensors) ? batch._ulTemporaryTensors : [];
  for (const tensor of tensors) {
    if (tensor?.buffer) {
      releaseBuffer(tensor.buffer);
    }
  }
}

export function computeLatentBitrateProxy(stats, ulConfig) {
  const variance = Math.max((stats?.noisy?.std ?? 0) ** 2, 1e-8);
  const lambda = Number.isFinite(ulConfig?.lambda0) ? ulConfig.lambda0 : 5;
  return Math.log2(1 + variance * Math.exp(lambda));
}
