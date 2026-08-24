import { getDevice, getDeviceEpoch, getKernelCapabilities } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { createUniformBufferWithView } from './uniform-utils.js';
import { getPipelineBindGroupLayout } from './pipeline-cache.js';
import { recordDispatch } from './dispatch.js';
import { planRMSNormDispatch } from './rmsnorm.js';
import { getShaderModule } from './shader-cache.js';

const WORKGROUP_SIZE = 256;

let cachedEpoch = -1;
const pipelineCache = new Map();

async function getPipeline(device, useSubgroups) {
  const epoch = getDeviceEpoch();
  if (cachedEpoch !== epoch) {
    pipelineCache.clear();
    cachedEpoch = epoch;
  }
  const key = useSubgroups ? 'subgroup' : 'default';
  const cached = pipelineCache.get(key);
  if (cached) return cached;
  const module = await getShaderModule(
    device,
    useSubgroups ? 'rmsnorm_stats_subgroups.wgsl' : 'rmsnorm_stats.wgsl',
    `rmsnorm_stats_${key}`
  );
  const pipeline = device.createComputePipeline({
    label: `rmsnorm_stats_pipeline_${key}`,
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'main',
      constants: { WORKGROUP_SIZE },
    },
  });
  pipelineCache.set(key, pipeline);
  return pipeline;
}

function createStatsUniform(device, recorder, options) {
  return createUniformBufferWithView(
    'rmsnorm_stats_uniforms',
    16,
    (view) => {
      view.setUint32(0, options.hiddenSize, true);
      view.setUint32(4, options.batchSize, true);
      view.setFloat32(8, options.eps, true);
      view.setUint32(12, options.tokenStride, true);
    },
    recorder,
    device
  );
}

function validateStatsInputs(input, residual, options) {
  const batchSize = options.batchSize ?? 1;
  const hiddenSize = options.hiddenSize;
  if (!Number.isInteger(batchSize) || batchSize <= 0) {
    throw new Error(`[rmsnorm_stats] batchSize must be a positive integer; got ${String(batchSize)}.`);
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
    throw new Error(`[rmsnorm_stats] hiddenSize must be a positive integer; got ${String(hiddenSize)}.`);
  }
  if (input?.dtype !== 'f32' || residual?.dtype !== 'f32') {
    throw new Error(`[rmsnorm_stats] requires f32 input and residual tensors; got input=${input?.dtype}, residual=${residual?.dtype}.`);
  }
  return { batchSize, hiddenSize };
}

function createBindGroup(device, pipeline, uniformBuffer, input, residual, prenormBuffer, invRmsBuffer) {
  return device.createBindGroup({
    label: 'rmsnorm_stats_bind_group',
    layout: getPipelineBindGroupLayout(pipeline, 0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: residual.buffer } },
      { binding: 3, resource: { buffer: prenormBuffer } },
      { binding: 4, resource: { buffer: invRmsBuffer } },
    ],
  });
}

export async function runRMSNormStats(input, residual, eps, options = {}) {
  const device = getDevice();
  if (!device) throw new Error('No GPU device');
  const { batchSize, hiddenSize } = validateStatsInputs(input, residual, options);
  const outputSize = batchSize * hiddenSize * 4;
  const ownedPrenorm = options.outputBuffer ? null : acquireBuffer(outputSize, undefined, 'rmsnorm_stats_prenorm_sum');
  const prenormBuffer = options.outputBuffer || ownedPrenorm;
  const invRmsBuffer = acquireBuffer(batchSize * 4, undefined, 'rmsnorm_stats_inv_rms');
  const useSubgroups = getKernelCapabilities().hasSubgroups === true;
  const dispatchPlan = planRMSNormDispatch(null, batchSize);
  const uniformBuffer = createStatsUniform(device, null, {
    batchSize,
    hiddenSize,
    eps,
    tokenStride: dispatchPlan.tokenStride,
  });
  try {
    const pipeline = await getPipeline(device, useSubgroups);
    const bindGroup = createBindGroup(device, pipeline, uniformBuffer, input, residual, prenormBuffer, invRmsBuffer);
    const encoder = device.createCommandEncoder({ label: 'rmsnorm_stats_encoder' });
    const pass = encoder.beginComputePass({ label: options.label ?? 'rmsnorm_stats' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...dispatchPlan.workgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
    return {
      prenormSum: createTensor(prenormBuffer, 'f32', [batchSize, hiddenSize], 'rmsnorm_stats_prenorm_sum'),
      invRmsBuffer,
    };
  } catch (error) {
    if (ownedPrenorm) releaseBuffer(ownedPrenorm);
    releaseBuffer(invRmsBuffer);
    throw error;
  }
}

export async function recordRMSNormStats(recorder, input, residual, eps, options = {}) {
  const { batchSize, hiddenSize } = validateStatsInputs(input, residual, options);
  const outputSize = batchSize * hiddenSize * 4;
  const ownedPrenorm = options.outputBuffer ? null : acquireBuffer(outputSize, undefined, 'rmsnorm_stats_prenorm_sum');
  const prenormBuffer = options.outputBuffer || ownedPrenorm;
  const invRmsBuffer = acquireBuffer(batchSize * 4, undefined, 'rmsnorm_stats_inv_rms');
  const useSubgroups = getKernelCapabilities().hasSubgroups === true;
  const dispatchPlan = planRMSNormDispatch(recorder, batchSize);
  const uniformBuffer = createStatsUniform(recorder.device, recorder, {
    batchSize,
    hiddenSize,
    eps,
    tokenStride: dispatchPlan.tokenStride,
  });
  try {
    const pipeline = await getPipeline(recorder.device, useSubgroups);
    const bindGroup = createBindGroup(recorder.device, pipeline, uniformBuffer, input, residual, prenormBuffer, invRmsBuffer);
    recordDispatch(recorder, pipeline, bindGroup, dispatchPlan.workgroups, options.label ?? 'rmsnorm_stats');
    return {
      prenormSum: createTensor(prenormBuffer, 'f32', [batchSize, hiddenSize], 'rmsnorm_stats_prenorm_sum'),
      invRmsBuffer,
    };
  } catch (error) {
    if (ownedPrenorm) releaseBuffer(ownedPrenorm);
    releaseBuffer(invRmsBuffer);
    throw error;
  }
}
