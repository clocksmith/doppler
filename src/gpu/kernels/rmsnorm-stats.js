import { getDevice, getDeviceEpoch, getKernelCapabilities } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import {
  createUniformBufferWithView,
  getPipelineBindGroupLayout,
} from './utils.js';
import { recordDispatch } from './dispatch.js';
import { planRMSNormDispatch } from './rmsnorm.js';

const WORKGROUP_SIZE = 256;

function buildRMSNormStatsShader(useSubgroups) {
  const subgroupEnable = useSubgroups ? 'enable subgroups;\n\n' : '';
  const subgroupHelpers = useSubgroups
    ? `
const MAX_SUBGROUPS: u32 = 32u;
var<workgroup> sg_partial_sums: array<f32, MAX_SUBGROUPS>;

fn reduce_sum(local_sum_sq: f32, thread_idx: u32, sg_lane: u32, sg_size: u32) -> f32 {
  let subgroup_id = thread_idx / sg_size;
  let num_subgroups = (WORKGROUP_SIZE + sg_size - 1u) / sg_size;
  let sg_sum = subgroupAdd(local_sum_sq);
  if (sg_lane == 0u && subgroup_id < num_subgroups) {
    sg_partial_sums[subgroup_id] = sg_sum;
  }
  workgroupBarrier();
  if (thread_idx == 0u) {
    var sum = 0.0;
    for (var s = 0u; s < num_subgroups; s = s + 1u) {
      sum = sum + sg_partial_sums[s];
    }
    sg_partial_sums[0] = sum;
  }
  workgroupBarrier();
  return sg_partial_sums[0];
}
`
    : `
var<workgroup> shared_sum: array<f32, WORKGROUP_SIZE>;

fn reduce_sum(local_sum_sq: f32, thread_idx: u32, sg_lane: u32, sg_size: u32) -> f32 {
  shared_sum[thread_idx] = local_sum_sq;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (thread_idx < stride) {
      shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
    }
    workgroupBarrier();
  }
  return shared_sum[0];
}
`;

  const subgroupParams = useSubgroups
    ? `,\n    @builtin(subgroup_invocation_id) sg_lane: u32,\n    @builtin(subgroup_size) sg_size: u32`
    : '';
  const reduceArgs = useSubgroups ? 'local_sum_sq, thread_idx, sg_lane, sg_size' : 'local_sum_sq, thread_idx, 0u, 1u';

  return /* wgsl */ `${subgroupEnable}override WORKGROUP_SIZE: u32 = 256u;

struct RMSNormStatsParams {
  hidden_size: u32,
  num_tokens: u32,
  eps: f32,
  token_stride: u32,
}

@group(0) @binding(0) var<uniform> params: RMSNormStatsParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> prenorm_sum: array<f32>;
@group(0) @binding(4) var<storage, read_write> inv_rms: array<f32>;

${subgroupHelpers}

fn token_index(wg_id: vec3<u32>) -> u32 {
  return wg_id.y * max(params.token_stride, 1u) + wg_id.x;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>${subgroupParams}
) {
  let token_idx = token_index(wg_id);
  let thread_idx = local_id.x;
  let hidden_size = params.hidden_size;
  if (token_idx >= params.num_tokens) {
    return;
  }

  let base = token_idx * hidden_size;
  let elements_per_thread = (hidden_size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
  var local_sum_sq = 0.0;
  for (var i = 0u; i < elements_per_thread; i = i + 1u) {
    let idx = thread_idx * elements_per_thread + i;
    if (idx < hidden_size) {
      let x = input[base + idx] + residual[base + idx];
      prenorm_sum[base + idx] = x;
      local_sum_sq = local_sum_sq + x * x;
    }
  }

  let total_sum = reduce_sum(${reduceArgs});
  if (thread_idx == 0u) {
    inv_rms[token_idx] = inverseSqrt(total_sum / f32(hidden_size) + params.eps);
  }
}
`;
}

let cachedEpoch = -1;
const pipelineCache = new Map();

function getPipeline(device, useSubgroups) {
  const epoch = getDeviceEpoch();
  if (cachedEpoch !== epoch) {
    pipelineCache.clear();
    cachedEpoch = epoch;
  }
  const key = useSubgroups ? 'subgroup' : 'default';
  const cached = pipelineCache.get(key);
  if (cached) return cached;
  const module = device.createShaderModule({
    label: `rmsnorm_stats_${key}`,
    code: buildRMSNormStatsShader(useSubgroups),
  });
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
    const pipeline = getPipeline(device, useSubgroups);
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
    const pipeline = getPipeline(recorder.device, useSubgroups);
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
