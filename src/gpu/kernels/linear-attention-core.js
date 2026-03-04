import { getDevice, getDeviceEpoch } from '../device.js';
import { WORKGROUP_SIZES } from './constants.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import {
  createUniformBufferFromData,
  getOrCreateBindGroupLayout,
  getOrCreatePipelineLayout,
} from './utils.js';
import { recordDispatch } from './dispatch.js';

const CONV_WORKGROUP_SIZE = WORKGROUP_SIZES.DEFAULT;
const HEAD_WORKGROUP_SIZE = 64;

const CONV_SHADER = /* wgsl */ `
override WORKGROUP_SIZE: u32 = 256u;

struct LinearAttentionParams {
  num_tokens: u32,
  conv_dim: u32,
  conv_kernel_size: u32,
  num_v_heads: u32,
  num_k_heads: u32,
  head_k_dim: u32,
  head_v_dim: u32,
  q_size: u32,
  k_size: u32,
  value_dim: u32,
  q_rep: u32,
  _pad_u32_0: u32,
  rms_norm_eps: f32,
  qk_l2norm_eps: f32,
  _pad_f32_0: f32,
  _pad_f32_1: f32,
}

@group(0) @binding(0) var<uniform> params: LinearAttentionParams;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> conv_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> conv_state: array<f32>;
@group(0) @binding(4) var<storage, read_write> conv_out: array<f32>;

fn silu(x: f32) -> f32 {
  if (x >= 0.0) {
    let z = exp(-x);
    return x / (1.0 + z);
  }
  let z = exp(x);
  return x * z / (1.0 + z);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let channel = gid.x;
  if (channel >= params.conv_dim) {
    return;
  }

  let kernel_size = params.conv_kernel_size;
  let state_base = channel * kernel_size;

  for (var token_idx: u32 = 0u; token_idx < params.num_tokens; token_idx = token_idx + 1u) {
    let qkv_idx = token_idx * params.conv_dim + channel;
    let newest = qkv[qkv_idx];

    for (var k: u32 = 0u; k + 1u < kernel_size; k = k + 1u) {
      conv_state[state_base + k] = conv_state[state_base + k + 1u];
    }
    conv_state[state_base + kernel_size - 1u] = newest;

    var mixed: f32 = 0.0;
    for (var k: u32 = 0u; k < kernel_size; k = k + 1u) {
      mixed = mixed + conv_state[state_base + k] * conv_weight[state_base + k];
    }

    conv_out[token_idx * params.conv_dim + channel] = silu(mixed);
  }
}
`;

const RECURRENT_SHADER = /* wgsl */ `
override WORKGROUP_SIZE: u32 = 64u;

struct LinearAttentionParams {
  num_tokens: u32,
  conv_dim: u32,
  conv_kernel_size: u32,
  num_v_heads: u32,
  num_k_heads: u32,
  head_k_dim: u32,
  head_v_dim: u32,
  q_size: u32,
  k_size: u32,
  value_dim: u32,
  q_rep: u32,
  norm_mode: u32,
  rms_norm_eps: f32,
  qk_l2norm_eps: f32,
  _pad_f32_0: f32,
  _pad_f32_1: f32,
}

@group(0) @binding(0) var<uniform> params: LinearAttentionParams;
@group(0) @binding(1) var<storage, read> conv_out: array<f32>;
@group(0) @binding(2) var<storage, read> z_proj: array<f32>;
@group(0) @binding(3) var<storage, read> a_proj: array<f32>;
@group(0) @binding(4) var<storage, read> b_proj: array<f32>;
@group(0) @binding(5) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(6) var<storage, read> a_neg_exp: array<f32>;
@group(0) @binding(7) var<storage, read> norm_weight: array<f32>;
@group(0) @binding(8) var<storage, read_write> recurrent_state: array<f32>;
@group(0) @binding(9) var<storage, read_write> output: array<f32>;

fn softplus(x: f32) -> f32 {
  if (x > 20.0) {
    return x;
  }
  if (x < -20.0) {
    return exp(x);
  }
  return log(1.0 + exp(x));
}

fn silu(x: f32) -> f32 {
  if (x >= 0.0) {
    let z = exp(-x);
    return x / (1.0 + z);
  }
  let z = exp(x);
  return x * z / (1.0 + z);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let head = gid.x;
  if (head >= params.num_v_heads) {
    return;
  }

  let head_k_dim = params.head_k_dim;
  let head_v_dim = params.head_v_dim;
  let head_scale = inverseSqrt(f32(head_k_dim));
  let recurrent_head_base = head * head_k_dim * head_v_dim;
  let recurrent_head_size = head_k_dim * head_v_dim;
  let q_rep = max(params.q_rep, 1u);
  let src_head = head / q_rep;
  let q_base = src_head * head_k_dim;
  let k_base = params.q_size + src_head * head_k_dim;
  let v_base = params.q_size + params.k_size + head * head_v_dim;

  for (var token_idx: u32 = 0u; token_idx < params.num_tokens; token_idx = token_idx + 1u) {
    let conv_row_base = token_idx * params.conv_dim;
    let z_row_base = token_idx * params.value_dim + head * head_v_dim;
    let ab_row_base = token_idx * params.num_v_heads + head;
    let out_row_base = token_idx * params.value_dim + head * head_v_dim;

    var q_norm_sq = 0.0;
    var k_norm_sq = 0.0;
    for (var d: u32 = 0u; d < head_k_dim; d = d + 1u) {
      let q_val = conv_out[conv_row_base + q_base + d];
      let k_val = conv_out[conv_row_base + k_base + d];
      q_norm_sq = q_norm_sq + q_val * q_val;
      k_norm_sq = k_norm_sq + k_val * k_val;
    }

    let q_norm_scale = head_scale / sqrt(q_norm_sq + params.qk_l2norm_eps);
    let k_norm_scale = inverseSqrt(k_norm_sq + params.qk_l2norm_eps);
    let beta = 1.0 / (1.0 + exp(-b_proj[ab_row_base]));
    let g = a_neg_exp[head] * softplus(a_proj[ab_row_base] + dt_bias[head]);
    let g_exp = exp(g);

    for (var i: u32 = 0u; i < recurrent_head_size; i = i + 1u) {
      recurrent_state[recurrent_head_base + i] = recurrent_state[recurrent_head_base + i] * g_exp;
    }

    for (var vd: u32 = 0u; vd < head_v_dim; vd = vd + 1u) {
      var kv_mem = 0.0;
      for (var kd: u32 = 0u; kd < head_k_dim; kd = kd + 1u) {
        let k_normed = conv_out[conv_row_base + k_base + kd] * k_norm_scale;
        let state_idx = recurrent_head_base + kd * head_v_dim + vd;
        kv_mem = kv_mem + recurrent_state[state_idx] * k_normed;
      }
      let delta = (conv_out[conv_row_base + v_base + vd] - kv_mem) * beta;
      for (var kd: u32 = 0u; kd < head_k_dim; kd = kd + 1u) {
        let k_normed = conv_out[conv_row_base + k_base + kd] * k_norm_scale;
        let state_idx = recurrent_head_base + kd * head_v_dim + vd;
        recurrent_state[state_idx] = recurrent_state[state_idx] + k_normed * delta;
      }
    }

    var mean_sq = 0.0;
    for (var vd: u32 = 0u; vd < head_v_dim; vd = vd + 1u) {
      var out_value = 0.0;
      for (var kd: u32 = 0u; kd < head_k_dim; kd = kd + 1u) {
        let q_normed = conv_out[conv_row_base + q_base + kd] * q_norm_scale;
        let state_idx = recurrent_head_base + kd * head_v_dim + vd;
        out_value = out_value + recurrent_state[state_idx] * q_normed;
      }
      output[out_row_base + vd] = out_value;
      let value = out_value;
      mean_sq = mean_sq + value * value;
    }
    let inv_rms = inverseSqrt(mean_sq / f32(head_v_dim) + params.rms_norm_eps);

    for (var vd: u32 = 0u; vd < head_v_dim; vd = vd + 1u) {
      let gate = silu(z_proj[z_row_base + vd]);
      let norm_index = select(vd, head * head_v_dim + vd, params.norm_mode == 1u);
      output[out_row_base + vd] = (output[out_row_base + vd] * inv_rms) * norm_weight[norm_index] * gate;
    }
  }
}
`;

let cachedEpoch = -1;
let convPipeline = null;
let recurrentPipeline = null;
let convBindGroupLayout = null;
let recurrentBindGroupLayout = null;

function createPipelines(device) {
  convBindGroupLayout = getOrCreateBindGroupLayout(
    'linear_attention_conv_layout',
    [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
    device
  );
  recurrentBindGroupLayout = getOrCreateBindGroupLayout(
    'linear_attention_recurrent_layout',
    [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
    device
  );

  const convModule = device.createShaderModule({
    label: 'linear_attention_conv',
    code: CONV_SHADER,
  });
  const recurrentModule = device.createShaderModule({
    label: 'linear_attention_recurrent',
    code: RECURRENT_SHADER,
  });

  convPipeline = device.createComputePipeline({
    label: 'linear_attention_conv_pipeline',
    layout: getOrCreatePipelineLayout('linear_attention_conv_pipeline_layout', [convBindGroupLayout], device),
    compute: {
      module: convModule,
      entryPoint: 'main',
      constants: {
        WORKGROUP_SIZE: CONV_WORKGROUP_SIZE,
      },
    },
  });
  recurrentPipeline = device.createComputePipeline({
    label: 'linear_attention_recurrent_pipeline',
    layout: getOrCreatePipelineLayout('linear_attention_recurrent_pipeline_layout', [recurrentBindGroupLayout], device),
    compute: {
      module: recurrentModule,
      entryPoint: 'main',
      constants: {
        WORKGROUP_SIZE: HEAD_WORKGROUP_SIZE,
      },
    },
  });
}

function ensurePipelines(device) {
  const epoch = getDeviceEpoch();
  if (epoch !== cachedEpoch || !convPipeline || !recurrentPipeline) {
    createPipelines(device);
    cachedEpoch = epoch;
  }
}

function buildParamsData(params) {
  const data = new ArrayBuffer(64);
  const view = new DataView(data);
  view.setUint32(0, params.numTokens, true);
  view.setUint32(4, params.convDim, true);
  view.setUint32(8, params.convKernelSize, true);
  view.setUint32(12, params.numVHeads, true);
  view.setUint32(16, params.numKHeads, true);
  view.setUint32(20, params.headKDim, true);
  view.setUint32(24, params.headVDim, true);
  view.setUint32(28, params.qSize, true);
  view.setUint32(32, params.kSize, true);
  view.setUint32(36, params.valueDim, true);
  view.setUint32(40, params.qRep, true);
  view.setUint32(44, params.normMode, true);
  view.setFloat32(48, params.rmsNormEps, true);
  view.setFloat32(52, params.qkL2NormEps, true);
  view.setFloat32(56, 0, true);
  view.setFloat32(60, 0, true);
  return data;
}

function requireGpuBuffer(buffer, label) {
  if (!(buffer instanceof GPUBuffer)) {
    throw new Error(`linear_attention kernel requires GPUBuffer for ${label}.`);
  }
}

export async function runLinearAttentionCoreGPU(qkvTensor, zTensor, aTensor, bTensor, layerState, options = {}) {
  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for linear_attention core.');
  }
  const recorder = options.recorder ?? null;
  const useRecorder = recorder
    && typeof recorder.getEncoder === 'function'
    && typeof recorder.trackTemporaryBuffer === 'function';

  requireGpuBuffer(qkvTensor?.buffer, 'qkvTensor');
  requireGpuBuffer(zTensor?.buffer, 'zTensor');
  requireGpuBuffer(aTensor?.buffer, 'aTensor');
  requireGpuBuffer(bTensor?.buffer, 'bTensor');
  requireGpuBuffer(layerState?.convWeightGPU, 'convWeightGPU');
  requireGpuBuffer(layerState?.dtBiasGPU, 'dtBiasGPU');
  requireGpuBuffer(layerState?.aNegExpGPU, 'aNegExpGPU');
  requireGpuBuffer(layerState?.normWeightGPU, 'normWeightGPU');
  requireGpuBuffer(layerState?.convStateGPU, 'convStateGPU');
  requireGpuBuffer(layerState?.recurrentStateGPU, 'recurrentStateGPU');

  const numTokens = Number(options.numTokens ?? 0);
  if (!Number.isFinite(numTokens) || numTokens <= 0) {
    throw new Error('runLinearAttentionCoreGPU requires numTokens > 0.');
  }
  if (!Number.isFinite(layerState.headVDim) || layerState.headVDim <= 0) {
    throw new Error(`linear_attention requires positive headVDim, got ${layerState.headVDim}.`);
  }
  if (layerState.normMode !== 'shared' && layerState.normMode !== 'per_head') {
    throw new Error(`linear_attention requires supported normMode, got ${layerState.normMode}.`);
  }

  ensurePipelines(device);

  const convOutSize = numTokens * layerState.convDim * Float32Array.BYTES_PER_ELEMENT;
  const outputSize = numTokens * layerState.valueDim * Float32Array.BYTES_PER_ELEMENT;
  const convOutBuffer = acquireBuffer(convOutSize, undefined, `L${options.layerIdx ?? 0}.linear_conv_out`);
  const outputBuffer = acquireBuffer(outputSize, undefined, `L${options.layerIdx ?? 0}.linear_attention_core_out`);
  if (useRecorder) {
    const paramsBuffer = createUniformBufferFromData(
      'linear_attention_params',
      buildParamsData({
        numTokens,
        convDim: layerState.convDim,
        convKernelSize: layerState.convKernelSize,
        numVHeads: layerState.numVHeads,
        numKHeads: layerState.numKHeads,
        headKDim: layerState.headKDim,
        headVDim: layerState.headVDim,
        qSize: layerState.qSize,
        kSize: layerState.kSize,
        valueDim: layerState.valueDim,
        qRep: layerState.qRep,
        normMode: layerState.normMode === 'per_head' ? 1 : 0,
        rmsNormEps: Number(layerState.rmsNormEps) || 1e-6,
        qkL2NormEps: Number(options.qkL2NormEps) || 1e-6,
      }),
      recorder
    );
    try {
      const convBindGroup = device.createBindGroup({
        label: 'linear_attention_conv_bind_group',
        layout: convBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: qkvTensor.buffer } },
          { binding: 2, resource: { buffer: layerState.convWeightGPU } },
          { binding: 3, resource: { buffer: layerState.convStateGPU } },
          { binding: 4, resource: { buffer: convOutBuffer } },
        ],
      });

      const recurrentBindGroup = device.createBindGroup({
        label: 'linear_attention_recurrent_bind_group',
        layout: recurrentBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: convOutBuffer } },
          { binding: 2, resource: { buffer: zTensor.buffer } },
          { binding: 3, resource: { buffer: aTensor.buffer } },
          { binding: 4, resource: { buffer: bTensor.buffer } },
          { binding: 5, resource: { buffer: layerState.dtBiasGPU } },
          { binding: 6, resource: { buffer: layerState.aNegExpGPU } },
          { binding: 7, resource: { buffer: layerState.normWeightGPU } },
          { binding: 8, resource: { buffer: layerState.recurrentStateGPU } },
          { binding: 9, resource: { buffer: outputBuffer } },
        ],
      });

      recordDispatch(
        recorder,
        convPipeline,
        convBindGroup,
        [Math.ceil(layerState.convDim / CONV_WORKGROUP_SIZE), 1, 1],
        'linear_attention_conv'
      );
      recordDispatch(
        recorder,
        recurrentPipeline,
        recurrentBindGroup,
        [Math.ceil(layerState.numVHeads / HEAD_WORKGROUP_SIZE), 1, 1],
        'linear_attention_recurrent'
      );

      recorder.trackTemporaryBuffer(convOutBuffer);

      return createTensor(
        outputBuffer,
        'f32',
        [numTokens, layerState.valueDim],
        `L${options.layerIdx ?? 0}.linear_attention_core`
      );
    } catch (error) {
      releaseBuffer(convOutBuffer);
      releaseBuffer(outputBuffer);
      throw error;
    }
  }

  const paramsBuffer = createUniformBufferFromData(
    'linear_attention_params',
    buildParamsData({
      numTokens,
      convDim: layerState.convDim,
      convKernelSize: layerState.convKernelSize,
      numVHeads: layerState.numVHeads,
      numKHeads: layerState.numKHeads,
      headKDim: layerState.headKDim,
      headVDim: layerState.headVDim,
      qSize: layerState.qSize,
      kSize: layerState.kSize,
      valueDim: layerState.valueDim,
      qRep: layerState.qRep,
      normMode: layerState.normMode === 'per_head' ? 1 : 0,
      rmsNormEps: Number(layerState.rmsNormEps) || 1e-6,
      qkL2NormEps: Number(options.qkL2NormEps) || 1e-6,
    }),
    null,
    device,
    { useCache: false }
  );
  let submitted = false;

  try {
    const convBindGroup = device.createBindGroup({
      label: 'linear_attention_conv_bind_group',
      layout: convBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: qkvTensor.buffer } },
        { binding: 2, resource: { buffer: layerState.convWeightGPU } },
        { binding: 3, resource: { buffer: layerState.convStateGPU } },
        { binding: 4, resource: { buffer: convOutBuffer } },
      ],
    });

    const recurrentBindGroup = device.createBindGroup({
      label: 'linear_attention_recurrent_bind_group',
      layout: recurrentBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: convOutBuffer } },
        { binding: 2, resource: { buffer: zTensor.buffer } },
        { binding: 3, resource: { buffer: aTensor.buffer } },
        { binding: 4, resource: { buffer: bTensor.buffer } },
        { binding: 5, resource: { buffer: layerState.dtBiasGPU } },
        { binding: 6, resource: { buffer: layerState.aNegExpGPU } },
        { binding: 7, resource: { buffer: layerState.normWeightGPU } },
        { binding: 8, resource: { buffer: layerState.recurrentStateGPU } },
        { binding: 9, resource: { buffer: outputBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: 'linear_attention_core' });

    {
      const pass = encoder.beginComputePass({ label: 'linear_attention_conv_pass' });
      pass.setPipeline(convPipeline);
      pass.setBindGroup(0, convBindGroup);
      pass.dispatchWorkgroups(Math.ceil(layerState.convDim / CONV_WORKGROUP_SIZE), 1, 1);
      pass.end();
    }

    {
      const pass = encoder.beginComputePass({ label: 'linear_attention_recurrent_pass' });
      pass.setPipeline(recurrentPipeline);
      pass.setBindGroup(0, recurrentBindGroup);
      pass.dispatchWorkgroups(Math.ceil(layerState.numVHeads / HEAD_WORKGROUP_SIZE), 1, 1);
      pass.end();
    }

    device.queue.submit([encoder.finish()]);
    submitted = true;

    return createTensor(
      outputBuffer,
      'f32',
      [numTokens, layerState.valueDim],
      `L${options.layerIdx ?? 0}.linear_attention_core`
    );
  } catch (error) {
    releaseBuffer(outputBuffer);
    throw error;
  } finally {
    if (submitted) {
      device.queue.onSubmittedWorkDone()
        .then(() => {
          paramsBuffer.destroy();
        })
        .catch(() => {
          paramsBuffer.destroy();
        });
    } else {
      paramsBuffer.destroy();
    }
    releaseBuffer(convOutBuffer);
  }
}
