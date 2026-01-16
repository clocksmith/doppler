

import { getDevice, getKernelCapabilities } from '../device.js';
import { acquireBuffer } from '../buffer-pool.js';
import { createTensor } from '../tensor.js';
import { KernelBase } from './kernel-base.js';
import { createUniformBufferWithView } from './utils.js';
import { trace } from '../../debug/index.js';
import { getBuffer, getWeightDtype } from '../weight-buffer.js';
import { isFusedQ4KDisabled } from './matmul.js';
import { getKernelThresholds } from '../../config/schema/index.js';
import { selectRuleValue } from './rule-registry.js';

class FusedFFNKernel extends KernelBase {
  
  async getPipeline(variant) {
    return this.getPipelineFor('fused_ffn', variant);
  }

  
  dispatch(pipeline, bindGroup, workgroupsX, workgroupsY = 1) {
    this.dispatchKernel(pipeline, bindGroup, [workgroupsX, workgroupsY, 1], 'fused_ffn');
  }

  
  record(recorder, pipeline, bindGroup, workgroupsX, workgroupsY = 1) {
    this.recordKernel(recorder, pipeline, bindGroup, [workgroupsX, workgroupsY, 1], 'fused_ffn');
  }
}


function selectFFNVariant(batchSize, weightDtype, intermediateSize) {
  const { multiOutputThreshold } = getKernelThresholds().ffn;
  const canUseQ4K = weightDtype === 'q4k' && !isFusedQ4KDisabled();
  const useMultiOutput = intermediateSize <= multiOutputThreshold;

  return selectRuleValue(
    'fusedFfn',
    'variant',
    { canUseQ4K, batchSize, weightDtype, useMultiOutput }
  );
}


function createFFNUniformBuffer(device, recorder, params) {
  return createUniformBufferWithView(
    'fused_ffn_uniforms',
    32,
    (view) => {
      view.setUint32(0, params.M, true);
      view.setUint32(4, params.hiddenSize, true);
      view.setUint32(8, params.intermediateSize, true);
      view.setFloat32(12, params.alpha, true);
      view.setUint32(16, params.activation === 'silu' ? 0 : 1, true);
      // Q4K needs num_blocks_per_row at offset 20
      if (params.isQ4K) {
        view.setUint32(20, Math.floor(params.hiddenSize / 256), true);
      }
      view.setFloat32(24, params.swigluLimit ?? 0, true);
    },
    recorder,
    device
  );
}


export async function runFusedFFN(
  input,
  W_gate,
  W_up,
  hiddenSize,
  intermediateSize,
  options = {}
) {
  const device = getDevice();
  const {
    batchSize = 1,
    activation = 'silu',
    alpha = 1.0,
    outputBuffer = null,
    swigluLimit = null,
  } = options;

  if (input.dtype !== 'f32') {
    throw new Error('Fused FFN requires f32 activations');
  }

  const gateDtype = getWeightDtype(W_gate) ?? 'f32';
  const upDtype = getWeightDtype(W_up) ?? 'f32';
  if (gateDtype !== upDtype) {
    throw new Error(`Fused FFN requires matching gate/up dtypes (gate=${gateDtype}, up=${upDtype})`);
  }
  if (gateDtype !== 'f16' && gateDtype !== 'f32' && gateDtype !== 'q4k') {
    throw new Error(`Fused FFN does not support ${gateDtype} weights`);
  }

  const isQ4K = gateDtype === 'q4k';
  const variant = selectFFNVariant(batchSize,  (gateDtype), intermediateSize);

  trace.kernels(`FusedFFN: variant=${variant}, batch=${batchSize}, hidden=${hiddenSize}, intermediate=${intermediateSize}, activation=${activation}, isQ4K=${isQ4K}`);

  const kernel = new FusedFFNKernel(device);
  const pipeline = await kernel.getPipeline(variant);

  // Create output buffer
  const outputSize = batchSize * intermediateSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'fused_ffn_output');

  // Create uniform buffer
  const uniformBuffer = createFFNUniformBuffer(device, null, {
    M: batchSize,
    hiddenSize,
    intermediateSize,
    alpha,
    activation,
    isQ4K,
    swigluLimit: activation === 'silu' ? swigluLimit : null,
  });

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'fused_ffn_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: getBuffer(W_gate) } },
      { binding: 3, resource: { buffer: getBuffer(W_up) } },
      { binding: 4, resource: { buffer: output } },
    ],
  });

  // Calculate workgroups
  
  let workgroupsX;
  let workgroupsY = 1;

  if (variant === 'multi') {
    const outputsPerWg = 4;
    workgroupsX = Math.ceil(intermediateSize / outputsPerWg);
  } else if (variant === 'q4k' || variant === 'q4k_batched') {
    // Q4K uses multi-column: 32 columns per workgroup
    const colsPerWg = 32;
    workgroupsX = Math.ceil(intermediateSize / colsPerWg);
    workgroupsY = variant === 'q4k_batched' ? batchSize : 1;
  } else if (variant === 'batched') {
    workgroupsX = intermediateSize;
    workgroupsY = batchSize;
  } else {
    workgroupsX = intermediateSize;
  }

  kernel.dispatch(pipeline, bindGroup, workgroupsX, workgroupsY);

  uniformBuffer.destroy();

  return createTensor(output, 'f32', [batchSize, intermediateSize], 'fused_ffn_output');
}


export async function recordFusedFFN(
  recorder,
  input,
  W_gate,
  W_up,
  hiddenSize,
  intermediateSize,
  options = {}
) {
  const device = recorder.device;
  const {
    batchSize = 1,
    activation = 'silu',
    alpha = 1.0,
    outputBuffer = null,
    swigluLimit = null,
  } = options;

  if (input.dtype !== 'f32') {
    throw new Error('Fused FFN requires f32 activations');
  }

  const gateDtype = getWeightDtype(W_gate) ?? 'f32';
  const upDtype = getWeightDtype(W_up) ?? 'f32';
  if (gateDtype !== upDtype) {
    throw new Error(`Fused FFN requires matching gate/up dtypes (gate=${gateDtype}, up=${upDtype})`);
  }
  if (gateDtype !== 'f16' && gateDtype !== 'f32' && gateDtype !== 'q4k') {
    throw new Error(`Fused FFN does not support ${gateDtype} weights`);
  }

  const isQ4K = gateDtype === 'q4k';
  const variant = selectFFNVariant(batchSize,  (gateDtype), intermediateSize);

  trace.kernels(`FusedFFN record: variant=${variant}, batch=${batchSize}, hidden=${hiddenSize}, intermediate=${intermediateSize}, activation=${activation}, isQ4K=${isQ4K}`);

  const kernel = new FusedFFNKernel(device);
  const pipeline = await kernel.getPipeline(variant);

  const outputSize = batchSize * intermediateSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'fused_ffn_output');

  const uniformBuffer = createFFNUniformBuffer(device, recorder, {
    M: batchSize,
    hiddenSize,
    intermediateSize,
    alpha,
    activation,
    isQ4K,
    swigluLimit: activation === 'silu' ? swigluLimit : null,
  });

  const bindGroup = device.createBindGroup({
    label: 'fused_ffn_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: getBuffer(W_gate) } },
      { binding: 3, resource: { buffer: getBuffer(W_up) } },
      { binding: 4, resource: { buffer: output } },
    ],
  });

  
  let workgroupsX;
  let workgroupsY = 1;

  if (variant === 'multi') {
    const outputsPerWg = 4;
    workgroupsX = Math.ceil(intermediateSize / outputsPerWg);
  } else if (variant === 'q4k' || variant === 'q4k_batched') {
    // Q4K uses multi-column: 32 columns per workgroup
    const colsPerWg = 32;
    workgroupsX = Math.ceil(intermediateSize / colsPerWg);
    workgroupsY = variant === 'q4k_batched' ? batchSize : 1;
  } else if (variant === 'batched') {
    workgroupsX = intermediateSize;
    workgroupsY = batchSize;
  } else {
    workgroupsX = intermediateSize;
  }

  kernel.record(recorder, pipeline, bindGroup, workgroupsX, workgroupsY);

  return createTensor(output, 'f32', [batchSize, intermediateSize], 'fused_ffn_output');
}


export function calculateFusedFFNSavings(
  batchSize,
  hiddenSize,
  intermediateSize
) {
  // Separate kernel approach:
  // - Read input 2x (once for gate, once for up)
  // - Write gate output, up output, final output
  const inputBytes = batchSize * hiddenSize * 4;
  const intermediateBytes = batchSize * intermediateSize * 4;
  const separateBytes = 2 * inputBytes + 3 * intermediateBytes;

  // Fused approach:
  // - Read input 1x
  // - Write final output 1x
  const fusedBytes = inputBytes + intermediateBytes;

  const savingsBytes = separateBytes - fusedBytes;
  const savingsPct = (savingsBytes / separateBytes) * 100;

  return {
    separateBytes,
    fusedBytes,
    savingsBytes,
    savingsPct,
  };
}
