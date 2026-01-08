/**
 * Fused FFN Kernel (Tier 2 P0)
 *
 * EXPERIMENTAL: Not currently wired into layer.ts.
 * Complete gate+up fusion kernel, kept for future integration.
 *
 * Fuses gate + up weight projections with activation for:
 * - 2x reduction in input reads
 * - Elimination of intermediate buffers
 * - Single kernel launch instead of 3
 *
 * Supports:
 * - SiLU (SwiGLU) and GeLU (GeGLU) activations
 * - F32 and F16 weight formats
 * - Batched execution for prefill
 * - Command recording for batched GPU operations
 */

import { getDevice, getKernelCapabilities } from '../device.js';
import { acquireBuffer } from '../buffer-pool.js';
import { createTensor } from '../tensor.js';
import { KernelBase } from './kernel-base.js';
import { createUniformBufferWithView } from './utils.js';
import { trace } from '../../debug/index.js';
import { getBuffer, getWeightDtype } from '../weight-buffer.js';
import { isFusedQ4KDisabled } from './matmul.js';

class FusedFFNKernel extends KernelBase {
  /**
   * @param {string} variant
   * @returns {Promise<GPUComputePipeline>}
   */
  async getPipeline(variant) {
    return this.getPipelineFor('fused_ffn', variant);
  }

  /**
   * @param {GPUComputePipeline} pipeline
   * @param {GPUBindGroup} bindGroup
   * @param {number} workgroupsX
   * @param {number} [workgroupsY]
   */
  dispatch(pipeline, bindGroup, workgroupsX, workgroupsY = 1) {
    this.dispatchKernel(pipeline, bindGroup, [workgroupsX, workgroupsY, 1], 'fused_ffn');
  }

  /**
   * @param {import('../command-recorder.js').CommandRecorder} recorder
   * @param {GPUComputePipeline} pipeline
   * @param {GPUBindGroup} bindGroup
   * @param {number} workgroupsX
   * @param {number} [workgroupsY]
   */
  record(recorder, pipeline, bindGroup, workgroupsX, workgroupsY = 1) {
    this.recordKernel(recorder, pipeline, bindGroup, [workgroupsX, workgroupsY, 1], 'fused_ffn');
  }
}

/**
 * @param {number} batchSize
 * @param {'f16' | 'f32' | 'q4k'} weightDtype
 * @param {number} intermediateSize
 * @returns {string}
 */
function selectFFNVariant(batchSize, weightDtype, intermediateSize) {
  // Q4K variants - only if fused path is enabled
  if (weightDtype === 'q4k' && !isFusedQ4KDisabled()) {
    return batchSize > 1 ? 'q4k_batched' : 'q4k';
  }

  // For small intermediate sizes, use multi-output variant
  if (batchSize > 1) {
    return 'batched';
  }

  if (weightDtype === 'f16') {
    return 'f16';
  }

  if (intermediateSize <= 1024) {
    return 'multi';
  }

  // Default F32 variant
  return 'default';
}

/**
 * @param {GPUDevice} device
 * @param {import('../command-recorder.js').CommandRecorder | null} recorder
 * @param {{ M: number; hiddenSize: number; intermediateSize: number; alpha: number; activation: import('./fused_ffn.js').FFNActivation; isQ4K?: boolean }} params
 * @returns {GPUBuffer}
 */
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
    },
    recorder,
    device
  );
}

/**
 * Run fused FFN forward pass
 *
 * Computes: output = activation(input @ W_gate^T) * (input @ W_up^T)
 *
 * @param {import('../tensor.js').Tensor} input Input tensor [batchSize, hiddenSize]
 * @param {GPUBuffer | import('../weight-buffer.js').WeightBuffer} W_gate Gate weight matrix [intermediateSize, hiddenSize]
 * @param {GPUBuffer | import('../weight-buffer.js').WeightBuffer} W_up Up weight matrix [intermediateSize, hiddenSize]
 * @param {number} hiddenSize Input dimension
 * @param {number} intermediateSize Output dimension
 * @param {import('./fused_ffn.js').FusedFFNOptions} [options] FFN options
 * @returns {Promise<import('../tensor.js').Tensor>} Output tensor [batchSize, intermediateSize]
 */
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
  const variant = selectFFNVariant(batchSize, /** @type {'f16' | 'f32' | 'q4k'} */ (gateDtype), intermediateSize);

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
  /** @type {number} */
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

/**
 * Record fused FFN forward pass (batched, no submit)
 * @param {import('../command-recorder.js').CommandRecorder} recorder
 * @param {import('../tensor.js').Tensor} input
 * @param {GPUBuffer | import('../weight-buffer.js').WeightBuffer} W_gate
 * @param {GPUBuffer | import('../weight-buffer.js').WeightBuffer} W_up
 * @param {number} hiddenSize
 * @param {number} intermediateSize
 * @param {import('./fused_ffn.js').FusedFFNOptions} [options]
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
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
  const variant = selectFFNVariant(batchSize, /** @type {'f16' | 'f32' | 'q4k'} */ (gateDtype), intermediateSize);

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

  /** @type {number} */
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

/**
 * Calculate memory savings from using fused FFN
 * @param {number} batchSize
 * @param {number} hiddenSize
 * @param {number} intermediateSize
 * @returns {{ separateBytes: number; fusedBytes: number; savingsBytes: number; savingsPct: number }}
 */
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
