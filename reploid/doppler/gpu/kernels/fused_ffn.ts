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
import { getBufferDtype } from '../buffer-dtypes.js';
import { acquireBuffer } from '../buffer-pool.js';
import type { CommandRecorder } from '../command-recorder.js';
import { KernelBase } from './kernel-base.js';
import { createUniformBufferWithView } from './utils.js';
import type { OutputBufferOptions } from './types.js';
import { trace } from '../../debug/index.js';

/** FFN activation type */
export type FFNActivation = 'silu' | 'gelu';

/** Fused FFN options */
export interface FusedFFNOptions extends OutputBufferOptions {
  /** Batch size (default: 1) */
  batchSize?: number;
  /** Activation function (default: 'silu') */
  activation?: FFNActivation;
  /** Scale factor (default: 1.0) */
  alpha?: number;
}

class FusedFFNKernel extends KernelBase {
  async getPipeline(variant: string): Promise<GPUComputePipeline> {
    return this.getPipelineFor('ffn_fused', variant);
  }

  dispatch(
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroupsX: number,
    workgroupsY: number = 1
  ): void {
    this.dispatchKernel(pipeline, bindGroup, [workgroupsX, workgroupsY, 1], 'ffn_fused');
  }

  record(
    recorder: CommandRecorder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroupsX: number,
    workgroupsY: number = 1
  ): void {
    this.recordKernel(recorder, pipeline, bindGroup, [workgroupsX, workgroupsY, 1], 'ffn_fused');
  }
}

function selectFFNVariant(
  batchSize: number,
  weightDtype: 'f16' | 'f32' | null,
  intermediateSize: number
): string {
  // For small intermediate sizes, use multi-output variant
  if (intermediateSize <= 1024 && batchSize === 1) {
    return 'multi';
  }

  // For batched execution
  if (batchSize > 1) {
    return 'batched';
  }

  // For F16 weights
  if (weightDtype === 'f16') {
    return 'f16';
  }

  // Default F32 variant
  return 'default';
}

function createFFNUniformBuffer(
  device: GPUDevice,
  recorder: CommandRecorder | null,
  params: {
    M: number;
    hiddenSize: number;
    intermediateSize: number;
    alpha: number;
    activation: FFNActivation;
  }
): GPUBuffer {
  return createUniformBufferWithView(
    'ffn_fused_uniforms',
    20,
    (view) => {
      view.setUint32(0, params.M, true);
      view.setUint32(4, params.hiddenSize, true);
      view.setUint32(8, params.intermediateSize, true);
      view.setFloat32(12, params.alpha, true);
      view.setUint32(16, params.activation === 'silu' ? 0 : 1, true);
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
 * @param input Input tensor [batchSize, hiddenSize]
 * @param W_gate Gate weight matrix [intermediateSize, hiddenSize]
 * @param W_up Up weight matrix [intermediateSize, hiddenSize]
 * @param hiddenSize Input dimension
 * @param intermediateSize Output dimension
 * @param options FFN options
 * @returns Output tensor [batchSize, intermediateSize]
 */
export async function runFusedFFN(
  input: GPUBuffer,
  W_gate: GPUBuffer,
  W_up: GPUBuffer,
  hiddenSize: number,
  intermediateSize: number,
  options: FusedFFNOptions = {}
): Promise<GPUBuffer> {
  const device = getDevice();
  const {
    batchSize = 1,
    activation = 'silu',
    alpha = 1.0,
    outputBuffer = null,
  } = options;

  const weightDtype = getBufferDtype(W_gate) as 'f16' | 'f32' | null;
  const variant = selectFFNVariant(batchSize, weightDtype, intermediateSize);

  trace.kernels(`FusedFFN: variant=${variant}, batch=${batchSize}, hidden=${hiddenSize}, intermediate=${intermediateSize}, activation=${activation}`);

  const kernel = new FusedFFNKernel(device);
  const pipeline = await kernel.getPipeline(variant);

  // Create output buffer
  const outputSize = batchSize * intermediateSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'ffn_fused_output');

  // Create uniform buffer
  const uniformBuffer = createFFNUniformBuffer(device, null, {
    M: batchSize,
    hiddenSize,
    intermediateSize,
    alpha,
    activation,
  });

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'ffn_fused_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: W_gate } },
      { binding: 3, resource: { buffer: W_up } },
      { binding: 4, resource: { buffer: output } },
    ],
  });

  // Calculate workgroups
  let workgroupsX: number;
  let workgroupsY: number = 1;

  if (variant === 'multi') {
    const outputsPerWg = 4;
    workgroupsX = Math.ceil(intermediateSize / outputsPerWg);
  } else if (variant === 'batched') {
    workgroupsX = intermediateSize;
    workgroupsY = batchSize;
  } else {
    workgroupsX = intermediateSize;
  }

  kernel.dispatch(pipeline, bindGroup, workgroupsX, workgroupsY);

  uniformBuffer.destroy();

  return output;
}

/**
 * Record fused FFN forward pass (batched, no submit)
 */
export async function recordFusedFFN(
  recorder: CommandRecorder,
  input: GPUBuffer,
  W_gate: GPUBuffer,
  W_up: GPUBuffer,
  hiddenSize: number,
  intermediateSize: number,
  options: FusedFFNOptions = {}
): Promise<GPUBuffer> {
  const device = recorder.device;
  const {
    batchSize = 1,
    activation = 'silu',
    alpha = 1.0,
    outputBuffer = null,
  } = options;

  const weightDtype = getBufferDtype(W_gate) as 'f16' | 'f32' | null;
  const variant = selectFFNVariant(batchSize, weightDtype, intermediateSize);

  trace.kernels(`FusedFFN record: variant=${variant}, batch=${batchSize}, hidden=${hiddenSize}, intermediate=${intermediateSize}, activation=${activation}`);

  const kernel = new FusedFFNKernel(device);
  const pipeline = await kernel.getPipeline(variant);

  const outputSize = batchSize * intermediateSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'ffn_fused_output');

  const uniformBuffer = createFFNUniformBuffer(device, recorder, {
    M: batchSize,
    hiddenSize,
    intermediateSize,
    alpha,
    activation,
  });

  const bindGroup = device.createBindGroup({
    label: 'ffn_fused_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: W_gate } },
      { binding: 3, resource: { buffer: W_up } },
      { binding: 4, resource: { buffer: output } },
    ],
  });

  let workgroupsX: number;
  let workgroupsY: number = 1;

  if (variant === 'multi') {
    const outputsPerWg = 4;
    workgroupsX = Math.ceil(intermediateSize / outputsPerWg);
  } else if (variant === 'batched') {
    workgroupsX = intermediateSize;
    workgroupsY = batchSize;
  } else {
    workgroupsX = intermediateSize;
  }

  kernel.record(recorder, pipeline, bindGroup, workgroupsX, workgroupsY);

  return output;
}

/**
 * Calculate memory savings from using fused FFN
 */
export function calculateFusedFFNSavings(
  batchSize: number,
  hiddenSize: number,
  intermediateSize: number
): {
  separateBytes: number;
  fusedBytes: number;
  savingsBytes: number;
  savingsPct: number;
} {
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
