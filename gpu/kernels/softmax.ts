/**
 * Softmax Kernels
 *
 * Provides softmax operations with support for:
 * - Temperature scaling
 * - Top-K fused softmax (for MoE routing)
 */

import { getDevice } from '../device.js';
import { setBufferDtype } from '../buffer-dtypes.js';
import { acquireBuffer } from '../buffer-pool.js';
import type { CommandRecorder } from '../command-recorder.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import type { OutputBufferOptions } from './types.js';

/** Softmax kernel options */
export interface SoftmaxOptions extends OutputBufferOptions {
  batchSize?: number;
  size?: number | null;
  seqLen?: number | null;
  temperature?: number;
  normalize?: boolean;
}

/**
 * Run softmax operation
 */
export async function runSoftmax(
  input: GPUBuffer,
  axis: number,
  options: SoftmaxOptions = {}
): Promise<GPUBuffer> {
  const device = getDevice();
  const { batchSize = 1, size, temperature = 1.0, outputBuffer = null } = options;

  const inferredSize = size || (input.size / (batchSize * 4));
  const pipeline = await createPipeline('softmax', 'default');

  const outputSize = batchSize * inferredSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'softmax_output');

  // Create uniform buffer
  // WGSL struct: { innerSize: u32, outerSize: u32, temperature: f32, _pad: u32 }
  const uniformBuffer = createUniformBufferWithView(
    'softmax_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);  // innerSize at offset 0
      view.setUint32(4, batchSize, true);     // outerSize at offset 4
      view.setFloat32(8, temperature, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'softmax_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  dispatch(device, pipeline, bindGroup, batchSize, 'softmax');

  uniformBuffer.destroy();

  return output;
}

/**
 * Run fused softmax + top-K for MoE routing
 */
export async function runSoftmaxTopK(
  logits: GPUBuffer,
  numTokens: number,
  numExperts: number,
  topK: number,
  options: SoftmaxOptions = {}
): Promise<{ indices: GPUBuffer; weights: GPUBuffer }> {
  const device = getDevice();
  const { normalize = true } = options;

  const pipeline = await createPipeline('topk', 'fused');

  // Output buffers: indices [numTokens, topK] as u32, weights [numTokens, topK] as f32
  const indicesSize = numTokens * topK * 4; // u32
  const weightsSize = numTokens * topK * 4; // f32

  const indices = acquireBuffer(indicesSize, undefined, 'softmax_topk_indices');
  const weights = acquireBuffer(weightsSize, undefined, 'softmax_topk_weights');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'softmax_topk_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, numExperts, true);
      view.setUint32(8, topK, true);
      view.setUint32(12, normalize ? 1 : 0, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'softmax_topk_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } },
    ],
  });

  dispatch(device, pipeline, bindGroup, numTokens, 'softmax_topk');

  uniformBuffer.destroy();

  setBufferDtype(indices, 'u32');
  setBufferDtype(weights, 'f32');

  return { indices, weights };
}

/**
 * Record softmax (batched, no submit)
 */
export async function recordSoftmax(
  recorder: CommandRecorder,
  input: GPUBuffer,
  axis: number,
  options: SoftmaxOptions = {}
): Promise<GPUBuffer> {
  const device = recorder.device;
  const {
    batchSize = 1,
    seqLen = null,
    outputBuffer = null,
  } = options;

  const inferredSeqLen = seqLen || (input.size / (batchSize * 4));
  const pipeline = await createPipeline('softmax', 'default');

  const outputSize = batchSize * inferredSeqLen * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'softmax_output');

  // Uniform buffer
  // WGSL struct: { innerSize: u32, outerSize: u32, temperature: f32, _pad: u32 }
  const uniformBuffer = createUniformBufferWithView(
    'softmax_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredSeqLen, true);  // innerSize at offset 0
      view.setUint32(4, batchSize, true);       // outerSize at offset 4
      view.setFloat32(8, 1.0, true);            // temperature (default 1.0)
    },
    recorder
  );

  // Bind group
  const bindGroup = device.createBindGroup({
    label: 'softmax_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  recordDispatch(recorder, pipeline, bindGroup, batchSize, 'softmax');

  setBufferDtype(output, 'f32');
  return output;
}
