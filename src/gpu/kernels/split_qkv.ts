/**
 * Split QKV Kernel
 *
 * Splits fused QKV projection output into separate Q, K, V buffers.
 * Used for 3→1 matmul optimization in attention.
 */

import { getDevice } from '../device.js';
import { acquireBuffer } from '../buffer-pool.js';
import type { CommandRecorder } from '../command-recorder.js';
import { createTensor, type Tensor, type TensorDtype, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';

/** Split QKV options */
export interface SplitQKVOptions {
  numTokens: number;
  qSize: number;  // numHeads * headDim
  kSize: number;  // numKVHeads * headDim
  vSize: number;  // numKVHeads * headDim
  /** Pre-allocated Q output tensor */
  qTensor?: Tensor | null;
  /** Pre-allocated K output tensor */
  kTensor?: Tensor | null;
  /** Pre-allocated V output tensor */
  vTensor?: Tensor | null;
}

/** Split QKV result */
export interface SplitQKVResult {
  Q: Tensor;
  K: Tensor;
  V: Tensor;
}

/**
 * Split fused QKV output into separate Q, K, V tensors.
 *
 * @param qkvTensor - Fused QKV output [numTokens, qSize + kSize + vSize]
 * @param options - Split configuration
 * @returns Separate Q, K, V tensors
 */
export async function runSplitQKV(
  qkvTensor: Tensor,
  options: SplitQKVOptions
): Promise<SplitQKVResult> {
  const device = getDevice();
  const { numTokens, qSize, kSize, vSize, qTensor = null, kTensor = null, vTensor = null } = options;

  const pipeline = await getPipelineFast('split_qkv', 'default');

  const outputDtype: TensorDtype = qkvTensor.dtype;
  const bytesPerElement = dtypeBytes(outputDtype);

  // Allocate output buffers if not provided
  const qBuffer = qTensor?.buffer || acquireBuffer(numTokens * qSize * bytesPerElement, undefined, 'Q');
  const kBuffer = kTensor?.buffer || acquireBuffer(numTokens * kSize * bytesPerElement, undefined, 'K');
  const vBuffer = vTensor?.buffer || acquireBuffer(numTokens * vSize * bytesPerElement, undefined, 'V');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'split_qkv_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, qSize, true);
      view.setUint32(8, kSize, true);
      view.setUint32(12, vSize, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'split_qkv_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: qkvTensor.buffer } },
      { binding: 2, resource: { buffer: qBuffer } },
      { binding: 3, resource: { buffer: kBuffer } },
      { binding: 4, resource: { buffer: vBuffer } },
    ],
  });

  // Dispatch - total elements across all outputs
  const totalElements = numTokens * (qSize + kSize + vSize);
  const workgroups = Math.ceil(totalElements / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'split_qkv');

  uniformBuffer.destroy();

  const Q = qTensor || createTensor(qBuffer, outputDtype, [numTokens, qSize], 'Q');
  const K = kTensor || createTensor(kBuffer, outputDtype, [numTokens, kSize], 'K');
  const V = vTensor || createTensor(vBuffer, outputDtype, [numTokens, vSize], 'V');

  return { Q, K, V };
}

/**
 * Record split QKV (batched, no submit).
 */
export async function recordSplitQKV(
  recorder: CommandRecorder,
  qkvTensor: Tensor,
  options: SplitQKVOptions
): Promise<SplitQKVResult> {
  const device = recorder.device;
  const { numTokens, qSize, kSize, vSize, qTensor = null, kTensor = null, vTensor = null } = options;

  const pipeline = await getPipelineFast('split_qkv', 'default');

  const outputDtype: TensorDtype = qkvTensor.dtype;
  const bytesPerElement = dtypeBytes(outputDtype);

  // Allocate output buffers if not provided
  const qBuffer = qTensor?.buffer || acquireBuffer(numTokens * qSize * bytesPerElement, undefined, 'Q');
  const kBuffer = kTensor?.buffer || acquireBuffer(numTokens * kSize * bytesPerElement, undefined, 'K');
  const vBuffer = vTensor?.buffer || acquireBuffer(numTokens * vSize * bytesPerElement, undefined, 'V');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'split_qkv_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, qSize, true);
      view.setUint32(8, kSize, true);
      view.setUint32(12, vSize, true);
    },
    recorder
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'split_qkv_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: qkvTensor.buffer } },
      { binding: 2, resource: { buffer: qBuffer } },
      { binding: 3, resource: { buffer: kBuffer } },
      { binding: 4, resource: { buffer: vBuffer } },
    ],
  });

  // Dispatch
  const totalElements = numTokens * (qSize + kSize + vSize);
  const workgroups = Math.ceil(totalElements / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'split_qkv');

  const Q = qTensor || createTensor(qBuffer, outputDtype, [numTokens, qSize], 'Q');
  const K = kTensor || createTensor(kBuffer, outputDtype, [numTokens, kSize], 'K');
  const V = vTensor || createTensor(vBuffer, outputDtype, [numTokens, vSize], 'V');

  return { Q, K, V };
}
