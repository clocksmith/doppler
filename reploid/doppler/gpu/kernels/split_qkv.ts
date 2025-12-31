/**
 * Split QKV Kernel
 *
 * Splits fused QKV projection output into separate Q, K, V buffers.
 * Used for 3→1 matmul optimization in attention.
 */

import { getDevice } from '../device.js';
import { setBufferDtype } from '../buffer-dtypes.js';
import { acquireBuffer } from '../buffer-pool.js';
import type { CommandRecorder } from '../command-recorder.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';

/** Split QKV options */
export interface SplitQKVOptions {
  numTokens: number;
  qSize: number;  // numHeads * headDim
  kSize: number;  // numKVHeads * headDim
  vSize: number;  // numKVHeads * headDim
  /** Pre-allocated Q output buffer */
  qBuffer?: GPUBuffer | null;
  /** Pre-allocated K output buffer */
  kBuffer?: GPUBuffer | null;
  /** Pre-allocated V output buffer */
  vBuffer?: GPUBuffer | null;
}

/** Split QKV result */
export interface SplitQKVResult {
  Q: GPUBuffer;
  K: GPUBuffer;
  V: GPUBuffer;
}

/**
 * Split fused QKV output into separate Q, K, V buffers.
 *
 * @param qkvBuffer - Fused QKV output [numTokens, qSize + kSize + vSize]
 * @param options - Split configuration
 * @returns Separate Q, K, V buffers
 */
export async function runSplitQKV(
  qkvBuffer: GPUBuffer,
  options: SplitQKVOptions
): Promise<SplitQKVResult> {
  const device = getDevice();
  const { numTokens, qSize, kSize, vSize, qBuffer = null, kBuffer = null, vBuffer = null } = options;

  const pipeline = await getPipelineFast('split_qkv', 'default');

  // Allocate output buffers if not provided
  const Q = qBuffer || acquireBuffer(numTokens * qSize * 4, undefined, 'Q');
  const K = kBuffer || acquireBuffer(numTokens * kSize * 4, undefined, 'K');
  const V = vBuffer || acquireBuffer(numTokens * vSize * 4, undefined, 'V');

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
      { binding: 1, resource: { buffer: qkvBuffer } },
      { binding: 2, resource: { buffer: Q } },
      { binding: 3, resource: { buffer: K } },
      { binding: 4, resource: { buffer: V } },
    ],
  });

  // Dispatch - total elements across all outputs
  const totalElements = numTokens * (qSize + kSize + vSize);
  const workgroups = Math.ceil(totalElements / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'split_qkv');

  uniformBuffer.destroy();

  setBufferDtype(Q, 'f32');
  setBufferDtype(K, 'f32');
  setBufferDtype(V, 'f32');

  return { Q, K, V };
}

/**
 * Record split QKV (batched, no submit).
 */
export async function recordSplitQKV(
  recorder: CommandRecorder,
  qkvBuffer: GPUBuffer,
  options: SplitQKVOptions
): Promise<SplitQKVResult> {
  const device = recorder.device;
  const { numTokens, qSize, kSize, vSize, qBuffer = null, kBuffer = null, vBuffer = null } = options;

  const pipeline = await getPipelineFast('split_qkv', 'default');

  // Allocate output buffers if not provided
  const Q = qBuffer || acquireBuffer(numTokens * qSize * 4, undefined, 'Q');
  const K = kBuffer || acquireBuffer(numTokens * kSize * 4, undefined, 'K');
  const V = vBuffer || acquireBuffer(numTokens * vSize * 4, undefined, 'V');

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
      { binding: 1, resource: { buffer: qkvBuffer } },
      { binding: 2, resource: { buffer: Q } },
      { binding: 3, resource: { buffer: K } },
      { binding: 4, resource: { buffer: V } },
    ],
  });

  // Dispatch
  const totalElements = numTokens * (qSize + kSize + vSize);
  const workgroups = Math.ceil(totalElements / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'split_qkv');

  setBufferDtype(Q, 'f32');
  setBufferDtype(K, 'f32');
  setBufferDtype(V, 'f32');

  return { Q, K, V };
}
