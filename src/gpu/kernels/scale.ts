/**
 * Scale kernel - multiply each element by a scalar factor
 * Used for embedding scaling in Gemma models (sqrt(hidden_size))
 */

import { getDevice } from '../device.js';
import { setBufferDtype } from '../buffer-dtypes.js';
import { acquireBuffer } from '../buffer-pool.js';
import type { CommandRecorder } from '../command-recorder.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import type { OutputBufferOptions } from './types.js';

/** Scale kernel options */
export interface ScaleOptions extends OutputBufferOptions {
  /** Number of elements to scale (inferred from buffer size if not provided) */
  count?: number;
  /** Whether to scale in-place (output = input) */
  inplace?: boolean;
}

/**
 * Run scale operation: output = input * scale
 */
export async function runScale(
  input: GPUBuffer,
  scale: number,
  options: ScaleOptions = {}
): Promise<GPUBuffer> {
  const device = getDevice();
  const { count, outputBuffer = null, inplace = false } = options;

  const inferredCount = count ?? Math.floor(input.size / 4);
  const variant = inplace ? 'inplace' : 'default';
  const pipeline = await createPipeline('scale', variant);

  const outputSize = inferredCount * 4;
  const output = inplace ? input : (outputBuffer || acquireBuffer(outputSize, undefined, 'scale_output'));

  // Create uniform buffer (16 bytes to match WGSL struct with padding)
  const uniformBuffer = createUniformBufferWithView(
    'scale_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredCount, true);
      view.setFloat32(4, scale, true);
      // _pad0 and _pad1 at offsets 8 and 12 (unused)
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'scale_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'scale');

  uniformBuffer.destroy();
  setBufferDtype(output, 'f32');

  return output;
}

/**
 * Record scale operation (batched, no submit)
 */
export async function recordScale(
  recorder: CommandRecorder,
  input: GPUBuffer,
  scale: number,
  options: ScaleOptions = {}
): Promise<GPUBuffer> {
  const device = recorder.device;
  const { count, outputBuffer = null, inplace = false } = options;

  const inferredCount = count ?? Math.floor(input.size / 4);
  const variant = inplace ? 'inplace' : 'default';
  const pipeline = await createPipeline('scale', variant);

  const outputSize = inferredCount * 4;
  const output = inplace ? input : (outputBuffer || acquireBuffer(outputSize, undefined, 'scale_output'));

  // Create uniform buffer via recorder (tracked for cleanup, 16 bytes to match WGSL)
  const uniformBuffer = createUniformBufferWithView(
    'scale_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredCount, true);
      view.setFloat32(4, scale, true);
      // _pad0 and _pad1 at offsets 8 and 12 (unused)
    },
    recorder
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'scale_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'scale');

  setBufferDtype(output, 'f32');

  return output;
}
