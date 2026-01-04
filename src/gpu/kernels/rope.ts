/**
 * RoPE (Rotary Position Embedding) Kernels
 *
 * Provides rotary position embedding with multiple variants:
 * - Standard RoPE
 * - NTK-scaled RoPE
 * - YaRN (Yet another RoPE extensioN)
 */

import { getDevice } from '../device.js';
import type { CommandRecorder } from '../command-recorder.js';
import { Tensor, createTensor } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';
import type { OutputBufferOptions } from './types.js';
import { getKernelThresholds } from '../../config/schema/index.js';

// Get RoPE defaults from schema
const getRopeDefaults = () => getKernelThresholds().rope;

/** RoPE kernel options */
export interface RoPEOptions extends OutputBufferOptions {
  numHeads?: number;
  headDim?: number;
  ropeTheta?: number;
  startPos?: number;
}

/**
 * Run RoPE operation
 */
export async function runRoPE(
  input: Tensor,
  freqsCos: GPUBuffer,
  freqsSin: GPUBuffer,
  seqLen: number,
  options: RoPEOptions = {}
): Promise<Tensor> {
  const device = getDevice();
  const ropeDefaults = getRopeDefaults();
  const {
    numHeads = 1,
    headDim = 64,
    ropeTheta = ropeDefaults.defaultTheta,
  } = options;

  const pipeline = await getPipelineFast('rope', 'default');

  // Note: RoPE shader modifies input in-place (no output buffer)

  // Create uniform buffer (size from schema to match WGSL struct)
  // struct RoPEUniforms { seqLen, numHeads, headDim, startPos, ropeBase, ropeScale, _pad0, _pad1 }
  const uniformBuffer = createUniformBufferWithView(
    'rope_uniforms',
    ropeDefaults.uniformSize,
    (view) => {
      view.setUint32(0, seqLen, true);          // seqLen
      view.setUint32(4, numHeads, true);        // numHeads
      view.setUint32(8, headDim, true);         // headDim
      view.setUint32(12, options.startPos || 0, true);  // startPos
      view.setFloat32(16, ropeTheta, true);     // ropeBase
      view.setFloat32(20, 1.0, true);           // ropeScale (default 1.0)
    },
    null,
    device
  );

  // Create bind group (only 4 bindings - shader modifies input in-place)
  const bindGroup = device.createBindGroup({
    label: 'rope_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: freqsCos } },
      { binding: 3, resource: { buffer: freqsSin } },
    ],
  });

  // Dispatch
  if (headDim % 2 !== 0) {
    throw new Error(`RoPE headDim must be even, got ${headDim}`);
  }
  const halfDim = headDim / 2;
  const workgroups = Math.ceil((seqLen * numHeads * halfDim) / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'rope');

  uniformBuffer.destroy();

  // Return tensor wrapping input buffer (modified in-place by shader)
  return createTensor(input.buffer, input.dtype, [...input.shape], 'rope_output');
}

/**
 * Record RoPE (batched, no submit)
 */
export async function recordRoPE(
  recorder: CommandRecorder,
  input: Tensor,
  freqsCos: GPUBuffer,
  freqsSin: GPUBuffer,
  seqLen: number,
  options: RoPEOptions = {}
): Promise<Tensor> {
  const device = recorder.device;
  const ropeDefaults = getRopeDefaults();
  const {
    numHeads = 1,
    headDim = 64,
    ropeTheta = ropeDefaults.defaultTheta,
  } = options;

  const pipeline = await getPipelineFast('rope', 'default');

  // Note: RoPE shader modifies input in-place (no output buffer)

  // Uniform buffer (size from schema to match WGSL struct)
  // struct RoPEUniforms { seqLen, numHeads, headDim, startPos, ropeBase, ropeScale, _pad0, _pad1 }
  const uniformBuffer = createUniformBufferWithView(
    'rope_uniforms',
    ropeDefaults.uniformSize,
    (view) => {
      view.setUint32(0, seqLen, true);          // seqLen
      view.setUint32(4, numHeads, true);        // numHeads
      view.setUint32(8, headDim, true);         // headDim
      view.setUint32(12, options.startPos || 0, true);  // startPos
      view.setFloat32(16, ropeTheta, true);     // ropeBase from options or schema default
      view.setFloat32(20, 1.0, true);           // ropeScale (default 1.0)
    },
    recorder
  );

  // Bind group (only 4 bindings - shader modifies input in-place)
  const bindGroup = device.createBindGroup({
    label: 'rope_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: freqsCos } },
      { binding: 3, resource: { buffer: freqsSin } },
    ],
  });

  if (headDim % 2 !== 0) {
    throw new Error(`RoPE headDim must be even, got ${headDim}`);
  }
  const halfDim = headDim / 2;
  const workgroups = Math.ceil((seqLen * numHeads * halfDim) / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'rope');

  // Return tensor wrapping input buffer (modified in-place by shader)
  return createTensor(input.buffer, input.dtype, [...input.shape], 'rope_output');
}
