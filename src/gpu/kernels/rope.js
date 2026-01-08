/**
 * RoPE (Rotary Position Embedding) Kernels
 *
 * Provides rotary position embedding with multiple variants:
 * - Standard RoPE
 * - NTK-scaled RoPE
 * - YaRN (Yet another RoPE extensioN)
 */

import { getDevice } from '../device.js';
import { createTensor } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';
import { getKernelThresholds } from '../../config/schema/index.js';

// Get RoPE defaults from schema
const getRopeDefaults = () => getKernelThresholds().rope;

/**
 * Run RoPE operation
 * @param {import('../tensor.js').Tensor} input
 * @param {GPUBuffer} freqsCos
 * @param {GPUBuffer} freqsSin
 * @param {number} seqLen
 * @param {import('./rope.js').RoPEOptions} [options]
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
export async function runRoPE(
  input,
  freqsCos,
  freqsSin,
  seqLen,
  options = {}
) {
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
 * @param {import('../command-recorder.js').CommandRecorder} recorder
 * @param {import('../tensor.js').Tensor} input
 * @param {GPUBuffer} freqsCos
 * @param {GPUBuffer} freqsSin
 * @param {number} seqLen
 * @param {import('./rope.js').RoPEOptions} [options]
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
export async function recordRoPE(
  recorder,
  input,
  freqsCos,
  freqsSin,
  seqLen,
  options = {}
) {
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
