

import { getDevice, getKernelCapabilities } from '../device.js';
import { createTensor } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';
import { getKernelThresholds } from '../../config/schema/index.js';
import { selectByRules } from './rule-matcher.js';

// Get RoPE defaults from schema
const getRopeDefaults = () => getKernelThresholds().rope;


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

  const caps = getKernelCapabilities();
  const useF16 = input.dtype === 'f16' && caps.hasF16;
  const variant = selectByRules(
    [
      { match: { useF16: true }, value: 'default_f16' },
      { match: {}, value: 'default' },
    ],
    { useF16 }
  );
  const pipeline = await getPipelineFast('rope', variant);

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

  const caps = getKernelCapabilities();
  const useF16 = input.dtype === 'f16' && caps.hasF16;
  const variant = selectByRules(
    [
      { match: { useF16: true }, value: 'default_f16' },
      { match: {}, value: 'default' },
    ],
    { useF16 }
  );
  const pipeline = await getPipelineFast('rope', variant);

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
