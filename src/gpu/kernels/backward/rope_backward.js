import { getDevice } from '../../device.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { createPipeline } from '../pipeline-cache.js';
import { createUniformBufferWithView } from '../uniform-utils.js';

function validateInputs(gradOutput, freqsCos, freqsSin, geometry) {
  const { seqLen, numHeads, headDim, rotaryDim, pairSpanDim } = geometry;
  if (!Number.isInteger(seqLen) || seqLen < 1
    || !Number.isInteger(numHeads) || numHeads < 1
    || !Number.isInteger(headDim) || headDim < 2 || headDim % 2 !== 0) {
    throw new Error('rope backward requires positive seqLen/numHeads and even headDim.');
  }
  if (!Number.isInteger(rotaryDim) || rotaryDim < 2 || rotaryDim > headDim || rotaryDim % 2 !== 0) {
    throw new Error('rope backward requires even rotaryDim within headDim.');
  }
  if (!Number.isInteger(pairSpanDim)
    || pairSpanDim < rotaryDim
    || pairSpanDim > headDim
    || pairSpanDim % 2 !== 0) {
    throw new Error('rope backward requires even pairSpanDim within [rotaryDim, headDim].');
  }
  for (const [label, tensor] of [
    ['gradOutput', gradOutput],
    ['freqsCos', freqsCos],
    ['freqsSin', freqsSin],
  ]) {
    if (tensor?.dtype !== 'f32') throw new Error(`rope backward requires f32 ${label}.`);
  }
}

export async function runRoPEBackward(gradOutput, freqsCos, freqsSin, options = {}) {
  const device = getDevice();
  const {
    seqLen,
    numHeads,
    headDim,
    startPos = 0,
    rotaryDim = headDim,
    pairSpanDim = rotaryDim,
    interleaved = false,
    outputBuffer = null,
  } = options;

  validateInputs(gradOutput, freqsCos, freqsSin, {
    seqLen, numHeads, headDim, rotaryDim, pairSpanDim,
  });

  const bytesPerElement = dtypeBytes(gradOutput.dtype);
  const outputSize = seqLen * numHeads * headDim * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'rope_backward_output');

  const pipeline = await createPipeline('rope_backward', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'rope_backward_uniforms',
    32,
    (view) => {
      view.setUint32(0, seqLen, true);
      view.setUint32(4, numHeads, true);
      view.setUint32(8, headDim, true);
      view.setUint32(12, startPos, true);
      view.setUint32(16, rotaryDim, true);
      view.setUint32(20, pairSpanDim, true);
      view.setUint32(24, interleaved ? 1 : 0, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'rope_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gradOutput.buffer } },
      { binding: 2, resource: { buffer: freqsCos.buffer } },
      { binding: 3, resource: { buffer: freqsSin.buffer } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const totalElements = seqLen * numHeads * headDim;
  const workgroups = Math.ceil(totalElements / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'rope_backward');

  uniformBuffer.destroy();

  return createTensor(outputBuf, gradOutput.dtype, [seqLen, numHeads, headDim], 'rope_backward_output');
}

export async function recordRoPEBackward(recorder, gradOutput, freqsCos, freqsSin, options = {}) {
  const device = recorder.device;
  const {
    seqLen,
    numHeads,
    headDim,
    startPos = 0,
    rotaryDim = headDim,
    pairSpanDim = rotaryDim,
    interleaved = false,
    outputBuffer = null,
  } = options;

  validateInputs(gradOutput, freqsCos, freqsSin, {
    seqLen, numHeads, headDim, rotaryDim, pairSpanDim,
  });

  const bytesPerElement = dtypeBytes(gradOutput.dtype);
  const outputSize = seqLen * numHeads * headDim * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'rope_backward_output');

  const pipeline = await createPipeline('rope_backward', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'rope_backward_uniforms',
    32,
    (view) => {
      view.setUint32(0, seqLen, true);
      view.setUint32(4, numHeads, true);
      view.setUint32(8, headDim, true);
      view.setUint32(12, startPos, true);
      view.setUint32(16, rotaryDim, true);
      view.setUint32(20, pairSpanDim, true);
      view.setUint32(24, interleaved ? 1 : 0, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'rope_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gradOutput.buffer } },
      { binding: 2, resource: { buffer: freqsCos.buffer } },
      { binding: 3, resource: { buffer: freqsSin.buffer } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const totalElements = seqLen * numHeads * headDim;
  const workgroups = Math.ceil(totalElements / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'rope_backward');

  return createTensor(outputBuf, gradOutput.dtype, [seqLen, numHeads, headDim], 'rope_backward_output');
}
