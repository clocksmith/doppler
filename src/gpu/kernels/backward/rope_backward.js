import { getDevice } from '../../device.js';
import { acquireBuffer } from '../../buffer-pool.js';
import { createTensor, dtypeBytes } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';

export async function runRoPEBackward(gradOutput, freqsCos, freqsSin, options = {}) {
  const device = getDevice();
  const {
    seqLen,
    numHeads,
    headDim,
    startPos = 0,
    outputBuffer = null,
  } = options;

  if (!seqLen || !numHeads || !headDim) {
    throw new Error('rope backward requires seqLen, numHeads, and headDim');
  }

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

  const totalPairs = seqLen * numHeads * (headDim / 2);
  const workgroups = Math.ceil(totalPairs / WORKGROUP_SIZES.DEFAULT);
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
    outputBuffer = null,
  } = options;

  if (!seqLen || !numHeads || !headDim) {
    throw new Error('rope backward requires seqLen, numHeads, and headDim');
  }

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

  const totalPairs = seqLen * numHeads * (headDim / 2);
  const workgroups = Math.ceil(totalPairs / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'rope_backward');

  return createTensor(outputBuf, gradOutput.dtype, [seqLen, numHeads, headDim], 'rope_backward_output');
}
