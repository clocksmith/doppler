import { getDevice } from '../../device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { createPipeline } from '../pipeline-cache.js';
import { createUniformBufferWithView } from '../uniform-utils.js';
import { castF16ToF32, recordCastF16ToF32 } from '../cast.js';
import { DEFAULT_HIGH_PRECISION_EPS } from '../../../config/schema/index.js';

export async function runRmsNormBackward(input, weight, gradOutput, options = {}) {
  const device = getDevice();
  const {
    numTokens,
    hiddenSize,
    eps = DEFAULT_HIGH_PRECISION_EPS,
    rmsNormWeightOffset = false,
    outputBuffer = null,
  } = options;

  if (!numTokens || !hiddenSize) {
    throw new Error('rmsnorm backward requires numTokens and hiddenSize');
  }

  const outputSize = numTokens * hiddenSize * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'rmsnorm_backward_output');
  let inputTensor = input;
  let weightTensor = weight;
  let gradTensor = gradOutput;
  let uniformBuffer = null;
  let completed = false;
  try {
    inputTensor = input.dtype === 'f16' ? await castF16ToF32(input) : input;
    weightTensor = weight.dtype === 'f16' ? await castF16ToF32(weight) : weight;
    gradTensor = gradOutput.dtype === 'f16' ? await castF16ToF32(gradOutput) : gradOutput;
    const pipeline = await createPipeline(
      'rmsnorm_backward',
      'default',
      null,
      { RMS_NORM_OFFSET: rmsNormWeightOffset }
    );
    uniformBuffer = createUniformBufferWithView(
      'rmsnorm_backward_uniforms',
      16,
      (view) => {
        view.setUint32(0, numTokens, true);
        view.setUint32(4, hiddenSize, true);
        view.setFloat32(8, eps, true);
      },
      null,
      device
    );
    const bindGroup = device.createBindGroup({
      label: 'rmsnorm_backward_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: inputTensor.buffer } },
        { binding: 2, resource: { buffer: weightTensor.buffer } },
        { binding: 3, resource: { buffer: gradTensor.buffer } },
        { binding: 4, resource: { buffer: outputBuf } },
      ],
    });
    dispatch(device, pipeline, bindGroup, numTokens, 'rmsnorm_backward');
    completed = true;
    return createTensor(outputBuf, 'f32', [numTokens, hiddenSize], 'rmsnorm_backward_output');
  } finally {
    uniformBuffer?.destroy();
    if (inputTensor !== input) releaseBuffer(inputTensor.buffer);
    if (weightTensor !== weight) releaseBuffer(weightTensor.buffer);
    if (gradTensor !== gradOutput) releaseBuffer(gradTensor.buffer);
    if (!completed && outputBuffer == null) releaseBuffer(outputBuf);
  }
}

export async function recordRmsNormBackward(recorder, input, weight, gradOutput, options = {}) {
  const device = recorder.device;
  const {
    numTokens,
    hiddenSize,
    eps = DEFAULT_HIGH_PRECISION_EPS,
    rmsNormWeightOffset = false,
    outputBuffer = null,
  } = options;

  if (!numTokens || !hiddenSize) {
    throw new Error('rmsnorm backward requires numTokens and hiddenSize');
  }

  const outputSize = numTokens * hiddenSize * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'rmsnorm_backward_output');
  let inputTensor = input;
  let weightTensor = weight;
  let gradTensor = gradOutput;
  let completed = false;
  try {
    if (input.dtype === 'f16') {
      inputTensor = await recordCastF16ToF32(recorder, input);
      recorder.trackTemporaryBuffer(inputTensor.buffer);
    }
    if (weight.dtype === 'f16') {
      weightTensor = await recordCastF16ToF32(recorder, weight);
      recorder.trackTemporaryBuffer(weightTensor.buffer);
    }
    if (gradOutput.dtype === 'f16') {
      gradTensor = await recordCastF16ToF32(recorder, gradOutput);
      recorder.trackTemporaryBuffer(gradTensor.buffer);
    }
    const pipeline = await createPipeline(
      'rmsnorm_backward',
      'default',
      null,
      { RMS_NORM_OFFSET: rmsNormWeightOffset }
    );
    const uniformBuffer = createUniformBufferWithView(
      'rmsnorm_backward_uniforms',
      16,
      (view) => {
        view.setUint32(0, numTokens, true);
        view.setUint32(4, hiddenSize, true);
        view.setFloat32(8, eps, true);
      },
      recorder
    );
    const bindGroup = device.createBindGroup({
      label: 'rmsnorm_backward_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: inputTensor.buffer } },
        { binding: 2, resource: { buffer: weightTensor.buffer } },
        { binding: 3, resource: { buffer: gradTensor.buffer } },
        { binding: 4, resource: { buffer: outputBuf } },
      ],
    });
    recordDispatch(recorder, pipeline, bindGroup, numTokens, 'rmsnorm_backward');
    completed = true;
    return createTensor(outputBuf, 'f32', [numTokens, hiddenSize], 'rmsnorm_backward_output');
  } finally {
    if (!completed && outputBuffer == null) releaseBuffer(outputBuf);
  }
}
