import { createTensor } from '../../tensor.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { dispatch, recordDispatch } from '../dispatch.js';

export async function runConv2DBackward(input, weight, gradOutput, options = {}) {
  const { inChannels, outChannels, height, width, outHeight, outWidth, kernelH, kernelW, stride, pad, computeGradInput = true, computeGradWeight = true } = options;

  let gradInput = null;
  let gradWeight = null;

  const device = gradOutput.buffer.device;

  const uniformBuffer = createUniformBufferWithView(
    'conv2d_backward_uniforms',
    48,
    (view) => {
      view.setUint32(0, inChannels, true);
      view.setUint32(4, outChannels, true);
      view.setUint32(8, height, true);
      view.setUint32(12, width, true);
      view.setUint32(16, outHeight, true);
      view.setUint32(20, outWidth, true);
      view.setUint32(24, kernelH, true);
      view.setUint32(28, kernelW, true);
      view.setUint32(32, stride, true);
      view.setUint32(36, pad, true);
    },
    null,
    device
  );

  if (computeGradInput) {
    const outputSize = inChannels * height * width * 4;
    const outputBuf = acquireBuffer(outputSize, undefined, 'conv2d_backward_input_output');
    const pipeline = await createPipeline('conv2d_backward_input', 'default');
    const bindGroup = device.createBindGroup({
      label: 'conv2d_backward_input_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: gradOutput.buffer } },
        { binding: 2, resource: { buffer: weight.buffer } },
        { binding: 3, resource: { buffer: outputBuf } },
      ],
    });
    dispatch(device, pipeline, bindGroup, Math.ceil((inChannels * height * width) / 256), 'conv2d_backward_input');
    gradInput = createTensor(outputBuf, 'f32', [inChannels, height, width], 'conv2d_grad_input');
  }

  if (computeGradWeight) {
    const outputSize = outChannels * inChannels * kernelH * kernelW * 4;
    const outputBuf = acquireBuffer(outputSize, undefined, 'conv2d_backward_weight_output');
    const pipeline = await createPipeline('conv2d_backward_weight', 'default');
    const bindGroup = device.createBindGroup({
      label: 'conv2d_backward_weight_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: gradOutput.buffer } },
        { binding: 2, resource: { buffer: input.buffer } },
        { binding: 3, resource: { buffer: outputBuf } },
      ],
    });
    dispatch(device, pipeline, bindGroup, Math.ceil((outChannels * inChannels * kernelH * kernelW) / 256), 'conv2d_backward_weight');
    gradWeight = createTensor(outputBuf, 'f32', [outChannels, inChannels, kernelH, kernelW], 'conv2d_grad_weight');
  }

  uniformBuffer.destroy();
  return { gradInput, gradWeight };
}

export async function recordConv2DBackward(recorder, input, weight, gradOutput, options = {}) {
  const { inChannels, outChannels, height, width, outHeight, outWidth, kernelH, kernelW, stride, pad, computeGradInput = true, computeGradWeight = true } = options;

  let gradInput = null;
  let gradWeight = null;

  const uniformBuffer = createUniformBufferWithView(
    'conv2d_backward_uniforms',
    48,
    (view) => {
      view.setUint32(0, inChannels, true);
      view.setUint32(4, outChannels, true);
      view.setUint32(8, height, true);
      view.setUint32(12, width, true);
      view.setUint32(16, outHeight, true);
      view.setUint32(20, outWidth, true);
      view.setUint32(24, kernelH, true);
      view.setUint32(28, kernelW, true);
      view.setUint32(32, stride, true);
      view.setUint32(36, pad, true);
    },
    recorder
  );

  if (computeGradInput) {
    const outputSize = inChannels * height * width * 4;
    const outputBuf = acquireBuffer(outputSize, undefined, 'conv2d_backward_input_output');
    const pipeline = await createPipeline('conv2d_backward_input', 'default');
    const bindGroup = recorder.device.createBindGroup({
      label: 'conv2d_backward_input_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: gradOutput.buffer } },
        { binding: 2, resource: { buffer: weight.buffer } },
        { binding: 3, resource: { buffer: outputBuf } },
      ],
    });
    recordDispatch(recorder, pipeline, bindGroup, Math.ceil((inChannels * height * width) / 256), 'conv2d_backward_input');
    gradInput = createTensor(outputBuf, 'f32', [inChannels, height, width], 'conv2d_grad_input');
  }

  if (computeGradWeight) {
    const outputSize = outChannels * inChannels * kernelH * kernelW * 4;
    const outputBuf = acquireBuffer(outputSize, undefined, 'conv2d_backward_weight_output');
    const pipeline = await createPipeline('conv2d_backward_weight', 'default');
    const bindGroup = recorder.device.createBindGroup({
      label: 'conv2d_backward_weight_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: gradOutput.buffer } },
        { binding: 2, resource: { buffer: input.buffer } },
        { binding: 3, resource: { buffer: outputBuf } },
      ],
    });
    recordDispatch(recorder, pipeline, bindGroup, Math.ceil((outChannels * inChannels * kernelH * kernelW) / 256), 'conv2d_backward_weight');
    gradWeight = createTensor(outputBuf, 'f32', [outChannels, inChannels, kernelH, kernelW], 'conv2d_grad_weight');
  }

  return { gradInput, gradWeight };
}
