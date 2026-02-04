import { runBackwardKernel, recordBackwardKernel } from './utils.js';

export function runScaleBackward(input, gradOutput, options = {}) {
  const { scale } = options;
  if (scale == null) {
    throw new Error('scale backward requires scale');
  }
  return runBackwardKernel(
    'scale_backward',
    input,
    gradOutput,
    16,
    (view, count) => {
      view.setUint32(0, count, true);
      view.setFloat32(4, scale, true);
    },
    options
  );
}

export function recordScaleBackward(recorder, input, gradOutput, options = {}) {
  const { scale } = options;
  if (scale == null) {
    throw new Error('scale backward requires scale');
  }
  return recordBackwardKernel(
    recorder,
    'scale_backward',
    input,
    gradOutput,
    16,
    (view, count) => {
      view.setUint32(0, count, true);
      view.setFloat32(4, scale, true);
    },
    options
  );
}
