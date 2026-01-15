import { runBackwardKernel, recordBackwardKernel } from './utils.js';

export function runGeluBackward(input, gradOutput, options = {}) {
  return runBackwardKernel(
    'gelu_backward',
    input,
    gradOutput,
    16,
    (view, count) => {
      view.setUint32(0, count, true);
    },
    options
  );
}

export function recordGeluBackward(recorder, input, gradOutput, options = {}) {
  return recordBackwardKernel(
    recorder,
    'gelu_backward',
    input,
    gradOutput,
    16,
    (view, count) => {
      view.setUint32(0, count, true);
    },
    options
  );
}
