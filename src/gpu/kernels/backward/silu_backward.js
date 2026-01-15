import { runBackwardKernel, recordBackwardKernel } from './utils.js';

export function runSiluBackward(input, gradOutput, options = {}) {
  return runBackwardKernel(
    'silu_backward',
    input,
    gradOutput,
    16,
    (view, count) => {
      view.setUint32(0, count, true);
    },
    options
  );
}

export function recordSiluBackward(recorder, input, gradOutput, options = {}) {
  return recordBackwardKernel(
    recorder,
    'silu_backward',
    input,
    gradOutput,
    16,
    (view, count) => {
      view.setUint32(0, count, true);
    },
    options
  );
}
