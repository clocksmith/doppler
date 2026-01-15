import { runBackwardKernel, recordBackwardKernel } from './utils.js';

export function runSoftmaxBackward(input, gradOutput, options = {}) {
  const { rows, cols } = options;
  if (!rows || !cols) {
    throw new Error('softmax backward requires rows and cols');
  }
  return runBackwardKernel(
    'softmax_backward',
    input,
    gradOutput,
    16,
    (view) => {
      view.setUint32(0, rows, true);
      view.setUint32(4, cols, true);
    },
    options
  );
}

export function recordSoftmaxBackward(recorder, input, gradOutput, options = {}) {
  const { rows, cols } = options;
  if (!rows || !cols) {
    throw new Error('softmax backward requires rows and cols');
  }
  return recordBackwardKernel(
    recorder,
    'softmax_backward',
    input,
    gradOutput,
    16,
    (view) => {
      view.setUint32(0, rows, true);
      view.setUint32(4, cols, true);
    },
    options
  );
}
