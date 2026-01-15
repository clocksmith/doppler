import { runBackwardKernel, recordBackwardKernel } from './utils.js';

export function runEmbedBackward(input, gradOutput, options = {}) {
  return runBackwardKernel(
    'embed_backward',
    input,
    gradOutput,
    16,
    (view, count) => {
      view.setUint32(0, count, true);
    },
    options
  );
}

export function recordEmbedBackward(recorder, input, gradOutput, options = {}) {
  return recordBackwardKernel(
    recorder,
    'embed_backward',
    input,
    gradOutput,
    16,
    (view, count) => {
      view.setUint32(0, count, true);
    },
    options
  );
}
