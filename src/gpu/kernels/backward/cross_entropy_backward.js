import { dtypeBytes } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { createBackwardKernel } from './utils.js';

const { run, record } = createBackwardKernel('cross_entropy_backward', {
  uniformSize: 16,
  writeUniforms: (view, opts) => {
    view.setUint32(0, opts.numTokens, true);
    view.setUint32(4, opts.vocabSize, true);
  },
  calcWorkgroups: (opts) => Math.ceil((opts.numTokens * opts.vocabSize) / WORKGROUP_SIZES.DEFAULT),
  outputBytes: (opts) => opts.numTokens * opts.vocabSize * opts._bytesPerElement,
  outputShape: (opts) => [opts.numTokens, opts.vocabSize],
  dtype: (opts, inputs) => inputs[0].dtype,
  getDevice: true,
  validate: (opts) => {
    if (!opts.numTokens || !opts.vocabSize) throw new Error('cross entropy backward requires numTokens and vocabSize');
  },
});

export async function runCrossEntropyBackward(softmax, targets, gradOutput, options = {}) {
  const bytesPerElement = dtypeBytes(softmax.dtype);
  return run(softmax, targets, gradOutput, { ...options, _bytesPerElement: bytesPerElement });
}

export async function recordCrossEntropyBackward(recorder, softmax, targets, gradOutput, options = {}) {
  const bytesPerElement = dtypeBytes(softmax.dtype);
  return record(recorder, softmax, targets, gradOutput, { ...options, _bytesPerElement: bytesPerElement });
}
