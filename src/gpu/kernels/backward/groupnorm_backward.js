import { createBackwardKernel } from './utils.js';

const { run, record } = createBackwardKernel('groupnorm_backward', {
  uniformSize: 32,
  writeUniforms: (view, opts) => {
    view.setUint32(0, opts.channels, true);
    view.setUint32(4, opts.height, true);
    view.setUint32(8, opts.width, true);
    view.setUint32(12, opts.numGroups, true);
    view.setFloat32(16, opts.eps ?? 1e-5, true);
  },
  calcWorkgroups: (opts) => opts.numGroups,
  outputBytes: (opts) => opts.channels * opts.height * opts.width * 4,
  outputShape: (opts) => [opts.channels, opts.height, opts.width],
  validate: (opts) => {
    if (!opts.channels || !opts.height || !opts.width || !opts.numGroups) {
      throw new Error('groupnorm backward requires all dimensions');
    }
  },
});

export async function runGroupNormBackward(input, weight, gradOutput, options = {}) {
  return run(input, weight, gradOutput, options);
}

export async function recordGroupNormBackward(recorder, input, weight, gradOutput, options = {}) {
  return record(recorder, input, weight, gradOutput, options);
}
