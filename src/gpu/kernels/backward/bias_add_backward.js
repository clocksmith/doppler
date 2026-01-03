import { createBackwardKernel } from './utils.js';

const { run, record } = createBackwardKernel('bias_add_backward', {
  uniformSize: 16,
  writeUniforms: (view, opts) => {
    view.setUint32(0, opts.numTokens, true);
    view.setUint32(4, opts.dim, true);
  },
  calcWorkgroups: (opts) => Math.ceil(opts.dim / 256),
  outputBytes: (opts) => opts.dim * 4,
  outputShape: (opts) => [opts.dim],
  getDevice: true,
  validate: (opts) => {
    if (!opts.numTokens || !opts.dim) throw new Error('bias_add backward requires numTokens and dim');
  },
});

export async function runBiasAddBackward(gradOutput, options = {}) {
  return run(gradOutput, options);
}

export async function recordBiasAddBackward(recorder, gradOutput, options = {}) {
  return record(recorder, gradOutput, options);
}
