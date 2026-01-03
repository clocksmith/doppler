import { createBackwardKernel } from './utils.js';

const { run, record } = createBackwardKernel('upsample2d_backward', {
  uniformSize: 32,
  writeUniforms: (view, opts) => {
    view.setUint32(0, opts.channels, true);
    view.setUint32(4, opts.inHeight, true);
    view.setUint32(8, opts.inWidth, true);
    view.setUint32(12, opts.outHeight, true);
    view.setUint32(16, opts.outWidth, true);
    view.setUint32(20, opts.scale, true);
  },
  calcWorkgroups: (opts) => Math.ceil((opts.channels * opts.inHeight * opts.inWidth) / 256),
  outputBytes: (opts) => opts.channels * opts.inHeight * opts.inWidth * 4,
  outputShape: (opts) => [opts.channels, opts.inHeight, opts.inWidth],
  getDevice: true,
  validate: (opts) => {
    if (!opts.channels || !opts.inHeight || !opts.inWidth || !opts.outHeight || !opts.outWidth || !opts.scale) {
      throw new Error('upsample2d backward requires channels, inHeight, inWidth, outHeight, outWidth, and scale');
    }
  },
});

export async function runUpsample2DBackward(gradOutput, options = {}) {
  return run(gradOutput, options);
}

export async function recordUpsample2DBackward(recorder, gradOutput, options = {}) {
  return record(recorder, gradOutput, options);
}
