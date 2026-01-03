import { createBackwardKernel } from './utils.js';

const { run, record } = createBackwardKernel('pixel_shuffle_backward', {
  uniformSize: 32,
  writeUniforms: (view, opts) => {
    view.setUint32(0, opts.outChannels, true);
    view.setUint32(4, opts.outHeight, true);
    view.setUint32(8, opts.outWidth, true);
    view.setUint32(12, opts.gridWidth, true);
    view.setUint32(16, opts.gridHeight, true);
    view.setUint32(20, opts.patchSize, true);
    view.setUint32(24, opts.patchChannels, true);
  },
  calcWorkgroups: (opts) => Math.ceil((opts.gridWidth * opts.gridHeight * opts.patchChannels) / 256),
  outputBytes: (opts) => opts.gridWidth * opts.gridHeight * opts.patchChannels * 4,
  outputShape: (opts) => [opts.gridHeight * opts.gridWidth, opts.patchChannels],
  validate: (opts) => {
    if (!opts.outChannels || !opts.outHeight || !opts.outWidth || !opts.gridWidth || !opts.gridHeight || !opts.patchSize || !opts.patchChannels) {
      throw new Error('pixel_shuffle backward requires all dimensions');
    }
  },
});

export async function runPixelShuffleBackward(gradOutput, options = {}) {
  return run(gradOutput, options);
}

export async function recordPixelShuffleBackward(recorder, gradOutput, options = {}) {
  return record(recorder, gradOutput, options);
}
