import { dtypeBytes } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { createBackwardKernel } from './utils.js';

const { run, record } = createBackwardKernel('scale_backward', {
  uniformSize: 16,
  writeUniforms: (view, opts) => {
    view.setUint32(0, opts._count, true);
    view.setFloat32(4, opts.scale, true);
  },
  calcWorkgroups: (opts) => Math.ceil(opts._count / WORKGROUP_SIZES.DEFAULT),
  outputBytes: (opts) => opts._count * opts._bytesPerElement,
  outputShape: (opts) => opts._shape,
  dtype: (opts) => opts._dtype,
  getDevice: true,
  validate: (opts) => {
    if (opts.scale == null) throw new Error('scale backward requires scale');
  },
});

export async function runScaleBackward(input, gradOutput, options = {}) {
  const bytesPerElement = dtypeBytes(gradOutput.dtype);
  const count = options.count ?? Math.floor(gradOutput.buffer.size / bytesPerElement);
  return run(input, gradOutput, {
    ...options,
    _count: count,
    _bytesPerElement: bytesPerElement,
    _shape: [...gradOutput.shape],
    _dtype: gradOutput.dtype,
  });
}

export async function recordScaleBackward(recorder, input, gradOutput, options = {}) {
  const bytesPerElement = dtypeBytes(gradOutput.dtype);
  const count = options.count ?? Math.floor(gradOutput.buffer.size / bytesPerElement);
  return record(recorder, input, gradOutput, {
    ...options,
    _count: count,
    _bytesPerElement: bytesPerElement,
    _shape: [...gradOutput.shape],
    _dtype: gradOutput.dtype,
  });
}
