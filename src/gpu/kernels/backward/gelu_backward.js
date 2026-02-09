import { dtypeBytes } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { createBackwardKernel } from './utils.js';

const { run, record } = createBackwardKernel('gelu_backward', {
  uniformSize: 16,
  writeUniforms: (view, opts) => {
    view.setUint32(0, opts._count, true);
  },
  calcWorkgroups: (opts) => Math.ceil(opts._count / WORKGROUP_SIZES.DEFAULT),
  outputBytes: (opts) => opts._count * opts._bytesPerElement,
  outputShape: (opts) => opts._shape,
  dtype: (opts) => opts._dtype,
  getDevice: true,
});

export function runGeluBackward(input, gradOutput, options = {}) {
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

export function recordGeluBackward(recorder, input, gradOutput, options = {}) {
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
