import { runScale, recordScale } from '../scale.js';

export function runScaleBackward(input, gradOutput, options = {}) {
  const { scale } = options;
  if (scale == null) {
    throw new Error('scale backward requires scale');
  }
  return runScale(gradOutput, scale, options);
}

export function recordScaleBackward(recorder, input, gradOutput, options = {}) {
  const { scale } = options;
  if (scale == null) {
    throw new Error('scale backward requires scale');
  }
  return recordScale(recorder, gradOutput, scale, options);
}
