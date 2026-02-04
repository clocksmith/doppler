import { runScale, recordScale } from '../scale.js';

export function runEmbedBackward(input, gradOutput, options = {}) {
  return runScale(gradOutput, 1.0, options);
}

export function recordEmbedBackward(recorder, input, gradOutput, options = {}) {
  return recordScale(recorder, gradOutput, 1.0, options);
}
