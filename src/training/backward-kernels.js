import { log } from '../debug/index.js';

function arraysEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export function validateBackwardInputShapes(forwardInputShape, backwardInputShape, opName) {
  if (!Array.isArray(forwardInputShape) || !Array.isArray(backwardInputShape)) {
    log.warn('Training', `${opName}: cannot validate shapes — one or both shapes are not arrays`);
    return false;
  }
  if (!arraysEqual(forwardInputShape, backwardInputShape)) {
    log.warn(
      'Training',
      `${opName}: backward input shape [${backwardInputShape}] does not match forward input shape [${forwardInputShape}]`
    );
    return false;
  }
  return true;
}

export function validateGradOutputShape(outputShape, gradOutputShape, opName) {
  if (!Array.isArray(outputShape) || !Array.isArray(gradOutputShape)) {
    log.warn('Training', `${opName}: cannot validate grad output shape — one or both shapes are not arrays`);
    return false;
  }
  if (!arraysEqual(outputShape, gradOutputShape)) {
    log.warn(
      'Training',
      `${opName}: grad output shape [${gradOutputShape}] does not match forward output shape [${outputShape}]`
    );
    return false;
  }
  return true;
}
