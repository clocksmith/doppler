


export function createTensor(
  buffer,
  dtype,
  shape,
  label
) {
  return {
    buffer,
    dtype,
    shape: Object.freeze([...shape]),
    label,
  };
}


export function assertDtype(
  tensor,
  expected,
  operation
) {
  if (tensor.dtype !== expected) {
    throw new Error(
      `${operation}: expected ${expected} tensor, got ${tensor.dtype}` +
      (tensor.label ? ` (${tensor.label})` : '')
    );
  }
}


export function assertShape(
  tensor,
  expected,
  operation
) {
  if (tensor.shape.length !== expected.length) {
    throw new Error(
      `${operation}: expected ${expected.length}D tensor, got ${tensor.shape.length}D` +
      (tensor.label ? ` (${tensor.label})` : '')
    );
  }
  for (let i = 0; i < expected.length; i++) {
    if (expected[i] !== -1 && tensor.shape[i] !== expected[i]) {
      throw new Error(
        `${operation}: shape mismatch at dim ${i}: expected ${expected[i]}, got ${tensor.shape[i]}` +
        (tensor.label ? ` (${tensor.label})` : '')
      );
    }
  }
}


export function dtypeBytes(dtype) {
  return dtype === 'f16' ? 2 : 4;
}


export function tensorBytes(shape, dtype) {
  return shape.reduce((a, b) => a * b, 1) * dtypeBytes(dtype);
}


export function dtypesMatch(a, b) {
  return a.dtype === b.dtype;
}


export function inferOutputDtype(a, b) {
  return (a.dtype === 'f16' && b.dtype === 'f16') ? 'f16' : 'f32';
}
