


import { selectRuleValue } from '../rules/rule-registry.js';


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


export function dtypeBytes(dtype) {
  return dtype === 'f16' ? 2 : 4;
}


export function tensorBytes(shape, dtype) {
  return shape.reduce((a, b) => a * b, 1) * dtypeBytes(dtype);
}


export function inferOutputDtype(a, b) {
  return selectRuleValue('shared', 'dtype', 'bothF16', { aDtype: a.dtype, bDtype: b.dtype });
}
