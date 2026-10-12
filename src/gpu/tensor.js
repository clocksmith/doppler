import { selectRuleValue } from '../rules/rule-registry.js';

// Representation is local to this tensor. It never changes weight dtypes or
// consults a session/global precision setting.
function pairedRepresentation(dtype, representation) {
  if (representation !== undefined && (representation === null || typeof representation !== 'object' || Array.isArray(representation))) {
    throw new TypeError('Tensor representation must be an explicit metadata object.');
  }
  if (representation === undefined || (representation?.precision === undefined
    && representation?.components === undefined && representation?.elementStrideBytes === undefined)) return false;
  if (dtype !== 'f32' || representation?.precision !== 'two-f32'
    || representation.components !== 2 || representation.elementStrideBytes !== 8) {
    throw new TypeError('Paired tensor representation requires f32 precision, two components and eight-byte stride.');
  }
  return true;
}

export function createTensor(buffer, dtype, shape, label, representation) {
  const paired = pairedRepresentation(dtype, representation);
  if (paired && (!Number.isSafeInteger(buffer?.size) || buffer.size < tensorBytes(shape, dtype, representation))) {
    throw new RangeError('Paired tensor buffer does not contain its declared bytes.');
  }
  const tensor = { buffer, dtype, shape: Object.freeze([...shape]), label };
  if (paired) Object.defineProperties(tensor, {
    precision: { value: 'two-f32', enumerable: true },
    components: { value: 2, enumerable: true },
    elementStrideBytes: { value: 8, enumerable: true },
  });
  return tensor;
}

// Physical storage word size; a paired element still consists of F32 words.
export function dtypeBytes(dtype) {
  return dtype === 'f16' ? 2 : 4;
}

export function tensorElementBytes(dtype, representation) {
  return pairedRepresentation(dtype, representation) ? 8 : dtypeBytes(dtype);
}

export function tensorBytes(shape, dtype, representation) {
  const paired = pairedRepresentation(dtype, representation);
  if (paired && (!Array.isArray(shape) || shape.some(dimension => !Number.isSafeInteger(dimension) || dimension < 0))) {
    throw new RangeError('Paired tensor shape requires nonnegative safe integer dimensions.');
  }
  const bytes = shape.reduce((a, b) => a * b, 1) * tensorElementBytes(dtype, representation);
  if (paired && !Number.isSafeInteger(bytes)) throw new RangeError('Paired tensor byte size exceeds the safe integer range.');
  return bytes;
}

export function inferOutputDtype(a, b) {
  return selectRuleValue('shared', 'dtype', 'bothF16', { aDtype: a.dtype, bDtype: b.dtype });
}
