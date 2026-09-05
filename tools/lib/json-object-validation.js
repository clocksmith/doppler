import { isPlainObject } from '../../src/formats/plain-object.js';

// Evidence checkers collect field errors in order. False means non-object input,
// not that the object passed every field check.
export function validateExactKeys(value, fields, label, errors) {
  if (!isPlainObject(value)) {
    errors.push(`${label} must be an object`);
    return false;
  }
  const expected = new Set(fields);
  for (const field of Object.keys(value)) {
    if (!expected.has(field)) errors.push(`${label}.${field} is not supported`);
  }
  for (const field of fields) {
    if (!Object.prototype.hasOwnProperty.call(value, field)) {
      errors.push(`${label}.${field} is required`);
    }
  }
  return true;
}
