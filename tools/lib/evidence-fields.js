import { isPlainObject } from '../../src/formats/plain-object.js';

/**
 * Accumulate field errors; the return value indicates object shape only.
 * @param {unknown} value
 * @param {readonly string[]} fields
 * @param {string} label
 * @param {string[]} errors
 * @returns {boolean}
 */
export function exactKeys(value, fields, label, errors) {
  if (!isPlainObject(value)) {
    errors.push(`${label} must be an object`);
    return false;
  }
  const expected = new Set(fields);
  for (const field of Object.keys(value)) {
    if (!expected.has(field)) errors.push(`${label}.${field} is not supported`);
  }
  for (const field of fields) {
    if (!Object.hasOwn(value, field)) errors.push(`${label}.${field} is required`);
  }
  return true;
}
