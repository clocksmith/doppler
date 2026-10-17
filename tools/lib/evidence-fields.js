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

/** @param {unknown} value */
export function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

/** @param {unknown} value @param {string} label @param {string[]} errors @param {boolean} [nullable] */
export function text(value, label, errors, nullable = false) {
  if (nullable && value === null) return null;
  const normalized = normalizeText(value);
  if (!normalized) errors.push(`${label} must be a non-empty string${nullable ? ' or null' : ''}`);
  return normalized || null;
}

/** @param {unknown} value @param {string} label @param {string[]} errors */
export function instant(value, label, errors) {
  const normalized = text(value, label, errors);
  const parsed = normalized ? new Date(normalized) : null;
  if (!parsed || !Number.isFinite(parsed.getTime()) || parsed.toISOString() !== normalized) {
    errors.push(`${label} must be an ISO instant`);
    return null;
  }
  return parsed;
}

/** @param {unknown} value @param {string} label @param {string[]} errors @param {boolean} [nullable] */
export function sha256(value, label, errors, nullable = false) {
  const normalized = text(value, label, errors, nullable)?.toLowerCase() ?? null;
  if (normalized && !/^sha256:[0-9a-f]{64}$/.test(normalized)) {
    errors.push(`${label} must be a SHA-256 identity${nullable ? ' or null' : ''}`);
  }
  return normalized;
}

/** @param {unknown} value @param {string} label @param {string[]} errors */
export function revision(value, label, errors) {
  const normalized = text(value, label, errors)?.toLowerCase() ?? null;
  if (normalized && !/^[0-9a-f]{40}$/.test(normalized)) {
    errors.push(`${label} must be a lowercase 40-hex revision`);
  }
  return normalized;
}

/** @param {unknown} value @param {string} label @param {string[]} errors */
export function boolean(value, label, errors) {
  if (typeof value !== 'boolean') errors.push(`${label} must be boolean`);
  return value === true;
}

/** @param {unknown} value @param {string} label @param {string[]} errors @param {number} [minimum] */
export function integer(value, label, errors, minimum = 0) {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < minimum) {
    errors.push(`${label} must be an integer of at least ${minimum}`);
    return null;
  }
  return value;
}

/** @param {unknown} value @param {string} label @param {string[]} errors @param {number} [minimum] */
export function number(value, label, errors, minimum = 0) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum) {
    errors.push(`${label} must be a finite number of at least ${minimum}`);
    return null;
  }
  return value;
}

/** Capture commands throw at their first error instead of accumulating diagnostics.
 * @param {unknown} value @param {string} label */
export function requireText(value, label) {
  const normalized = normalizeText(value);
  if (!normalized) throw new Error(`${label} must be a non-empty string.`);
  return normalized;
}
