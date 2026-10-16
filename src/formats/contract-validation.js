import { isPlainObject } from './plain-object.js';

const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;

/** @type {import('./contract-validation.js').requireObject} */
export function requireObject(value, label, errors) {
  if (!isPlainObject(value)) {
    errors.push(`${label} must be an object.`);
    return false;
  }
  return true;
}

/** @type {import('./contract-validation.js').requireExactKeys} */
export function requireExactKeys(value, allowed, label, errors) {
  if (!isPlainObject(value)) return;
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) errors.push(`${label}.${key} is not allowed.`);
  }
}

/** @type {import('./contract-validation.js').requireString} */
export function requireString(value, label, errors) {
  if (typeof value !== 'string' || !value.trim()) errors.push(`${label} must be a non-empty string.`);
}

/** @type {import('./contract-validation.js').requireDigest} */
export function requireDigest(value, label, errors) {
  if (!SHA256_PATTERN.test(value || '')) errors.push(`${label} must be a SHA-256 digest.`);
}
