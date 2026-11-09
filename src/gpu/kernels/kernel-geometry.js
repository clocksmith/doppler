// Strict geometry validation at kernel boundaries. Kernels receive integers
// their callers already resolved; coercion of strings or fractional values
// belongs to import and CLI normalizers, never here, so every kernel rejects
// 1.9 and "4" the same way.

/**
 * @param {string} context
 * @param {string} label
 * @param {string} expectation
 * @param {unknown} value
 */
function describe(context, label, expectation, value) {
  const got = `got ${typeof value === 'string' ? JSON.stringify(value) : String(value)}`;
  return context
    ? `${context} requires ${label} to be ${expectation}; ${got}.`
    : `${label} must be ${expectation}; ${got}.`;
}

/**
 * @param {unknown} value
 * @param {string} label
 * @param {string} [context]
 * @returns {number}
 */
export function requirePositiveInteger(value, label, context = '') {
  if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
    throw new Error(describe(context, label, 'a positive integer', value));
  }
  return value;
}

/**
 * @param {unknown} value
 * @param {string} label
 * @param {string} [context]
 * @returns {number}
 */
export function requireNonNegativeInteger(value, label, context = '') {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new Error(describe(context, label, 'a non-negative integer', value));
  }
  return value;
}
