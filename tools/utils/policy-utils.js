function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function normalizeTrimmedText(value) {
  return value == null ? '' : String(value).trim();
}

function resolveText(value, fallback) {
  if (value === null || value === undefined) return fallback;
  const normalized = String(value).trim();
  if (normalized === '') {
    throw new Error(`Expected non-empty text value, got ${JSON.stringify(value)}`);
  }
  return normalized;
}

function resolvePolicyText(value, fallback) {
  return resolveText(value, fallback);
}

function resolveTextArray(value, fallback) {
  if (value === null || value === undefined) return fallback;
  if (!Array.isArray(value)) {
    throw new Error(`Expected array value, got ${typeof value}`);
  }
  return value;
}

export {
  isObject,
  normalizeTrimmedText,
  resolveText,
  resolvePolicyText,
  resolveTextArray,
};
