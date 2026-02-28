function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function normalizeTrimmedText(value) {
  return value == null ? '' : String(value).trim();
}

function resolveText(value, fallback) {
  const normalized = normalizeTrimmedText(value);
  return normalized === '' ? fallback : normalized;
}

function resolvePolicyText(value, fallback) {
  return resolveText(value, fallback);
}

function resolveTextArray(value, fallback) {
  return Array.isArray(value) ? value : fallback;
}

export {
  isObject,
  normalizeTrimmedText,
  resolveText,
  resolvePolicyText,
  resolveTextArray,
};
