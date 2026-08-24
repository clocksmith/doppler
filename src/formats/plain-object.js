export function isPlainObject(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}
