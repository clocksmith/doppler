export function formatRate(value) {
  if (!Number.isFinite(value)) return '--';
  return `${value.toFixed(2)} tok/s`;
}

export function formatMs(value) {
  if (!Number.isFinite(value)) return '--';
  return `${Math.round(value)}ms`;
}

export function formatScalar(value, digits = 4) {
  if (!Number.isFinite(value)) return '--';
  return value.toFixed(digits);
}

export function formatAutoValue(value, { integer = false } = {}) {
  if (!Number.isFinite(value)) return '--';
  if (integer) return `${Math.round(value)}`;
  const rounded = Math.round(value * 1000) / 1000;
  return `${rounded}`;
}
