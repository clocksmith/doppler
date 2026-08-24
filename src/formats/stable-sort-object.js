export function stableSortObject(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => stableSortObject(entry));
  }
  if (!value || typeof value !== 'object') {
    return value;
  }
  const sorted = {};
  for (const key of Object.keys(value).sort()) {
    sorted[key] = stableSortObject(value[key]);
  }
  return sorted;
}
