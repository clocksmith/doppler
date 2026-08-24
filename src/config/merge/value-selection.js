export function chooseNullish(overrideValue, fallbackValue) {
  return overrideValue ?? fallbackValue;
}

export function chooseDefined(overrideValue, fallbackValue) {
  return overrideValue !== undefined ? overrideValue : fallbackValue;
}

export function chooseDefinedWithSource(path, overrideValue, fallbackValue, sources) {
  const value = chooseDefined(overrideValue, fallbackValue);
  if (sources && typeof sources.set === 'function') {
    sources.set(
      path,
      overrideValue !== undefined ? 'runtime' : 'manifest',
    );
  }
  return value;
}

export function mergeShallowObject(base, override) {
  if (override === undefined) {
    return base;
  }
  if (override === null || typeof override !== 'object' || Array.isArray(override)) {
    throw new Error(
      'DopplerConfigError: shallow object overrides must be plain objects when provided explicitly.'
    );
  }
  return { ...base, ...override };
}

export function mergeLayeredShallowObjects(...layers) {
  return layers.reduce((merged, layer) => mergeShallowObject(merged, layer), {});
}

export function replaceSubtree(overrideValue, fallbackValue) {
  return chooseDefined(overrideValue, fallbackValue);
}
