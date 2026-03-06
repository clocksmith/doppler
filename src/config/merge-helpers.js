export function chooseNullish(overrideValue, fallbackValue) {
  return overrideValue ?? fallbackValue;
}

export function chooseDefined(overrideValue, fallbackValue) {
  return overrideValue !== undefined ? overrideValue : fallbackValue;
}

export function chooseDefinedWithSource(path, overrideValue, fallbackValue, sources) {
  const value = chooseDefined(overrideValue, fallbackValue);
  if (sources && typeof sources.set === 'function') {
    sources.set(path, overrideValue !== undefined ? 'runtime' : 'manifest');
  }
  return value;
}

export function mergeShallowObject(base, override) {
  if (!override || typeof override !== 'object' || Array.isArray(override)) {
    return base;
  }
  return { ...base, ...override };
}

export function mergeLayeredShallowObjects(...layers) {
  return layers.reduce((merged, layer) => mergeShallowObject(merged, layer), {});
}

export function replaceSubtree(overrideValue, fallbackValue) {
  return chooseNullish(overrideValue, fallbackValue);
}

export function mergeKernelPathPolicy(basePolicy, overridePolicy) {
  const base = basePolicy ?? {};
  const override = overridePolicy ?? {};
  const baseSourceScope = base.sourceScope ?? base.allowSources;
  const overrideSourceScope = override.sourceScope ?? override.allowSources;
  const sourceScope = overrideSourceScope ?? baseSourceScope;
  return {
    mode: override.mode ?? base.mode,
    sourceScope,
    allowSources: sourceScope,
    onIncompatible: override.onIncompatible ?? base.onIncompatible,
  };
}

export function mergeExecutionPatchLists(basePatch, overridePatch) {
  const base = basePatch ?? {};
  const override = overridePatch ?? {};
  return {
    set: chooseNullish(override.set, chooseNullish(base.set, [])),
    remove: chooseNullish(override.remove, chooseNullish(base.remove, [])),
    add: chooseNullish(override.add, chooseNullish(base.add, [])),
  };
}
