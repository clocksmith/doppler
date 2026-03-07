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
  return chooseNullish(overrideValue, fallbackValue);
}

const DEFAULT_KERNEL_PATH_POLICY = Object.freeze({
  mode: 'locked',
  sourceScope: Object.freeze(['model', 'manifest']),
  onIncompatible: 'error',
});

const VALID_KERNEL_PATH_POLICY_SOURCES = new Set([
  'model',
  'manifest',
  'config',
  'execution-v0',
]);

function normalizeKernelPathPolicyMode(value) {
  if (value === undefined) {
    return DEFAULT_KERNEL_PATH_POLICY.mode;
  }
  const normalized = String(value).trim().toLowerCase();
  if (normalized === 'locked' || normalized === 'capability-aware') {
    return normalized;
  }
  throw new Error(
    `DopplerConfigError: runtime.inference.kernelPathPolicy.mode must be "locked" or "capability-aware"; got ${JSON.stringify(value)}.`
  );
}

function normalizeKernelPathPolicySource(source) {
  const normalized = String(source ?? '').trim().toLowerCase();
  if (!normalized) {
    throw new Error(
      'DopplerConfigError: runtime.inference.kernelPathPolicy.sourceScope entries must be non-empty strings.'
    );
  }
  if (normalized === 'runtime') {
    throw new Error(
      'DopplerConfigError: runtime.inference.kernelPathPolicy.sourceScope does not accept legacy "runtime". Use "config".'
    );
  }
  if (normalized === 'execution_v0') {
    throw new Error(
      'DopplerConfigError: runtime.inference.kernelPathPolicy.sourceScope does not accept legacy "execution_v0". Use "execution-v0".'
    );
  }
  if (!VALID_KERNEL_PATH_POLICY_SOURCES.has(normalized)) {
    throw new Error(
      `DopplerConfigError: runtime.inference.kernelPathPolicy.sourceScope entries must be model|manifest|config|execution-v0; got ${JSON.stringify(source)}.`
    );
  }
  return normalized;
}

function normalizeKernelPathPolicySourceScope(value) {
  if (value === undefined) {
    return [...DEFAULT_KERNEL_PATH_POLICY.sourceScope];
  }
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(
      'DopplerConfigError: runtime.inference.kernelPathPolicy.sourceScope must be a non-empty array.'
    );
  }
  return [...new Set(value.map((source) => normalizeKernelPathPolicySource(source)))];
}

function normalizeKernelPathPolicyOnIncompatible(value) {
  if (value === undefined) {
    return DEFAULT_KERNEL_PATH_POLICY.onIncompatible;
  }
  const normalized = String(value).trim().toLowerCase();
  if (normalized === 'error' || normalized === 'remap') {
    return normalized;
  }
  throw new Error(
    `DopplerConfigError: runtime.inference.kernelPathPolicy.onIncompatible must be "error" or "remap"; got ${JSON.stringify(value)}.`
  );
}

function assertKernelPathPolicyObject(value) {
  if (value == null) {
    return;
  }
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(
      'DopplerConfigError: runtime.inference.kernelPathPolicy must be an object.'
    );
  }
}

export function mergeKernelPathPolicy(basePolicy, overridePolicy) {
  assertKernelPathPolicyObject(basePolicy);
  assertKernelPathPolicyObject(overridePolicy);
  const base = basePolicy ?? {};
  const override = overridePolicy ?? {};
  const sourceScope = normalizeKernelPathPolicySourceScope(
    override.sourceScope
    ?? override.allowSources
    ?? base.sourceScope
    ?? base.allowSources
  );
  return {
    mode: normalizeKernelPathPolicyMode(override.mode ?? base.mode),
    sourceScope,
    allowSources: [...sourceScope],
    onIncompatible: normalizeKernelPathPolicyOnIncompatible(
      override.onIncompatible ?? base.onIncompatible
    ),
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
