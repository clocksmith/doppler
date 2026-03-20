import { selectRuleValue } from '../../../rules/rule-registry.js';

const DEFAULT_KERNEL_PATH_POLICY = Object.freeze({
  mode: 'locked',
  sourceScope: Object.freeze(['model', 'manifest']),
  onIncompatible: 'error',
});

function normalizeKernelPathSource(source) {
  const normalized = String(source ?? '').trim().toLowerCase();
  if (normalized === 'runtime') {
    return 'config';
  }
  return normalized || 'none';
}

function normalizeKernelPathPolicyMode(value) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (normalized === 'capability-aware') {
    return 'capability-aware';
  }
  return 'locked';
}

function normalizeKernelPathPolicySourceScope(value) {
  if (!Array.isArray(value)) {
    return [...DEFAULT_KERNEL_PATH_POLICY.sourceScope];
  }
  const normalized = new Set();
  for (const source of value) {
    const normalizedSource = normalizeKernelPathSource(source);
    if (normalizedSource === 'none') continue;
    normalized.add(normalizedSource);
  }
  if (normalized.size === 0) {
    return [...DEFAULT_KERNEL_PATH_POLICY.sourceScope];
  }
  return [...normalized];
}

function normalizeKernelPathPolicyOnIncompatible(value) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (normalized === 'remap') {
    return 'remap';
  }
  return 'error';
}

export function resolveKernelPathPolicy(policy) {
  if (!policy || typeof policy !== 'object' || Array.isArray(policy)) {
    return {
      mode: DEFAULT_KERNEL_PATH_POLICY.mode,
      sourceScope: [...DEFAULT_KERNEL_PATH_POLICY.sourceScope],
      allowSources: [...DEFAULT_KERNEL_PATH_POLICY.sourceScope],
      onIncompatible: DEFAULT_KERNEL_PATH_POLICY.onIncompatible,
    };
  }

  const sourceScope = normalizeKernelPathPolicySourceScope(
    policy.sourceScope ?? policy.allowSources
  );

  return {
    mode: normalizeKernelPathPolicyMode(policy.mode),
    sourceScope,
    allowSources: [...sourceScope],
    onIncompatible: normalizeKernelPathPolicyOnIncompatible(policy.onIncompatible),
  };
}

export function resolveCapabilityKernelPathRef(configuredKernelPathRef, kernelPathSource, capabilities, kernelPathPolicy = null) {
  if (typeof configuredKernelPathRef !== 'string') {
    return configuredKernelPathRef;
  }

  const normalizedPolicy = resolveKernelPathPolicy(kernelPathPolicy);
  const hasSubgroups = capabilities?.hasSubgroups === true;
  const hasF16 = capabilities?.hasF16 === true;
  const normalizedSource = normalizeKernelPathSource(kernelPathSource);
  const allowCapabilityAutoSelection = normalizedPolicy.mode === 'capability-aware'
    && normalizedPolicy.sourceScope.includes(normalizedSource);

  return selectRuleValue('inference', 'kernelPath', 'autoSelect', {
    kernelPathRef: configuredKernelPathRef,
    hasSubgroups,
    hasF16,
    allowCapabilityAutoSelection,
  });
}
