import { getKernelCapabilities } from '../../../gpu/device.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';

const KERNEL_PATH_PROFILE_RESOLVERS = Object.freeze({
  'gpt-oss': resolveGptOssKernelPathProfile,
  mixtral: resolveMixtralKernelPathProfile,
});

function asVendorString(caps) {
  const raw = caps?.adapterInfo?.vendor;
  return typeof raw === 'string' && raw.trim() !== '' ? raw.toLowerCase() : 'unknown';
}

function requireRuleContextValue(value, label) {
  if (value == null || value === '') {
    throw new Error(`[MoE] ${label} is required for rule selection.`);
  }
  return value;
}

function requireMoeProfileField(profile, field) {
  const value = profile?.[field];
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`[MoE] execution profile is missing ${field}.`);
  }
  return value;
}

function requireManifestString(value, label) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`[MoE] ${label} is required for MoE execution profile resolution.`);
  }
  return value;
}

function validateVendorProfile(profile) {
  if (!profile || typeof profile !== 'object' || Array.isArray(profile)) {
    throw new Error('[MoE] vendor profile must resolve to an object.');
  }
  if (typeof profile.preferVec4Dequant !== 'boolean') {
    throw new Error('[MoE] vendor profile missing preferVec4Dequant boolean.');
  }
  if (profile.dequantTileShape !== null && profile.dequantTileShape !== 'vec4' && profile.dequantTileShape !== 'scalar') {
    throw new Error('[MoE] vendor profile dequantTileShape must be null, "vec4", or "scalar".');
  }
  if (!Number.isFinite(profile.routerWorkgroupSize) || profile.routerWorkgroupSize <= 0) {
    throw new Error('[MoE] vendor profile missing positive routerWorkgroupSize.');
  }
  if (!Number.isFinite(profile.maxTokensPerExpertScale) || profile.maxTokensPerExpertScale <= 0) {
    throw new Error('[MoE] vendor profile missing positive maxTokensPerExpertScale.');
  }
  return profile;
}

function validateMoeExecutionProfile(profile) {
  if (!profile || typeof profile !== 'object' || Array.isArray(profile)) {
    throw new Error('[MoE] executionProfile rule must resolve to an object.');
  }
  requireMoeProfileField(profile, 'id');
  requireMoeProfileField(profile, 'label');
  requireMoeProfileField(profile, 'expertExecutor');
  requireMoeProfileField(profile, 'intermediateSizeSource');
  if (typeof profile.requiresShaderF16 !== 'boolean') {
    throw new Error('[MoE] execution profile is missing requiresShaderF16 boolean.');
  }
  if (profile.routerScaleMode !== 'none' && profile.routerScaleMode !== 'optional' && profile.routerScaleMode !== 'required') {
    throw new Error('[MoE] execution profile routerScaleMode must be none, optional, or required.');
  }
  if (!profile.vendorProfile || typeof profile.vendorProfile !== 'object' || Array.isArray(profile.vendorProfile)) {
    throw new Error('[MoE] execution profile is missing vendorProfile selector.');
  }
  return profile;
}

export function resolveMoeExecutionProfile(config, options = {}) {
  const expertFormat = requireManifestString(config?.expertFormat, 'manifest moeConfig.expertFormat');
  const modelType = requireManifestString(
    typeof options?.modelType === 'string' && options.modelType.length > 0
      ? options.modelType
      : config?.modelType,
    'manifest modelType'
  );
  const profile = selectRuleValue('inference', 'moe', 'executionProfile', {
    modelType,
    expertFormat,
  });
  return validateMoeExecutionProfile(profile);
}

export function resolveMoeIntermediateSize(config, moeProfile) {
  const source = requireMoeProfileField(moeProfile, 'intermediateSizeSource');
  const resolvers = {
    architecture: () => config?.intermediateSize,
    expert: () => config?.expertIntermediateSize,
  };
  const resolver = resolvers[source];
  if (typeof resolver !== 'function') {
    throw new Error(`[MoE] Unknown intermediateSizeSource "${source}" in execution profile "${moeProfile.id}".`);
  }
  const value = Number(resolver());
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`[MoE] Invalid ${source} intermediate size for profile "${moeProfile.id}": ${String(value)}.`);
  }
  return value;
}

export function resolveMoeVendorProfile(moeProfile) {
  const selector = moeProfile?.vendorProfile;
  if (selector?.kind === 'rule') {
    const caps = getKernelCapabilities();
    const vendor = asVendorString(caps);
    const profile = selectRuleValue(selector.domain, selector.group, selector.name, { vendor });
    return validateVendorProfile(profile);
  }
  if (selector?.kind === 'static') {
    return validateVendorProfile(selector.value);
  }
  throw new Error(`[MoE] execution profile "${moeProfile?.id ?? 'unknown'}" has invalid vendorProfile selector.`);
}

function resolveGptOssRuleContext(context) {
  return {
    modelType: requireRuleContextValue(context?.modelType, 'GPT-OSS kernel rule modelType'),
    hasF16: context?.hasF16,
    hasSubgroups: context?.hasSubgroups,
    inputDtype: requireRuleContextValue(context?.routerDtype ?? context?.inputDtype, 'GPT-OSS router/input dtype'),
    weightsDtype: requireRuleContextValue(context?.weightsDtype, 'GPT-OSS weights dtype'),
    outputDtype: requireRuleContextValue(context?.outputDtype, 'GPT-OSS output dtype'),
    groupSize: context?.groupSize,
    dequantTileShape: context?.tileShape ?? context?.dequantTileShape,
  };
}

export async function resolveGptOssKernelPathProfile(context) {
  const ruleContext = resolveGptOssRuleContext(context);
  return {
    routerTopK: selectRuleValue('kernels', 'moeGptoss', 'routerTopKVariant', ruleContext),
    dequantExpert: selectRuleValue('kernels', 'moeGptoss', 'dequantVariant', ruleContext),
  };
}

function resolveMixtralRuleContext(context) {
  return {
    modelType: requireRuleContextValue(context?.modelType, 'Mixtral kernel rule modelType'),
    hasF16: context?.hasF16,
    hasSubgroups: context?.hasSubgroups,
    routerDtype: requireRuleContextValue(context?.routerDtype, 'Mixtral router dtype'),
    weightsDtype: requireRuleContextValue(context?.weightsDtype, 'Mixtral weights dtype'),
    outputDtype: requireRuleContextValue(context?.outputDtype, 'Mixtral output dtype'),
  };
}

export async function resolveMixtralKernelPathProfile(context) {
  const ruleContext = resolveMixtralRuleContext(context);
  return {
    routerTopK: selectRuleValue('kernels', 'moeMixtral', 'routerTopKVariant', ruleContext),
    dequantExpert: selectRuleValue('kernels', 'moeMixtral', 'dequantVariant', ruleContext),
  };
}

export async function resolveMoeKernelPathProfile(moeProfile, context) {
  const resolverId = moeProfile?.kernelPathProfileResolver ?? null;
  if (resolverId === null) {
    return null;
  }
  const resolver = KERNEL_PATH_PROFILE_RESOLVERS[resolverId];
  if (typeof resolver !== 'function') {
    throw new Error(`[MoE] Unknown kernelPathProfileResolver "${resolverId}" in execution profile "${moeProfile.id}".`);
  }
  const modelType = requireRuleContextValue(moeProfile.kernelRuleModelType, `${moeProfile.id} kernel rule modelType`);
  return resolver({ ...context, modelType });
}

function resolveShapePolicy(moeProfile, context) {
  const selector = moeProfile?.shapePolicy ?? null;
  if (selector === null) {
    return null;
  }
  if (selector.kind !== 'rule') {
    throw new Error(`[MoE] execution profile "${moeProfile.id}" has invalid shapePolicy selector.`);
  }
  return selectRuleValue(selector.domain, selector.group, selector.name, context);
}

export function validateMoeShape(config, options = {}) {
  const {
    hiddenSize,
    intermediateSize,
    moeTopK,
    numExperts,
    expertFormat,
  } = config;
  if (typeof options.modelType !== 'string' || options.modelType.length === 0) {
    throw new Error('[MoE] validateMoeShape requires options.modelType from the manifest.');
  }
  const moeProfile = options.moeProfile ?? resolveMoeExecutionProfile(
    { expertFormat, modelType: options.modelType },
    { modelType: options.modelType }
  );

  if (!Number.isFinite(hiddenSize) || hiddenSize <= 0) {
    throw new Error(`[MoE] hiddenSize must be > 0, got ${hiddenSize}.`);
  }
  if (!Number.isFinite(intermediateSize) || intermediateSize <= 0) {
    throw new Error(`[MoE] intermediateSize must be > 0, got ${intermediateSize}.`);
  }
  if (!Number.isFinite(numExperts) || numExperts <= 0) {
    throw new Error(`[MoE] numExperts must be > 0, got ${numExperts}.`);
  }
  if (!Number.isFinite(moeTopK) || moeTopK <= 0 || moeTopK > numExperts) {
    throw new Error(`[MoE] topK must be in range [1, ${numExperts}], got ${moeTopK}.`);
  }

  const policy = resolveShapePolicy(moeProfile, { modelType: options.modelType });
  if (policy != null) {
    if (policy.hiddenSizeDivisor == null || policy.intermediateSizeDivisor == null) {
      throw new Error(`[MoE] ${moeProfile.label} shapePolicy is missing hiddenSizeDivisor or intermediateSizeDivisor.`);
    }
    const hiddenDivisor = policy.hiddenSizeDivisor;
    const intermediateDivisor = policy.intermediateSizeDivisor;
    if (hiddenSize % hiddenDivisor !== 0 || intermediateSize % intermediateDivisor !== 0) {
      throw new Error(
        `[MoE] ${moeProfile.label} shape policy violation: hiddenSize (${hiddenSize}) % ${hiddenDivisor} = ${hiddenSize % hiddenDivisor}, ` +
        `intermediateSize (${intermediateSize}) % ${intermediateDivisor} = ${intermediateSize % intermediateDivisor}.`
      );
    }
  }
}
