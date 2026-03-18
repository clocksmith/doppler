import { getKernelCapabilities } from '../../../gpu/device.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';

function asVendorString(caps) {
  const raw = caps?.adapterInfo?.vendor;
  return typeof raw === 'string' && raw.trim() !== '' ? raw.toLowerCase() : 'unknown';
}

export function resolveMoeVendorProfile(modelType) {
  const caps = getKernelCapabilities();
  const vendor = asVendorString(caps);
  if (modelType === 'gpt-oss') {
    return selectRuleValue('kernels', 'moeGptoss', 'vendorQuirkProfile', { vendor });
  }
  if (modelType === 'mixtral') {
    return selectRuleValue('kernels', 'moeMixtral', 'vendorQuirkProfile', { vendor });
  }
  throw new Error(`[MoE] Unknown modelType "${modelType}" for vendor profile resolution.`);
}

function resolveGptOssRuleContext(context) {
  return {
    modelType: 'gpt-oss',
    hasF16: context?.hasF16,
    hasSubgroups: context?.hasSubgroups,
    inputDtype: context?.routerDtype ?? context?.inputDtype,
    weightsDtype: context?.weightsDtype,
    outputDtype: context?.outputDtype ?? context?.weightsDtype,
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
    modelType: 'mixtral',
    hasF16: context?.hasF16,
    hasSubgroups: context?.hasSubgroups,
    routerDtype: context?.routerDtype ?? 'f32',
    weightsDtype: context?.weightsDtype,
    outputDtype: context?.outputDtype ?? context?.weightsDtype,
  };
}

export async function resolveMixtralKernelPathProfile(context) {
  const ruleContext = resolveMixtralRuleContext(context);
  return {
    routerTopK: selectRuleValue('kernels', 'moeMixtral', 'routerTopKVariant', ruleContext),
    dequantExpert: selectRuleValue('kernels', 'moeMixtral', 'dequantVariant', ruleContext),
  };
}

export function validateMoeShape(config, options = {}) {
  const {
    hiddenSize,
    intermediateSize,
    moeTopK,
    numExperts,
    expertFormat,
  } = config;
  const modelType = options.modelType ?? (expertFormat === 'gpt-oss' ? 'gpt-oss' : 'mixtral');

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

  if (modelType === 'gpt-oss') {
    const policy = selectRuleValue('kernels', 'moeGptoss', 'shapePolicy', { modelType });
    if (policy.hiddenSizeDivisor == null || policy.intermediateSizeDivisor == null) {
      throw new Error('[MoE] GPT-OSS shapePolicy is missing hiddenSizeDivisor or intermediateSizeDivisor.');
    }
    const hiddenDivisor = policy.hiddenSizeDivisor;
    const intermediateDivisor = policy.intermediateSizeDivisor;
    if (hiddenSize % hiddenDivisor !== 0 || intermediateSize % intermediateDivisor !== 0) {
      throw new Error(
        `[MoE] GPT-OSS shape policy violation: hiddenSize (${hiddenSize}) % ${hiddenDivisor} = ${hiddenSize % hiddenDivisor}, ` +
        `intermediateSize (${intermediateSize}) % ${intermediateDivisor} = ${intermediateSize % intermediateDivisor}.`
      );
    }
  }
}
