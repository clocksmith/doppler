import { getKernelCapabilities } from '../../../gpu/device.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';

function asVendorString(caps) {
  const raw = caps?.adapterInfo?.vendor;
  return typeof raw === 'string' && raw.trim() !== '' ? raw.toLowerCase() : 'unknown';
}

export function resolveMoeVendorProfile(modelType) {
  if (modelType !== 'gpt-oss') {
    return {
      preferVec4Dequant: false,
      dequantTileShape: 'scalar',
      routerWorkgroupSize: 128,
      maxTokensPerExpertScale: 1.0,
    };
  }
  const caps = getKernelCapabilities();
  const vendor = asVendorString(caps);
  return selectRuleValue('kernels', 'moeGptoss', 'vendorQuirkProfile', { vendor });
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
    const hiddenDivisor = policy.hiddenSizeDivisor ?? 32;
    const intermediateDivisor = policy.intermediateSizeDivisor ?? 32;
    if (hiddenSize % hiddenDivisor !== 0 || intermediateSize % intermediateDivisor !== 0) {
      throw new Error(
        `[MoE] GPT-OSS shape policy violation: hiddenSize (${hiddenSize}) % ${hiddenDivisor} = ${hiddenSize % hiddenDivisor}, ` +
        `intermediateSize (${intermediateSize}) % ${intermediateDivisor} = ${intermediateSize % intermediateDivisor}.`
      );
    }
  }
}
