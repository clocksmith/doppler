import { getKernelCapabilities } from '../../../gpu/device.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { selectByRules } from '../../../gpu/kernels/rule-matcher.js';

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

let gptOssKernelPathConfigPromise = null;

async function loadGptOssKernelPathConfig() {
  if (!gptOssKernelPathConfigPromise) {
    gptOssKernelPathConfigPromise = fetch(new URL('../../../config/kernels/moe/gpt-oss.paths.json', import.meta.url))
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Failed to load GPT-OSS MoE path config: ${response.status}`);
        }
        return response.json();
      });
  }
  return gptOssKernelPathConfigPromise;
}

export async function resolveGptOssKernelPathProfile(context) {
  const config = await loadGptOssKernelPathConfig();
  return {
    routerTopK: selectByRules(config.router.topk, context),
    dequantExpert: selectByRules(config.dequant.mxfp4Expert, context),
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
