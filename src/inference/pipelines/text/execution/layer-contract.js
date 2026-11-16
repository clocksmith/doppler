import { RMSNORM_PAIR_CACHE_LIMIT } from "../../../../gpu/kernels/rmsnorm-pair.js";
import { postNormContractMatchesBase } from "../normalization-contract.js";
import { isConvLayerType, normalizeLayerType } from "./layer-input-execution.js";

// ============================================================================
// Architecture Detection
// ============================================================================
export function detectSandwichNorm(config) {
  const hasPreFeedforwardNorm = config?.preFeedforwardNorm === true;
  const hasPostFeedforwardNorm = config?.postFeedforwardNorm === true;
  const hasPostAttentionNorm = config?.postAttentionNorm === true;
  return {
    useSandwichNorm: hasPreFeedforwardNorm || hasPostFeedforwardNorm,
    hasPreFeedforwardNorm,
    hasPostFeedforwardNorm,
    hasPostAttentionNorm,
  };
}

export function shouldUseSandwichRMSNormPairFusion({
  context,
  sandwichNorm,
  layerWeights,
  numTokens,
  hiddenSize,
  attnOutput,
  inputTensor,
}) {
  if (context.config?.normalizationType === 'layernorm' || !postNormContractMatchesBase(context.config)) return false;
  if (context.useSandwichRMSNormPairFusion !== true) {
    return false;
  }
  if (numTokens !== 1) {
    return false;
  }
  if (!sandwichNorm.useSandwichNorm || !sandwichNorm.hasPostAttentionNorm || !sandwichNorm.hasPreFeedforwardNorm) {
    return false;
  }
  if (!layerWeights?.postAttentionNorm || !layerWeights?.preFeedforwardNorm) {
    return false;
  }
  if (attnOutput?.dtype !== 'f32' || (inputTensor && inputTensor.dtype !== 'f32')) {
    throw new Error(
      'useSandwichRMSNormPairFusion requires f32 attention and residual tensors ' +
      `(attn=${String(attnOutput?.dtype)}, residual=${String(inputTensor?.dtype)}).`
    );
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0 || hiddenSize > RMSNORM_PAIR_CACHE_LIMIT) {
    throw new Error(
      `useSandwichRMSNormPairFusion requires hiddenSize in 1..${RMSNORM_PAIR_CACHE_LIMIT}; got ${String(hiddenSize)}.`
    );
  }
  return true;
}

const UNSUPPORTED_LAYER_RUNTIME_SET = new Set(['mamba', 'rwkv']);

export function assertSupportedLayerRuntime(layerIdx, config) {
  const modelType = normalizeLayerType(config?.modelType);
  if (UNSUPPORTED_LAYER_RUNTIME_SET.has(modelType)) {
    throw new Error(
      `Unsupported runtime family "${modelType}" for layer ${layerIdx}. ` +
      'Mamba/RWKV execution is fail-closed until implemented.'
    );
  }

  const layerType = normalizeLayerType(config?.layerTypes?.[layerIdx]);
  if (UNSUPPORTED_LAYER_RUNTIME_SET.has(layerType)) {
    throw new Error(
      `Unsupported layer type "${layerType}" at layer ${layerIdx}. ` +
      'Mamba/RWKV execution is fail-closed until implemented.'
    );
  }
}

export function hasConvLayers(layerTypes) {
  if (!Array.isArray(layerTypes)) return false;
  for (let i = 0; i < layerTypes.length; i++) {
    if (isConvLayerType(layerTypes[i])) return true;
  }
  return false;
}

export function isLinearLayerType(layerType) {
  const normalized = normalizeLayerType(layerType);
  return normalized === 'linear_attention'
    || normalized === 'linear'
    || normalized === 'gated_delta'
    || normalized === 'gated_delta_net';
}
