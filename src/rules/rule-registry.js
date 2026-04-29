import { selectByRules } from '../gpu/kernels/rule-matcher.js';
import { buildInferenceExecutionRulesContractArtifact } from './execution-rules-contract-check.js';
import { buildLayerPatternContractArtifact } from './layer-pattern-contract-check.js';
import { cloneJsonValue as cloneRuleValue } from '../utils/clone-json.js';

// Rule files imported statically (JSON import attributes). Sync at module
// load time, no top-level await, and bundle-friendly. Requires Node >=22
// and any bundler with JSON-module support (Vite/Rollup/esbuild/webpack5).
import attentionRules from './kernels/attention.rules.json' with { type: 'json' };
import conv2dRules from './kernels/conv2d.rules.json' with { type: 'json' };
import depthwiseConv2dRules from './kernels/depthwise-conv2d.rules.json' with { type: 'json' };
import dequantRules from './kernels/dequant.rules.json' with { type: 'json' };
import energyRules from './kernels/energy.rules.json' with { type: 'json' };
import fusedFfnRules from './kernels/fused-ffn.rules.json' with { type: 'json' };
import fusedMatmulResidualRules from './kernels/fused-matmul-residual.rules.json' with { type: 'json' };
import fusedMatmulRmsnormRules from './kernels/fused-matmul-rmsnorm.rules.json' with { type: 'json' };
import gatherRules from './kernels/gather.rules.json' with { type: 'json' };
import geluRules from './kernels/gelu.rules.json' with { type: 'json' };
import groupedPointwiseConv2dRules from './kernels/grouped-pointwise-conv2d.rules.json' with { type: 'json' };
import groupnormRules from './kernels/groupnorm.rules.json' with { type: 'json' };
import kvQuantizeRules from './kernels/kv_quantize.rules.json' with { type: 'json' };
import layernormRules from './kernels/layernorm.rules.json' with { type: 'json' };
import matmulRules from './kernels/matmul.rules.json' with { type: 'json' };
import kernelMoeRules from './kernels/moe.rules.json' with { type: 'json' };
import kernelMoeGptOssRules from './kernels/moe.rules.gptoss.json' with { type: 'json' };
import kernelMoeMixtralRules from './kernels/moe.rules.mixtral.json' with { type: 'json' };
import modulateRules from './kernels/modulate.rules.json' with { type: 'json' };
import pixelShuffleRules from './kernels/pixel_shuffle.rules.json' with { type: 'json' };
import repeatChannelsRules from './kernels/repeat-channels.rules.json' with { type: 'json' };
import reluRules from './kernels/relu.rules.json' with { type: 'json' };
import residualRules from './kernels/residual.rules.json' with { type: 'json' };
import rmsnormRules from './kernels/rmsnorm.rules.json' with { type: 'json' };
import ropeRules from './kernels/rope.rules.json' with { type: 'json' };
import linearAttentionRules from './kernels/linear-attention.rules.json' with { type: 'json' };
import sampleRules from './kernels/sample.rules.json' with { type: 'json' };
import scaleRules from './kernels/scale.rules.json' with { type: 'json' };
import siluRules from './kernels/silu.rules.json' with { type: 'json' };
import splitQkvRules from './kernels/split-qkv.rules.json' with { type: 'json' };
import splitQgRules from './kernels/split-qg.rules.json' with { type: 'json' };
import softmaxRules from './kernels/softmax.rules.json' with { type: 'json' };
import upsample2dRules from './kernels/upsample2d.rules.json' with { type: 'json' };
import configRules from './inference/config.rules.json' with { type: 'json' };
import inferenceExecutionRules from './inference/execution.rules.json' with { type: 'json' };
import inferenceAttentionRules from './inference/attention.rules.json' with { type: 'json' };
import dtypeRules from './inference/dtype.rules.json' with { type: 'json' };
import ffnRules from './inference/ffn.rules.json' with { type: 'json' };
import layerRules from './inference/layer.rules.json' with { type: 'json' };
import layerPatternRules from './inference/layer-pattern.rules.json' with { type: 'json' };
import inferenceMoeRules from './inference/moe.rules.json' with { type: 'json' };
import tokenizerRules from './converter/tokenizer.rules.json' with { type: 'json' };
import tensorRolesRules from './converter/tensor-roles.rules.json' with { type: 'json' };
import converterExecutionRules from './converter/execution.rules.json' with { type: 'json' };
import loaderWeightRules from './loader/weights.rules.json' with { type: 'json' };
import tensorLoaderRules from './loader/tensor-loader.rules.json' with { type: 'json' };
import toolingCommandRuntimeRules from './tooling/command-runtime.rules.json' with { type: 'json' };


// deepFreeze assumes all values in the tree are plain objects, arrays, or
// primitives. Typed arrays, Maps, Sets, and other exotic objects will be
// frozen but their internal slots are not traversed. This is acceptable
// because rule JSON payloads only contain plain JSON-representable values.
function deepFreeze(value, seen = new WeakSet()) {
  if (!value || typeof value !== 'object' || seen.has(value)) {
    return value;
  }
  seen.add(value);
  for (const entry of Object.values(value)) {
    deepFreeze(entry, seen);
  }
  return Object.freeze(value);
}
const INFERENCE_EXECUTION_RULES_CONTRACT_ARTIFACT = buildInferenceExecutionRulesContractArtifact(
  inferenceExecutionRules
);
if (!INFERENCE_EXECUTION_RULES_CONTRACT_ARTIFACT.ok) {
  throw new Error(
    `RuleRegistry: inference.execution rules contract failed (file: inference/execution.rules.json): ` +
    `${INFERENCE_EXECUTION_RULES_CONTRACT_ARTIFACT.errors.join(' | ')}`
  );
}
const INFERENCE_LAYER_PATTERN_CONTRACT_ARTIFACT = buildLayerPatternContractArtifact(
  layerPatternRules
);
if (!INFERENCE_LAYER_PATTERN_CONTRACT_ARTIFACT.ok) {
  throw new Error(
    `RuleRegistry: inference.layerPattern rules contract failed (file: inference/layer-pattern.rules.json): ` +
    `${INFERENCE_LAYER_PATTERN_CONTRACT_ARTIFACT.errors.join(' | ')}`
  );
}

const RULE_SETS = {
  shared: {
    dtype: dtypeRules,
  },
  kernels: {
    attention: attentionRules,
    conv2d: conv2dRules,
    depthwiseConv2d: depthwiseConv2dRules,
    dequant: dequantRules,
    energy: energyRules,
    fusedFfn: fusedFfnRules,
    fusedMatmulResidual: fusedMatmulResidualRules,
    fusedMatmulRmsnorm: fusedMatmulRmsnormRules,
    gather: gatherRules,
    gelu: geluRules,
    groupedPointwiseConv2d: groupedPointwiseConv2dRules,
    groupnorm: groupnormRules,
    kv_quantize: kvQuantizeRules,
    layernorm: layernormRules,
    matmul: matmulRules,
    moe: kernelMoeRules,
    moeGptoss: kernelMoeGptOssRules,
    moeMixtral: kernelMoeMixtralRules,
    modulate: modulateRules,
    pixel_shuffle: pixelShuffleRules,
    repeatChannels: repeatChannelsRules,
    relu: reluRules,
    residual: residualRules,
    rmsnorm: rmsnormRules,
    rope: ropeRules,
    linearAttention: linearAttentionRules,
    sample: sampleRules,
    scale: scaleRules,
    silu: siluRules,
    splitQkv: splitQkvRules,
    splitQg: splitQgRules,
    softmax: softmaxRules,
    upsample2d: upsample2dRules,
  },
  inference: {
    config: configRules,
    execution: inferenceExecutionRules,
    attention: inferenceAttentionRules,
    // ALIAS: same rule set as shared.dtype — dtype.rules.json is loaded once and
    // registered under both namespaces so that callers in the inference domain can
    // use selectRuleValue('inference', 'dtype', ...) without reaching into 'shared'.
    // Do not remove this alias; existing call sites depend on both registration paths.
    dtype: dtypeRules,
    ffn: ffnRules,
    layer: layerRules,
    layerPattern: layerPatternRules,
    moe: inferenceMoeRules,
  },
  loader: {
    weights: loaderWeightRules,
    tensorLoader: tensorLoaderRules,
  },
  converter: {
    tokenizer: tokenizerRules,
    tensorRoles: tensorRolesRules,
    execution: converterExecutionRules,
  },
  tooling: {
    commandRuntime: toolingCommandRuntimeRules,
  },
};

export function getRuleSet(domain, group, name) {
  const domainRules = RULE_SETS[domain];
  if (!domainRules) {
    throw new Error(`RuleRegistry: unknown domain "${domain}".`);
  }
  const groupRules = domainRules[group];
  if (!groupRules) {
    throw new Error(`RuleRegistry: unknown rule group "${domain}.${group}".`);
  }
  const rules = groupRules[name];
  if (!rules) {
    throw new Error(`RuleRegistry: unknown rule set "${domain}.${group}.${name}".`);
  }
  return rules;
}

export function selectRuleValue(domain, group, name, context) {
  const rules = getRuleSet(domain, group, name);
  const value = selectByRules(rules, context);
  return resolveRuleValue(value, context);
}

export function registerRuleGroup(domain, group, rules) {
  if (!RULE_SETS[domain]) {
    RULE_SETS[domain] = {};
  }
  RULE_SETS[domain][group] = deepFreeze(cloneRuleValue(rules));
}

export function getInferenceExecutionRulesContractArtifact() {
  return INFERENCE_EXECUTION_RULES_CONTRACT_ARTIFACT;
}

export function getInferenceLayerPatternContractArtifact() {
  return INFERENCE_LAYER_PATTERN_CONTRACT_ARTIFACT;
}

function resolveRuleValue(value, context) {
  if (Array.isArray(value)) {
    return value.map((entry) => resolveRuleValue(entry, context));
  }
  if (!value || typeof value !== 'object') {
    return value;
  }

  if (isTemplateDirective(value)) {
    return applyTemplate(value.template, context);
  }
  if (isContextDirective(value)) {
    const resolved = context[value.context];
    if (resolved === undefined) {
      throw new Error(`RuleRegistry: missing context value "${value.context}".`);
    }
    return resolved;
  }

  const resolved = {};
  for (const [key, entry] of Object.entries(value)) {
    resolved[key] = resolveRuleValue(entry, context);
  }
  return resolved;
}

function isTemplateDirective(value) {
  return Object.keys(value).length === 1 && typeof value.template === 'string';
}

function isContextDirective(value) {
  return Object.keys(value).length === 1 && typeof value.context === 'string';
}

function applyTemplate(template, context) {
  return template.replace(/\{([a-zA-Z0-9_]+)\}/g, (match, key) => {
    if (!(key in context)) {
      throw new Error(`RuleRegistry: missing template key "${key}" for "${template}".`);
    }
    return String(context[key]);
  });
}

for (const domainRules of Object.values(RULE_SETS)) {
  for (const rules of Object.values(domainRules)) {
    deepFreeze(rules);
  }
}
