import { computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { selectByRules } from './rule-matcher.js';
import { buildInferenceExecutionRulesContractArtifact } from './execution-rules-contract-check.js';
import { buildLayerPatternContractArtifact } from './layer-pattern-contract-check.js';
import { cloneJsonValue as cloneRuleValue } from '../formats/clone-json.js';

// Canonical policy remains in the individual rule files. This generated mirror
// collapses browser startup to one JSON module; CI proves it matches its sources.
import ruleBundle from './generated/rule-bundle.json' with { type: 'json' };

const {
  'kernels/attention.rules.json': attentionRules,
  'kernels/conv2d.rules.json': conv2dRules,
  'kernels/depthwise-conv2d.rules.json': depthwiseConv2dRules,
  'kernels/dequant.rules.json': dequantRules,
  'kernels/energy.rules.json': energyRules,
  'kernels/fused-ffn.rules.json': fusedFfnRules,
  'kernels/fused-matmul-residual.rules.json': fusedMatmulResidualRules,
  'kernels/fused-matmul-rmsnorm.rules.json': fusedMatmulRmsnormRules,
  'kernels/gather.rules.json': gatherRules,
  'kernels/gelu.rules.json': geluRules,
  'kernels/grouped-pointwise-conv2d.rules.json': groupedPointwiseConv2dRules,
  'kernels/groupnorm.rules.json': groupnormRules,
  'kernels/kv_quantize.rules.json': kvQuantizeRules,
  'kernels/layernorm.rules.json': layernormRules,
  'kernels/lm-head-argmax.rules.json': lmHeadArgmaxRules,
  'kernels/matmul.rules.json': matmulRules,
  'kernels/moe.rules.json': kernelMoeRules,
  'kernels/moe.rules.gptoss.json': kernelMoeGptOssRules,
  'kernels/moe.rules.mixtral.json': kernelMoeMixtralRules,
  'kernels/modulate.rules.json': modulateRules,
  'kernels/pixel_shuffle.rules.json': pixelShuffleRules,
  'kernels/repeat-channels.rules.json': repeatChannelsRules,
  'kernels/rep-penalty.rules.json': repPenaltyRules,
  'kernels/relu.rules.json': reluRules,
  'kernels/residual.rules.json': residualRules,
  'kernels/rmsnorm.rules.json': rmsnormRules,
  'kernels/rmsnorm-qk.rules.json': rmsnormQkRules,
  'kernels/rope-qk.rules.json': ropeQkRules,
  'kernels/rope.rules.json': ropeRules,
  'kernels/linear-attention.rules.json': linearAttentionRules,
  'kernels/sample.rules.json': sampleRules,
  'kernels/scale.rules.json': scaleRules,
  'kernels/silu.rules.json': siluRules,
  'kernels/split-qkv.rules.json': splitQkvRules,
  'kernels/split-qg.rules.json': splitQgRules,
  'kernels/softmax.rules.json': softmaxRules,
  'kernels/upsample2d.rules.json': upsample2dRules,
  'kernels/vision-patch-embed.rules.json': visionPatchEmbedRules,
  'kernels/vision-spatial-merge.rules.json': visionSpatialMergeRules,
  'kernels/vision-rope-2d.rules.json': visionRope2dRules,
  'kernels/vision-average-pool.rules.json': visionAveragePoolRules,
  'kernels/vision-position-embedding.rules.json': visionPositionEmbeddingRules,
  'inference/config.rules.json': configRules,
  'inference/execution.rules.json': inferenceExecutionRules,
  'inference/attention.rules.json': inferenceAttentionRules,
  'inference/dtype.rules.json': dtypeRules,
  'inference/ffn.rules.json': ffnRules,
  'inference/layer.rules.json': layerRules,
  'inference/layer-pattern.rules.json': layerPatternRules,
  'inference/moe.rules.json': inferenceMoeRules,
  'converter/tokenizer.rules.json': tokenizerRules,
  'converter/tensor-roles.rules.json': tensorRolesRules,
  'converter/execution.rules.json': converterExecutionRules,
  'loader/weights.rules.json': loaderWeightRules,
  'loader/tensor-loader.rules.json': tensorLoaderRules,
  'tooling/command-runtime.rules.json': toolingCommandRuntimeRules,
} = ruleBundle.files;


function deepFreeze(value, seen = new WeakSet()) {
  if (!value || typeof value !== 'object' || seen.has(value)) {
    return value;
  }
  if (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null) {
    throw new Error('RuleRegistry: rules must contain only JSON values.');
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
    lmHeadArgmax: lmHeadArgmaxRules,
    matmul: matmulRules,
    moe: kernelMoeRules,
    moeGptoss: kernelMoeGptOssRules,
    moeMixtral: kernelMoeMixtralRules,
    modulate: modulateRules,
    pixel_shuffle: pixelShuffleRules,
    repeatChannels: repeatChannelsRules,
    repPenalty: repPenaltyRules,
    relu: reluRules,
    residual: residualRules,
    rmsnorm: rmsnormRules,
    rmsnormQk: rmsnormQkRules,
    ropeQk: ropeQkRules,
    rope: ropeRules,
    linearAttention: linearAttentionRules,
    sample: sampleRules,
    scale: scaleRules,
    silu: siluRules,
    splitQkv: splitQkvRules,
    splitQg: splitQgRules,
    softmax: softmaxRules,
    upsample2d: upsample2dRules,
    visionPatchEmbed: visionPatchEmbedRules,
    visionSpatialMerge: visionSpatialMergeRules,
    visionRope2d: visionRope2dRules,
    visionAveragePool: visionAveragePoolRules,
    visionPositionEmbedding: visionPositionEmbeddingRules,
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

const instances = new WeakSet();
export function isRuleRegistry(value) { return instances.has(value); }

export function createRuleRegistry({ extensions = [], base = null } = {}) {
  if (base !== null && !instances.has(base)) throw new Error('RuleRegistry: invalid base instance.');
  const sets = cloneRuleValue(base ? base.ruleSets : RULE_SETS);
  for (const { domain, group, rules } of extensions) {
    for (const key of [domain, group]) {
      if (typeof key !== 'string' || !key || ['__proto__', 'constructor', 'prototype'].includes(key)) {
        throw new Error('RuleRegistry: extension requires safe domain and group names.');
      }
    }
    if (!rules || typeof rules !== 'object' || Array.isArray(rules)) {
      throw new Error('RuleRegistry: extension rules must be a JSON rule group.');
    }
    sets[domain] = { ...sets[domain], [group]: cloneRuleValue(rules) };
  }
  const identity = computeCanonicalSha256(sets);
  deepFreeze(sets);
  const instance = Object.freeze({
    identity,
    ruleSets: sets,
    getRuleSet(domain, group, name) {
      if (!Object.hasOwn(sets, domain)) throw new Error(`RuleRegistry: unknown domain "${domain}".`);
      if (!Object.hasOwn(sets[domain], group)) throw new Error(`RuleRegistry: unknown rule group "${domain}.${group}".`);
      const rules = sets[domain][group];
      if (!Object.hasOwn(rules, name)) throw new Error(`RuleRegistry: unknown rule set "${domain}.${group}.${name}".`);
      return rules[name];
    },
    selectRuleValue(domain, group, name, context) {
      return resolveRuleValue(selectByRules(instance.getRuleSet(domain, group, name), context), context);
    },
  });
  instances.add(instance);
  return instance;
}

export const DEFAULT_RULE_REGISTRY = createRuleRegistry();
let compatibilityRegistry = DEFAULT_RULE_REGISTRY;
let activeRegistry = null;

export function getDefaultRuleRegistry() { return compatibilityRegistry; }
export function getRuleRegistry() { return activeRegistry ?? compatibilityRegistry; }

// Only the serialized legacy pipeline boundary enters this adapter.
export function enterRuleRegistry(registry) {
  if (!instances.has(registry)) throw new Error('RuleRegistry: expected a constructed registry instance.');
  const previous = activeRegistry;
  activeRegistry = registry;
  return () => { activeRegistry = previous; };
}

export function getRuleSet(domain, group, name) {
  return getRuleRegistry().getRuleSet(domain, group, name);
}

export function selectRuleValue(domain, group, name, context) {
  return getRuleRegistry().selectRuleValue(domain, group, name, context);
}

// Compatibility registration replaces the default for future construction only.
export function registerRuleGroup(domain, group, rules) {
  compatibilityRegistry = createRuleRegistry({ base: compatibilityRegistry, extensions: [{ domain, group, rules }] });
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
