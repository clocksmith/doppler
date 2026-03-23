import { selectByRules } from '../gpu/kernels/rule-matcher.js';
import { buildInferenceExecutionRulesContractArtifact } from './execution-rules-contract-check.js';
import { buildLayerPatternContractArtifact } from './layer-pattern-contract-check.js';
import { loadJson } from '../utils/load-json.js';

function cloneRuleValue(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

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

const ruleLoadFailures = [];

async function safeLoadJson(path, label) {
  try {
    return await loadJson(path, import.meta.url, 'Failed to load rules');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`RuleRegistry: failed to load "${path}": ${message}`);
    ruleLoadFailures.push(path);
    return null;
  }
}

const attentionRules = await safeLoadJson('./kernels/attention.rules.json');
const conv2dRules = await safeLoadJson('./kernels/conv2d.rules.json');
const depthwiseConv2dRules = await safeLoadJson('./kernels/depthwise-conv2d.rules.json');
const dequantRules = await safeLoadJson('./kernels/dequant.rules.json');
const energyRules = await safeLoadJson('./kernels/energy.rules.json');
const fusedFfnRules = await safeLoadJson('./kernels/fused-ffn.rules.json');
const fusedMatmulResidualRules = await safeLoadJson('./kernels/fused-matmul-residual.rules.json');
const fusedMatmulRmsnormRules = await safeLoadJson('./kernels/fused-matmul-rmsnorm.rules.json');
const gatherRules = await safeLoadJson('./kernels/gather.rules.json');
const geluRules = await safeLoadJson('./kernels/gelu.rules.json');
const groupedPointwiseConv2dRules = await safeLoadJson('./kernels/grouped-pointwise-conv2d.rules.json');
const groupnormRules = await safeLoadJson('./kernels/groupnorm.rules.json');
const kvQuantizeRules = await safeLoadJson('./kernels/kv_quantize.rules.json');
const layernormRules = await safeLoadJson('./kernels/layernorm.rules.json');
const matmulRules = await safeLoadJson('./kernels/matmul.rules.json');
const kernelMoeRules = await safeLoadJson('./kernels/moe.rules.json');
const kernelMoeGptOssRules = await safeLoadJson('./kernels/moe.rules.gptoss.json');
const kernelMoeMixtralRules = await safeLoadJson('./kernels/moe.rules.mixtral.json');
const modulateRules = await safeLoadJson('./kernels/modulate.rules.json');
const pixelShuffleRules = await safeLoadJson('./kernels/pixel_shuffle.rules.json');
const repeatChannelsRules = await safeLoadJson('./kernels/repeat-channels.rules.json');
const reluRules = await safeLoadJson('./kernels/relu.rules.json');
const residualRules = await safeLoadJson('./kernels/residual.rules.json');
const rmsnormRules = await safeLoadJson('./kernels/rmsnorm.rules.json');
const ropeRules = await safeLoadJson('./kernels/rope.rules.json');
const sanaLinearAttentionRules = await safeLoadJson('./kernels/sana-linear-attention.rules.json');
const sampleRules = await safeLoadJson('./kernels/sample.rules.json');
const scaleRules = await safeLoadJson('./kernels/scale.rules.json');
const siluRules = await safeLoadJson('./kernels/silu.rules.json');
const splitQkvRules = await safeLoadJson('./kernels/split-qkv.rules.json');
const splitQgRules = await safeLoadJson('./kernels/split-qg.rules.json');
const softmaxRules = await safeLoadJson('./kernels/softmax.rules.json');
const upsample2dRules = await safeLoadJson('./kernels/upsample2d.rules.json');
const configRules = await safeLoadJson('./inference/config.rules.json');
const inferenceExecutionRules = await safeLoadJson('./inference/execution.rules.json');
const inferenceAttentionRules = await safeLoadJson('./inference/attention.rules.json');
const dtypeRules = await safeLoadJson('./inference/dtype.rules.json');
const ffnRules = await safeLoadJson('./inference/ffn.rules.json');
const layerRules = await safeLoadJson('./inference/layer.rules.json');
const layerPatternRules = await safeLoadJson('./inference/layer-pattern.rules.json');
const inferenceMoeRules = await safeLoadJson('./inference/moe.rules.json');
const tokenizerRules = await safeLoadJson('./converter/tokenizer.rules.json');
const tensorRolesRules = await safeLoadJson('./converter/tensor-roles.rules.json');
const converterExecutionRules = await safeLoadJson('./converter/execution.rules.json');
const loaderWeightRules = await safeLoadJson('./loader/weights.rules.json');
const tensorLoaderRules = await safeLoadJson('./loader/tensor-loader.rules.json');
const toolingCommandRuntimeRules = await safeLoadJson('./tooling/command-runtime.rules.json');

if (ruleLoadFailures.length > 0) {
  console.warn(
    `RuleRegistry: ${ruleLoadFailures.length} rule file(s) failed to load: ${ruleLoadFailures.join(', ')}`
  );
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
    sanaLinearAttention: sanaLinearAttentionRules,
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
