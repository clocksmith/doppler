import { selectByRules } from '../gpu/kernels/rule-matcher.js';
import { buildInferenceExecutionRulesContractArtifact } from './execution-rules-contract-check.js';
import { buildLayerPatternContractArtifact } from './layer-pattern-contract-check.js';
import { loadJson } from '../utils/load-json.js';

const attentionRules = await loadJson('./kernels/attention.rules.json', import.meta.url, 'Failed to load rules');
const conv2dRules = await loadJson('./kernels/conv2d.rules.json', import.meta.url, 'Failed to load rules');
const depthwiseConv2dRules = await loadJson('./kernels/depthwise-conv2d.rules.json', import.meta.url, 'Failed to load rules');
const dequantRules = await loadJson('./kernels/dequant.rules.json', import.meta.url, 'Failed to load rules');
const energyRules = await loadJson('./kernels/energy.rules.json', import.meta.url, 'Failed to load rules');
const fusedFfnRules = await loadJson('./kernels/fused-ffn.rules.json', import.meta.url, 'Failed to load rules');
const fusedMatmulResidualRules = await loadJson('./kernels/fused-matmul-residual.rules.json', import.meta.url, 'Failed to load rules');
const fusedMatmulRmsnormRules = await loadJson('./kernels/fused-matmul-rmsnorm.rules.json', import.meta.url, 'Failed to load rules');
const gatherRules = await loadJson('./kernels/gather.rules.json', import.meta.url, 'Failed to load rules');
const geluRules = await loadJson('./kernels/gelu.rules.json', import.meta.url, 'Failed to load rules');
const groupedPointwiseConv2dRules = await loadJson('./kernels/grouped-pointwise-conv2d.rules.json', import.meta.url, 'Failed to load rules');
const groupnormRules = await loadJson('./kernels/groupnorm.rules.json', import.meta.url, 'Failed to load rules');
const kvQuantizeRules = await loadJson('./kernels/kv_quantize.rules.json', import.meta.url, 'Failed to load rules');
const layernormRules = await loadJson('./kernels/layernorm.rules.json', import.meta.url, 'Failed to load rules');
const matmulRules = await loadJson('./kernels/matmul.rules.json', import.meta.url, 'Failed to load rules');
const kernelMoeRules = await loadJson('./kernels/moe.rules.json', import.meta.url, 'Failed to load rules');
const kernelMoeGptOssRules = await loadJson('./kernels/moe.rules.gptoss.json', import.meta.url, 'Failed to load rules');
const modulateRules = await loadJson('./kernels/modulate.rules.json', import.meta.url, 'Failed to load rules');
const pixelShuffleRules = await loadJson('./kernels/pixel_shuffle.rules.json', import.meta.url, 'Failed to load rules');
const residualRules = await loadJson('./kernels/residual.rules.json', import.meta.url, 'Failed to load rules');
const rmsnormRules = await loadJson('./kernels/rmsnorm.rules.json', import.meta.url, 'Failed to load rules');
const ropeRules = await loadJson('./kernels/rope.rules.json', import.meta.url, 'Failed to load rules');
const sanaLinearAttentionRules = await loadJson('./kernels/sana-linear-attention.rules.json', import.meta.url, 'Failed to load rules');
const sampleRules = await loadJson('./kernels/sample.rules.json', import.meta.url, 'Failed to load rules');
const scaleRules = await loadJson('./kernels/scale.rules.json', import.meta.url, 'Failed to load rules');
const siluRules = await loadJson('./kernels/silu.rules.json', import.meta.url, 'Failed to load rules');
const splitQkvRules = await loadJson('./kernels/split-qkv.rules.json', import.meta.url, 'Failed to load rules');
const softmaxRules = await loadJson('./kernels/softmax.rules.json', import.meta.url, 'Failed to load rules');
const upsample2dRules = await loadJson('./kernels/upsample2d.rules.json', import.meta.url, 'Failed to load rules');
const configRules = await loadJson('./inference/config.rules.json', import.meta.url, 'Failed to load rules');
const inferenceExecutionRules = await loadJson('./inference/execution.rules.json', import.meta.url, 'Failed to load rules');
const inferenceAttentionRules = await loadJson('./inference/attention.rules.json', import.meta.url, 'Failed to load rules');
const dtypeRules = await loadJson('./inference/dtype.rules.json', import.meta.url, 'Failed to load rules');
const ffnRules = await loadJson('./inference/ffn.rules.json', import.meta.url, 'Failed to load rules');
const inferenceKernelPathRules = await loadJson('./inference/kernel-path.rules.json', import.meta.url, 'Failed to load rules');
const layerRules = await loadJson('./inference/layer.rules.json', import.meta.url, 'Failed to load rules');
const layerPatternRules = await loadJson('./inference/layer-pattern.rules.json', import.meta.url, 'Failed to load rules');
const inferenceMoeRules = await loadJson('./inference/moe.rules.json', import.meta.url, 'Failed to load rules');
const tokenizerRules = await loadJson('./converter/tokenizer.rules.json', import.meta.url, 'Failed to load rules');
const tensorRolesRules = await loadJson('./converter/tensor-roles.rules.json', import.meta.url, 'Failed to load rules');
const converterExecutionRules = await loadJson('./converter/execution.rules.json', import.meta.url, 'Failed to load rules');
const loaderWeightRules = await loadJson('./loader/weights.rules.json', import.meta.url, 'Failed to load rules');
const tensorLoaderRules = await loadJson('./loader/tensor-loader.rules.json', import.meta.url, 'Failed to load rules');
const toolingCommandRuntimeRules = await loadJson(
  './tooling/command-runtime.rules.json',
  import.meta.url,
  'Failed to load rules'
);
const INFERENCE_EXECUTION_RULES_CONTRACT_ARTIFACT = buildInferenceExecutionRulesContractArtifact(
  inferenceExecutionRules
);
if (!INFERENCE_EXECUTION_RULES_CONTRACT_ARTIFACT.ok) {
  throw new Error(
    `RuleRegistry: inference.execution rules contract failed: ` +
    `${INFERENCE_EXECUTION_RULES_CONTRACT_ARTIFACT.errors.join(' | ')}`
  );
}
const INFERENCE_LAYER_PATTERN_CONTRACT_ARTIFACT = buildLayerPatternContractArtifact(
  layerPatternRules
);
if (!INFERENCE_LAYER_PATTERN_CONTRACT_ARTIFACT.ok) {
  throw new Error(
    `RuleRegistry: inference.layerPattern rules contract failed: ` +
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
    modulate: modulateRules,
    pixel_shuffle: pixelShuffleRules,
    residual: residualRules,
    rmsnorm: rmsnormRules,
    rope: ropeRules,
    sanaLinearAttention: sanaLinearAttentionRules,
    sample: sampleRules,
    scale: scaleRules,
    silu: siluRules,
    splitQkv: splitQkvRules,
    softmax: softmaxRules,
    upsample2d: upsample2dRules,
  },
  inference: {
    config: configRules,
    execution: inferenceExecutionRules,
    attention: inferenceAttentionRules,
    dtype: dtypeRules,
    ffn: ffnRules,
    kernelPath: inferenceKernelPathRules,
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
  RULE_SETS[domain][group] = rules;
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
