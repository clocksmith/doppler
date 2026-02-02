import { selectByRules } from '../gpu/kernels/rule-matcher.js';
const loadJson = async (path) => {
  const response = await fetch(new URL(path, import.meta.url));
  if (!response.ok) throw new Error(`Failed to load rules: ${path}`);
  return response.json();
};

const attentionRules = await loadJson('./kernels/attention.rules.json');
const conv2dRules = await loadJson('./kernels/conv2d.rules.json');
const dequantRules = await loadJson('./kernels/dequant.rules.json');
const fusedFfnRules = await loadJson('./kernels/fused-ffn.rules.json');
const fusedMatmulResidualRules = await loadJson('./kernels/fused-matmul-residual.rules.json');
const fusedMatmulRmsnormRules = await loadJson('./kernels/fused-matmul-rmsnorm.rules.json');
const gatherRules = await loadJson('./kernels/gather.rules.json');
const geluRules = await loadJson('./kernels/gelu.rules.json');
const groupnormRules = await loadJson('./kernels/groupnorm.rules.json');
const kvQuantizeRules = await loadJson('./kernels/kv_quantize.rules.json');
const layernormRules = await loadJson('./kernels/layernorm.rules.json');
const matmulRules = await loadJson('./kernels/matmul.rules.json');
const kernelMoeRules = await loadJson('./kernels/moe.rules.json');
const residualRules = await loadJson('./kernels/residual.rules.json');
const rmsnormRules = await loadJson('./kernels/rmsnorm.rules.json');
const ropeRules = await loadJson('./kernels/rope.rules.json');
const sampleRules = await loadJson('./kernels/sample.rules.json');
const scaleRules = await loadJson('./kernels/scale.rules.json');
const siluRules = await loadJson('./kernels/silu.rules.json');
const splitQkvRules = await loadJson('./kernels/split-qkv.rules.json');
const softmaxRules = await loadJson('./kernels/softmax.rules.json');
const upsample2dRules = await loadJson('./kernels/upsample2d.rules.json');
const configRules = await loadJson('./inference/config.rules.json');
const inferenceAttentionRules = await loadJson('./inference/attention.rules.json');
const dtypeRules = await loadJson('./inference/dtype.rules.json');
const ffnRules = await loadJson('./inference/ffn.rules.json');
const layerRules = await loadJson('./inference/layer.rules.json');
const layerPatternRules = await loadJson('./inference/layer-pattern.rules.json');
const inferenceMoeRules = await loadJson('./inference/moe.rules.json');
const tokenizerRules = await loadJson('./converter/tokenizer.rules.json');
const tensorRolesRules = await loadJson('./converter/tensor-roles.rules.json');
const loaderWeightRules = await loadJson('./loader/weights.rules.json');
const tensorLoaderRules = await loadJson('./loader/tensor-loader.rules.json');

const RULE_SETS = {
  shared: {
    dtype: dtypeRules,
  },
  kernels: {
    attention: attentionRules,
    conv2d: conv2dRules,
    dequant: dequantRules,
    fusedFfn: fusedFfnRules,
    fusedMatmulResidual: fusedMatmulResidualRules,
    fusedMatmulRmsnorm: fusedMatmulRmsnormRules,
    gather: gatherRules,
    gelu: geluRules,
    groupnorm: groupnormRules,
    kv_quantize: kvQuantizeRules,
    layernorm: layernormRules,
    matmul: matmulRules,
    moe: kernelMoeRules,
    residual: residualRules,
    rmsnorm: rmsnormRules,
    rope: ropeRules,
    sample: sampleRules,
    scale: scaleRules,
    silu: siluRules,
    splitQkv: splitQkvRules,
    softmax: softmaxRules,
    upsample2d: upsample2dRules,
  },
  inference: {
    config: configRules,
    attention: inferenceAttentionRules,
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
