import { selectByRules } from '../gpu/kernels/rule-matcher.js';
import attentionRules from './kernels/attention.rules.json' with { type: 'json' };
import dequantRules from './kernels/dequant.rules.json' with { type: 'json' };
import fusedFfnRules from './kernels/fused-ffn.rules.json' with { type: 'json' };
import fusedMatmulResidualRules from './kernels/fused-matmul-residual.rules.json' with { type: 'json' };
import fusedMatmulRmsnormRules from './kernels/fused-matmul-rmsnorm.rules.json' with { type: 'json' };
import gatherRules from './kernels/gather.rules.json' with { type: 'json' };
import geluRules from './kernels/gelu.rules.json' with { type: 'json' };
import matmulRules from './kernels/matmul.rules.json' with { type: 'json' };
import kernelMoeRules from './kernels/moe.rules.json' with { type: 'json' };
import residualRules from './kernels/residual.rules.json' with { type: 'json' };
import rmsnormRules from './kernels/rmsnorm.rules.json' with { type: 'json' };
import ropeRules from './kernels/rope.rules.json' with { type: 'json' };
import sampleRules from './kernels/sample.rules.json' with { type: 'json' };
import scaleRules from './kernels/scale.rules.json' with { type: 'json' };
import siluRules from './kernels/silu.rules.json' with { type: 'json' };
import splitQkvRules from './kernels/split-qkv.rules.json' with { type: 'json' };
import softmaxRules from './kernels/softmax.rules.json' with { type: 'json' };
import configRules from './inference/config.rules.json' with { type: 'json' };
import inferenceAttentionRules from './inference/attention.rules.json' with { type: 'json' };
import dtypeRules from './inference/dtype.rules.json' with { type: 'json' };
import ffnRules from './inference/ffn.rules.json' with { type: 'json' };
import layerRules from './inference/layer.rules.json' with { type: 'json' };
import layerPatternRules from './inference/layer-pattern.rules.json' with { type: 'json' };
import inferenceMoeRules from './inference/moe.rules.json' with { type: 'json' };
import tokenizerRules from './converter/tokenizer.rules.json' with { type: 'json' };
import tensorRolesRules from './converter/tensor-roles.rules.json' with { type: 'json' };
import loaderWeightRules from './loader/weights.rules.json' with { type: 'json' };
import tensorLoaderRules from './loader/tensor-loader.rules.json' with { type: 'json' };

const RULE_SETS = {
  shared: {
    dtype: dtypeRules,
  },
  kernels: {
    attention: attentionRules,
    dequant: dequantRules,
    fusedFfn: fusedFfnRules,
    fusedMatmulResidual: fusedMatmulResidualRules,
    fusedMatmulRmsnorm: fusedMatmulRmsnormRules,
    gather: gatherRules,
    gelu: geluRules,
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
