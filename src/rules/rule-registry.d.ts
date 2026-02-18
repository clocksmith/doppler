import type { Rule } from '../gpu/kernels/rule-matcher.js';

type RuleSet = Array<Rule<unknown>>;

type RuleDomain = 'kernels' | 'inference' | 'shared' | 'loader' | 'converter';

type KernelRuleGroup =
  | 'attention'
  | 'conv2d'
  | 'dequant'
  | 'energy'
  | 'fusedFfn'
  | 'fusedMatmulResidual'
  | 'fusedMatmulRmsnorm'
  | 'gather'
  | 'gelu'
  | 'groupnorm'
  | 'kv_quantize'
  | 'layernorm'
  | 'matmul'
  | 'moe'
  | 'moeGptoss'
  | 'residual'
  | 'rmsnorm'
  | 'rope'
  | 'sample'
  | 'scale'
  | 'silu'
  | 'splitQkv'
  | 'softmax'
  | 'upsample2d';

type RuleGroup = KernelRuleGroup | string;

export declare function getRuleSet(domain: RuleDomain, group: RuleGroup, name: string): RuleSet;

export declare function selectRuleValue<T>(
  domain: RuleDomain,
  group: RuleGroup,
  name: string,
  context: Record<string, unknown>
): T;

export declare function registerRuleGroup(
  domain: RuleDomain,
  group: RuleGroup,
  rules: Record<string, RuleSet>
): void;
