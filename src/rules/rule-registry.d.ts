import type { Rule } from '../gpu/kernels/rule-matcher.js';

type RuleSet = Array<Rule<unknown>>;

type RuleDomain = 'kernels' | 'inference' | 'shared' | 'loader' | 'converter';

type KernelRuleGroup =
  | 'attention'
  | 'dequant'
  | 'fusedFfn'
  | 'fusedMatmulResidual'
  | 'fusedMatmulRmsnorm'
  | 'gather'
  | 'gelu'
  | 'matmul'
  | 'moe'
  | 'residual'
  | 'rmsnorm'
  | 'rope'
  | 'sample'
  | 'scale'
  | 'silu'
  | 'splitQkv'
  | 'softmax';

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
