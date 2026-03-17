import type { Rule } from '../gpu/kernels/rule-matcher.js';

type RuleSet = Array<Rule<unknown>>;

type RuleDomain = 'kernels' | 'inference' | 'shared' | 'loader' | 'converter' | 'tooling';

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
  | 'moeMixtral'
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

export declare function getInferenceExecutionRulesContractArtifact(): {
  schemaVersion: 1;
  source: 'doppler';
  ok: boolean;
  checks: Array<{ id: string; ok: boolean }>;
  errors: string[];
  stats: {
    decodeRecorderRules: number;
    batchDecodeRules: number;
    decodeRecorderContexts: number;
    batchDecodeContexts: number;
  };
};

export declare function getInferenceLayerPatternContractArtifact(): {
  schemaVersion: 1;
  source: 'doppler';
  ok: boolean;
  checks: Array<{ id: string; ok: boolean }>;
  errors: string[];
  stats: {
    patternKindRules: number;
    layerTypeRules: number;
    patternKindContexts: number;
    layerTypeContexts: number;
  };
};
