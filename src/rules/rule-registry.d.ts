import type { Rule } from './rule-matcher.js';

type RuleSet = Array<Rule<unknown>>;
type DeepReadonly<T> = T extends readonly unknown[] ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
  : T extends object ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
  : T;
export type ResolvedRuleSet = ReadonlyArray<DeepReadonly<Rule<unknown>>>;

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

export declare function getRuleSet(domain: RuleDomain, group: RuleGroup, name: string): ResolvedRuleSet;

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

export interface RuleRegistry {
  readonly identity: string;
  readonly ruleSets: Readonly<Record<string, Readonly<Record<string, Readonly<Record<string, ResolvedRuleSet>>>>>>;
  getRuleSet(domain: string, group: string, name: string): ResolvedRuleSet;
  selectRuleValue<T>(domain: string, group: string, name: string, context: Record<string, unknown>): T;
}
export declare function createRuleRegistry(options?: {
  base?: RuleRegistry | null;
  extensions?: Array<{ domain: string; group: string; rules: Record<string, RuleSet> }>;
}): RuleRegistry;
export declare const DEFAULT_RULE_REGISTRY: RuleRegistry;
export declare function isRuleRegistry(value: unknown): value is RuleRegistry;
export declare function getRuleRegistry(): RuleRegistry;
/** Internal compatibility lease; caller must serialize the entire operation. */
export declare function enterRuleRegistry(registry: RuleRegistry): () => void;

export declare function getDefaultRuleRegistry(): RuleRegistry;
