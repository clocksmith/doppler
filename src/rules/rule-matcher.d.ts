export interface RuleMatch {
  [key: string]:
    | string
    | number
    | boolean
    | {
        eq?: string | number | boolean;
        neq?: string | number | boolean;
        gt?: number;
        gte?: number;
        lt?: number;
        lte?: number;
        in?: Array<string | number | boolean>;
        contains?: string | string[];
        startsWith?: string | string[];
        endsWith?: string | string[];
      };
}

export interface Rule<T> {
  match: RuleMatch;
  value: T;
}

export declare function matchesRule(match: RuleMatch | null | undefined, context: Record<string, unknown>): boolean;

export declare function selectByRules<T>(
  rules: Array<Rule<T>>,
  context: Record<string, unknown>
): T;

export interface RuleGroupStructureSpec {
  contextKeys: readonly string[];
  isValue(value: unknown): boolean;
  valueType: string;
  catchAll: boolean;
}

export declare function validateRuleGroupStructure(
  rules: unknown,
  spec: RuleGroupStructureSpec,
  label: string
): string[];

export interface RuleFileStructureArtifact {
  schemaVersion: 1;
  source: 'doppler';
  ok: boolean;
  checks: Array<{ id: string; ok: boolean }>;
  errors: string[];
  stats: Record<string, number>;
}

export declare function buildRuleFileStructureArtifact(
  ruleFile: unknown,
  specs: Record<string, RuleGroupStructureSpec>,
  options: {
    idPrefix: string;
    label: string;
    extraChecks?: Array<{ id: string; run(ruleFile: unknown): string[] }>;
  }
): RuleFileStructureArtifact;
