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
      };
}

export interface Rule<T> {
  match: RuleMatch;
  value: T;
}

export declare function matchesRule(match: RuleMatch | null | undefined, context: Record<string, unknown>): boolean;

export declare function selectByRules<T>(
  rules: Array<Rule<T>>,
  context: Record<string, unknown>,
  fallback: T
): T;
