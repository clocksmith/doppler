export declare function getRuleSet(group: string, name: string): import('../../rules/rule-registry.js').ResolvedRuleSet;

export declare function selectRuleValue<T>(
  group: string,
  name: string,
  context: Record<string, unknown>
): T;
