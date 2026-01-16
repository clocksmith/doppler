export declare function getRuleSet(group: string, name: string): Array<{ match: Record<string, unknown>; value: unknown }>;

export declare function selectRuleValue<T>(
  group: string,
  name: string,
  context: Record<string, unknown>
): T;
