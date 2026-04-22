export declare function chooseNullish<T>(
  overrideValue: T | null | undefined,
  fallbackValue: T
): T;

export declare function chooseDefined<T>(
  overrideValue: T | undefined,
  fallbackValue: T
): T;

export declare function chooseDefinedWithSource<T>(
  path: string,
  overrideValue: T | undefined,
  fallbackValue: T,
  sources: Map<string, string> | null | undefined
): T;

export declare function mergeShallowObject<T extends object>(
  base: T,
  override: Partial<T> | null | undefined
): T;

export declare function mergeLayeredShallowObjects<T extends object>(
  ...layers: Array<Partial<T> | null | undefined>
): T;

export declare function replaceSubtree<T>(
  overrideValue: T | null | undefined,
  fallbackValue: T
): T | null;

export declare function mergeKernelPathPolicy<T extends {
  mode?: unknown;
  sourceScope?: unknown;
  allowSources?: unknown;
  onIncompatible?: unknown;
}>(
  basePolicy: T | null | undefined,
  overridePolicy: T | null | undefined
): {
  mode: unknown;
  sourceScope: unknown;
  allowSources: unknown;
  onIncompatible: unknown;
};

export declare function mergeExecutionPatchLists<T extends {
  set?: unknown;
  remove?: unknown;
  add?: unknown;
}>(
  basePatch: T | null | undefined,
  overridePatch: T | null | undefined
): {
  set: unknown;
  remove: unknown;
  add: unknown;
};
