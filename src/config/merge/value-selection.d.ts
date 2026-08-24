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
