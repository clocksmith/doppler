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
