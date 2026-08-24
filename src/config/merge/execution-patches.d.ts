export declare function mergeExecutionPatchLists<T extends {
  addKernels?: unknown;
  set?: unknown;
  remove?: unknown;
  add?: unknown;
}>(
  basePatch: T | null | undefined,
  overridePatch: T | null | undefined
): {
  addKernels: unknown;
  set: unknown;
  remove: unknown;
  add: unknown;
};
