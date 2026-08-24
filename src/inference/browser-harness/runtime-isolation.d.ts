export declare function cloneRuntimeConfig(
  runtimeConfig: Record<string, unknown> | null | undefined
): Record<string, unknown> | null;

export declare function snapshotRuntimeState(): {
  runtimeConfig: Record<string, unknown> | null;
  activeKernelPath: unknown;
  activeKernelPathSource: string | null;
  activeKernelPathPolicy: unknown;
};

export declare function restoreRuntimeState(snapshot: Record<string, unknown> | null | undefined): void;
export declare function runWithRuntimeIsolationForSuite<T>(run: () => Promise<T>): Promise<T>;
