export declare function resolveDeviceInfo(): Record<string, unknown> | null;
export declare function resolveKernelPathForModel(options?: Record<string, unknown>): Promise<{
  modelId: string | null;
  kernelPath: unknown;
  source: string | null;
} | null>;
export declare function initializeInferenceFromStorage(
  modelId: string,
  options?: Record<string, unknown>
): Promise<Record<string, unknown>>;
export declare function initializeInferenceFromSourcePath(
  sourcePath: string,
  options?: Record<string, unknown>
): Promise<Record<string, unknown>>;
export declare function resolveHarnessOverride(options?: Record<string, unknown>): Promise<Record<string, unknown>>;
export declare function initializeSuiteModel(options?: Record<string, unknown>): Promise<Record<string, unknown>>;
