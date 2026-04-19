export declare function resolveDeviceInfo(): Record<string, unknown> | null;
export declare function resolveKernelPathForModel(options?: Record<string, unknown>): Promise<{
  modelId: string | null;
  kernelPath: unknown;
  source: string | null;
} | null>;
export declare function resolveLocalSourceRuntimePathFromModelUrl(
  modelUrl?: string | null
): Promise<string | null>;
export declare function initializeSuiteModel(options?: Record<string, unknown>): Promise<Record<string, unknown>>;
