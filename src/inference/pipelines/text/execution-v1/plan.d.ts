export declare function resolveRuntimeInferenceOverrideSection(
  runtimeOverrides: Record<string, unknown> | null | undefined,
  key: string
): unknown;

export declare function preserveRuntimeDecodeLoop(
  updatedInference: Record<string, unknown>,
  runtimeConfig: Record<string, unknown> | null | undefined
): Record<string, unknown>;

export declare function preserveConfiguredKernelPath(
  updatedInference: Record<string, unknown>,
  runtimeConfig: Record<string, unknown> | null | undefined
): Record<string, unknown>;
