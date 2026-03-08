export declare function resolveFinitenessFallbackKernelPathId(defaultKernelPathId: string | null): string | null;
export declare function buildInlineKernelPath(
  steps: Array<Record<string, unknown>>,
  sessionDefaults: Record<string, unknown>,
  modelId: string,
  numLayers: number,
  finitenessFallbackKernelPathId?: string | null
): Record<string, unknown> | null;
export declare function buildLayerPipelineFromExecution(
  steps: Array<Record<string, unknown>>
): Record<string, unknown> | null;
export declare function buildSessionRuntimePatch(sessionDefaults: Record<string, unknown>): Record<string, unknown>;
export declare function buildModelRuntimeOverrides(
  manifestInference: Record<string, unknown> | null | undefined
): Record<string, unknown> | null;
