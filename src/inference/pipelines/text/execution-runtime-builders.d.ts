export declare const PIPELINE_COMPATIBLE_OPS: ReadonlySet<string>;

export declare function normalizeDtype(value: unknown, label: string): 'f16' | 'f32';
export declare function isPhaseMatch(phase: string, targetPhase: string): boolean;
export declare function stepHasLayer(step: { layers: 'all' | number[] }, layerIdx: number): boolean;
export declare function requireSessionActivationDtype(
  sessionDefaults: Record<string, unknown> | null | undefined,
  label?: string
): 'f16' | 'f32';

export declare function resolveFinitenessFallbackKernelPathId(
  defaultKernelPathId: string | null | undefined
): string | null;

export declare function buildInlineKernelPath(
  steps: readonly Record<string, unknown>[],
  sessionDefaults: Record<string, unknown> | null,
  modelId: string,
  numLayers: number,
  finitenessFallbackKernelPathId?: string | null
): Record<string, unknown> | null;

export declare function buildLayerPipelineFromExecution(
  steps: readonly Record<string, unknown>[]
): { steps: Record<string, unknown>[]; overrides: unknown[] } | { incompatibleOps: string[] } | null;

export declare function buildSessionRuntimePatch(
  sessionDefaults: Record<string, unknown> | null | undefined
): Record<string, unknown>;
